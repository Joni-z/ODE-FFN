"""Independent composition prediction, native gradients, continuation and evaluation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import numpy as np
from scipy.linalg import expm
from schedule_trial import HERE, TAG, SCHEDULES, base, torch, build_model, state_builder
from sharing_check import helpers


def math_checks() -> dict:
    """Exact linear flows and native nonlinear maps test the registered sign."""
    p = np.array([[.2, -.3, .1], [.1, -.2, .4], [-.2, .1, .1]])
    b = np.array([[.1, .2, -.1], [-.3, .1, .1], [.2, -.1, .2]])
    x = np.array([.7, -1.1, .4])
    bracket = (b@p-p@b)@x
    horizons = [.25, .125, .0625, .03125, .015625]
    linear = []
    for tau in horizons:
        values = {}
        for family, signs in SCHEDULES.items():
            y = x.copy()
            for sign in signs:
                y = expm((tau/4)*(p+sign*b))@y
            values[family] = y
        pal, alt = [values[n] for n in SCHEDULES]
        linear.append(dict(tau=tau,
            predicted_difference=(-tau*tau*bracket/8).tolist(),
            actual_difference=(alt-pal).tolist(),
            remainder=float(np.linalg.norm(alt-pal+tau*tau*bracket/8)),
            palindrome_mean_gap=float(np.linalg.norm(pal-expm(tau*p)@x))))
    linear_order = float(np.log2(linear[-2]['remainder']/linear[-1]['remainder']))
    assert 2.8 < linear_order < 3.2, linear

    torch.manual_seed(0)
    models = [build_model(2, f).double() for f in SCHEDULES]
    layers = [next(m.ode_layers())[1] for m in models]
    gen = torch.Generator().manual_seed(0)
    p64 = torch.randn(64, 64, dtype=torch.double, generator=gen)*.3/8
    b64 = torch.randn(64, 64, dtype=torch.double, generator=gen)*.2/8
    h = torch.linspace(-1.1, 1.3, 64, dtype=torch.double)
    q = lambda v: v/torch.sqrt(1+1.2*v.square())
    f, g = lambda v: p64@q(v), lambda v: b64@q(v)
    jg_f = torch.autograd.functional.jvp(g, h, f(h))[1]
    jf_g = torch.autograd.functional.jvp(f, h, g(h))[1]
    derivative = (1+1.2*h.square()).pow(-1.5)
    exact_bracket = b64@(derivative*f(h))-p64@(derivative*g(h))
    jvp_error = float((jg_f-jf_g-exact_bracket).abs().max())
    assert jvp_error < 1e-13
    nonlinear, arrays = [], {}
    for tau in horizons:
        with torch.no_grad():
            for layer in layers:
                layer.proj.weight.copy_(tau*p64)
                layer.stage_offset.copy_(tau*b64)
            pal, alt = [layer(h[None])[0] for layer in layers]
            remainder = alt-pal+tau*tau*exact_bracket/8
            nonlinear.append(dict(tau=tau, remainder=float(remainder.norm()),
                difference_norm=float((alt-pal).norm())))
            arrays[str(tau)] = dict(palindrome=pal.tolist(), alternating=alt.tolist())
    nonlinear_order = float(np.log2(nonlinear[-2]['remainder']/nonlinear[-1]['remainder']))
    assert 2.8 < nonlinear_order < 3.2, nonlinear
    nulls = []
    for multiplier in [0., .4]:
        remainders = []
        for tau in horizons:
            with torch.no_grad():
                for layer in layers:
                    layer.proj.weight.copy_(tau*p64)
                    layer.stage_offset.copy_(tau*multiplier*p64)
                pal, alt = [layer(h[None])[0] for layer in layers]
            remainders.append(float((alt-pal).norm()))
        if multiplier == 0:
            assert max(remainders) == 0.
        else:
            assert remainders[-1] < remainders[-2]/6
        nulls.append(dict(B_multiple_of_P=multiplier, difference_norms=remainders))
    # Check a nonzero-B native map's input derivative, independent of the expansion.
    assert torch.autograd.gradcheck(lambda v: layers[0](v[None])[0],
                                    (h.requires_grad_(),), fast_mode=True)
    calls = []
    for steps in [4, 8, 16]:
        for layer in layers:
            with patch('torch.nn.functional.linear', wraps=torch.nn.functional.linear) as linear_call:
                layer.solve(h[None], steps)
                assert linear_call.call_count == 2*steps
            calls.append(dict(schedule=list(layer.schedule), substeps=steps, matrix_calls=2*steps))
    return dict(passed=True, linear_exact_exponential=linear, linear_remainder_order=linear_order,
        nonlinear_native=nonlinear, nonlinear_remainder_order=nonlinear_order,
        analytic_vs_jvp_max_error=jvp_error, nulls=nulls, actual_matrix_calls=calls,
        nonlinear_arrays=arrays, P=p64.tolist(), B=b64.tolist(), entrance=h.detach().tolist())


def model_checks(dim: int) -> dict:
    torch.manual_seed(0)
    shared = state_builder(dim, 'state_midpoint4').double()
    rng = torch.get_rng_state()
    variants = []
    for family in SCHEDULES:
        torch.manual_seed(0)
        model = build_model(dim, family).double()
        assert torch.equal(rng, torch.get_rng_state())
        assert model.parameter_count()-shared.parameter_count() == 24576
        for name, value in shared.state_dict().items():
            assert torch.equal(value, model.state_dict()[name]), name
        helpers.same(dict(shared.named_buffers()), dict(model.named_buffers()))
        variants.append(model)
    assert list(dict(variants[0].named_parameters())) == list(dict(variants[1].named_parameters()))
    with torch.no_grad():
        readout = torch.linspace(-.03, .04, shared.output.weight.numel(), dtype=torch.double).reshape_as(shared.output.weight)
        for model in [shared, *variants]:
            model.output.weight.copy_(readout)
    x = torch.randn(8, dim, dtype=torch.double, requires_grad=True)
    t = x.new_full((len(x), 1), .3)
    reference = shared(x, t)
    gx = torch.autograd.grad(reference.square().sum(), x, retain_graph=True)[0]
    reference.square().mean().backward()
    rows = []
    for family, model in zip(SCHEDULES, variants):
        y = model(x, t)
        assert torch.equal(reference, y)
        gy = torch.autograd.grad(y.square().sum(), x, retain_graph=True)[0]
        assert torch.allclose(gx, gy, atol=1e-14, rtol=1e-10)
        y.square().mean().backward()
        maximum = 0.
        for name, value in shared.named_parameters():
            other = dict(model.named_parameters())[name]
            assert (value.grad is None) == (other.grad is None)
            if value.grad is not None:
                maximum = max(maximum, float((value.grad-other.grad).abs().max()))
                assert torch.allclose(value.grad, other.grad, atol=1e-14, rtol=1e-10), name
        optimizer = base.ProjectionOptimizer(model, 'adamw', .0003)
        auxiliary = {id(p) for group in optimizer.aux.param_groups for p in group['params']}
        offsets = [layer.stage_offset for _, layer in model.ode_layers()]
        assert all(id(p) in auxiliary and p.grad is not None and torch.isfinite(p.grad).all() for p in offsets)
        rows.append(dict(family=family, parameters=model.parameter_count(),
            initial_output_exact=True, common_gradient_max_error=maximum,
            input_gradient_max_error=float((gx-gy).abs().max()),
            offset_gradient_norms=[float(p.grad.norm()) for p in offsets],
            full_gradient_norm=float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None))),
            offset_optimizer='native auxiliary AdamW lr0.0003 wd0',
            P_applications_per_local_solve=8))
    return dict(dim=dim, unchanged_initialization_rng=True, extra_parameters=24576, arms=rows)


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ['data-root', 'assets', 'output']:
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    base.verify_sources()
    base.dump(out/'math.json', math_checks())
    rows = [model_checks(dim) for dim in [2, 16]]
    base.dump(out/'kernel.json', dict(passed=True, rows=rows))
    for cell in json.loads((HERE/'schedule-grid.json').read_text())['cells']:
        def folder(root: Path) -> Path:
            return root/TAG/cell['dataset']/cell['id']/'seed000'
        def run(root: Path, steps: int, label: str, resume=None) -> None:
            cmd = [sys.executable, str(HERE/'schedule_trial.py'), '--cell', cell['id'],
                '--data-root', a.data_root, '--output', str(root), '--device', 'cpu', '--steps', str(steps)]
            if resume:
                cmd += ['--resume', str(resume)]
            with (out/(cell['id']+label+'.log')).open('x') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        run(out/'resume', 8, '-8')
        run(out/'resume', 16, '-resume16', folder(out/'resume')/'checkpoints/step_0000008.pt')
        run(out/'continuous', 16, '-continuous16')
        left = torch.load(folder(out/'resume')/'checkpoints/step_0000016.pt', weights_only=False)
        right = torch.load(folder(out/'continuous')/'checkpoints/step_0000016.pt', weights_only=False)
        helpers.numerical(left, right)
        cmd = [sys.executable, str(HERE/'schedule_evaluate.py'), '--folder', str(folder(out/'resume')),
            '--data-root', a.data_root, '--assets', a.assets, '--output', str(out/('evaluation-'+cell['id'])),
            '--device', 'cpu', '--steps', '16', '--witness']
        with (out/(cell['id']+'-eval.log')).open('x') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    base.dump(out/'PASS.json', dict(passed=True, kernels=rows, native_continuations=4,
        real_evaluator_witnesses=4, source_sha256={f.name: base.digest(f) for f in
            [Path(__file__), HERE/'schedule_trial.py', HERE/'schedule-grid.json',
             HERE/'schedule_evaluate.py', HERE/'schedule_verify.py']}))
    print(json.dumps(dict(passed=True, output=str(out))))


if __name__ == '__main__':
    main()
