"""Check coordinates, native clipping/optimizer behavior and exact continuation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from contrast_trial import HERE, TAG, FAMILY, base, torch, build_model, anchored_builder
from sharing_trial import state_builder
from sharing_check import helpers


def kernels(dim: int) -> dict:
    torch.manual_seed(0)
    shared = state_builder(dim, 'state_midpoint4').double()
    rng = torch.get_rng_state()
    torch.manual_seed(0)
    model = build_model(dim, FAMILY).double()
    assert torch.equal(rng, torch.get_rng_state())
    common, current = dict(shared.named_parameters()), dict(model.named_parameters())
    assert model.parameter_count()-shared.parameter_count() == 73728
    for name, value in common.items():
        assert torch.equal(value, current[name]), name
    helpers.same(dict(shared.named_buffers()), dict(model.named_buffers()))
    with torch.no_grad():
        readout = torch.linspace(-.03, .04, shared.output.weight.numel(), dtype=torch.double).reshape_as(shared.output.weight)
        shared.output.weight.copy_(readout)
        model.output.weight.copy_(readout)
    x = torch.randn(8, dim, dtype=torch.double, requires_grad=True)
    t = x.new_full((len(x), 1), .3)
    left, right = shared(x, t), model(x, t)
    assert torch.equal(left, right)
    gx = torch.autograd.grad(left.square().sum(), x, retain_graph=True)[0]
    gy = torch.autograd.grad(right.square().sum(), x, retain_graph=True)[0]
    assert torch.allclose(gx, gy, atol=1e-14, rtol=1e-10)
    left.square().mean().backward()
    right.square().mean().backward()
    maximum = 0.
    for name, value in common.items():
        assert (value.grad is None) == (current[name].grad is None)
        if value.grad is not None:
            maximum = max(maximum, float((value.grad-current[name].grad).abs().max()))
            assert torch.allclose(value.grad, current[name].grad, atol=1e-14, rtol=1e-10), name
    for _, layer in model.ode_layers():
        assert torch.isfinite(layer.stage_delta.grad).all() and layer.stage_delta.grad.norm() > 0
    norms = [float(torch.sqrt(sum(p.grad.square().sum() for p in m.parameters() if p.grad is not None)))
             for m in [shared, model]]
    # Controlled, unclipped common-gradient comparison uses the real optimizer.
    opt_left = base.ProjectionOptimizer(shared, 'adamw', .0003)
    opt_right = base.ProjectionOptimizer(model, 'adamw', .0003)
    opt_left.step()
    opt_right.step()
    update_error = max(float((p-dict(model.named_parameters())[n]).detach().abs().max())
                       for n, p in base.projection_parameters(shared))
    assert update_error < 1e-13
    mean_error = max(float((l.stage_weights().mean(0)-l.operator()).detach().abs().max()) for _, l in model.ode_layers())
    assert mean_error < 1e-12
    # Nonzero contrasts can represent exactly the same four matrices as before.
    torch.manual_seed(0)
    anchored = anchored_builder(dim, 'stage_operator_midpoint4').double()
    torch.manual_seed(0)
    converted = build_model(dim, FAMILY).double()
    c = x.new_tensor([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])/2
    assert torch.equal(c.sum(0), torch.zeros(3, dtype=torch.double))
    assert torch.equal(c.T@c, torch.eye(3, dtype=torch.double))
    with torch.no_grad():
        anchored.output.weight.copy_(readout)
        converted.output.weight.copy_(readout)
        for (_, old), (_, new) in zip(anchored.ode_layers(), converted.ode_layers()):
            old.stage_delta.copy_(torch.linspace(-.02, .03, old.stage_delta.numel(), dtype=torch.double).reshape_as(old.stage_delta))
            weights = torch.cat([old.operator()[None], old.operator()[None]+old.stage_delta])
            new.proj.weight.copy_(weights.mean(0))
            new.stage_delta.copy_(torch.einsum('sk,sij->kij', c, weights))
            assert torch.allclose(new.stage_weights(), weights, atol=1e-14, rtol=1e-12)
    assert converted.parameter_count() == anchored.parameter_count()
    equivalent = float((converted(x, t)-anchored(x, t)).detach().abs().max())
    assert equivalent < 1e-12
    layer = next(converted.ode_layers())[1]
    h = torch.linspace(-1.1, 1.3, 64, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda v: layer(v[None])[0], (h,), fast_mode=True)
    # Check the native global clipping path separately; its factors need not match.
    for m in [anchored, converted]:
        m.zero_grad(set_to_none=True)
        (20*m(x, t)).square().mean().backward()
    clip_norms = [float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)) for m in [anchored, converted]]
    base.ProjectionOptimizer(converted, 'adamw', .0003).step()
    clipped_mean_error = max(float((l.stage_weights().mean(0)-l.operator()).detach().abs().max()) for _, l in converted.ode_layers())
    assert clipped_mean_error < 1e-12
    return dict(dim=dim, parameters=model.parameter_count(), extra_vs_shared=73728,
        initial_output_exact=True, input_gradient_max_error=float((gx-gy).abs().max()),
        common_gradient_max_error=maximum, all_contrast_gradients_active=True,
        controlled_unclipped_first_P_update_error=update_error,
        nonzero_coordinate_conversion_error=equivalent, formula_and_gradcheck=True,
        initial_shared_and_contrast_gradient_norms=norms,
        native_nonzero_contrast_anchored_and_contrast_gradient_norms=clip_norms,
        native_clip_factors=[min(1., 1/(n+1e-6)) for n in clip_norms],
        mean_invariance_after_native_clipped_step=clipped_mean_error)


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ['data-root', 'assets', 'output']:
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    base.verify_sources()
    rows = [kernels(d) for d in [2, 16]]
    base.dump(out/'kernel.json', dict(passed=True, rows=rows))
    grid = json.loads((HERE/'contrast-grid.json').read_text())
    for cell in grid['cells']:
        def folder(root: Path) -> Path:
            return root/TAG/cell['dataset']/cell['id']/'seed000'
        def run(root: Path, steps: int, label: str, resume=None) -> None:
            cmd = [sys.executable, str(HERE/'contrast_trial.py'), '--cell', cell['id'],
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
        cmd = [sys.executable, str(HERE/'contrast_evaluate.py'), '--folder', str(folder(out/'resume')),
            '--data-root', a.data_root, '--assets', a.assets, '--output', str(out/('evaluation-'+cell['dataset'])),
            '--device', 'cpu', '--steps', '16', '--witness']
        with (out/(cell['id']+'-eval.log')).open('x') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    base.dump(out/'PASS.json', dict(passed=True, kernels=rows, native_continuations=2,
        real_evaluator_witnesses=2, source_sha256={f.name: base.digest(f) for f in
            [Path(__file__), HERE/'contrast_trial.py', HERE/'contrast-grid.json', HERE/'contrast_evaluate.py']}))
    print(json.dumps(dict(passed=True, output=str(out))))


if __name__ == '__main__':
    main()
