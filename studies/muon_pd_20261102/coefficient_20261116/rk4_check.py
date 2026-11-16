"""Copied native continuation/evaluator checks plus RK4-specific kernel witnesses."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import numpy as np
from rk4_trial import HERE, TAG, base, torch, build_model, state_builder
from sharing_check import helpers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--assets', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    base.verify_sources()
    rows = []
    for dimension in (2, 16):
        torch.manual_seed(0)
        midpoint = state_builder(dimension, 'state_midpoint4').double()
        rng = torch.get_rng_state()
        torch.manual_seed(0)
        candidate = build_model(dimension, 'state_rk4_2').double()
        helpers.same(midpoint.state_dict(), candidate.state_dict())
        helpers.same(dict(midpoint.named_buffers()), dict(candidate.named_buffers()))
        assert torch.equal(rng, torch.get_rng_state())
        assert midpoint.parameter_count() == candidate.parameter_count()
        layer = next(candidate.ode_layers())[1]
        reference_layer = next(midpoint.ode_layers())[1]
        h = torch.linspace(-1.1, 1.3, 64, dtype=torch.double, requires_grad=True)
        with torch.no_grad():
            p = layer.operator().numpy().copy()
            expected = h.numpy().copy()
            def f(value):
                return p@(value/np.sqrt(1+1.2*value*value))
            for _ in range(2):
                a = f(expected)
                b = f(expected+.25*a)
                c = f(expected+.25*b)
                d = f(expected+.5*c)
                expected += (a+2*b+2*c+d)/12
            assert np.allclose(layer(h[None])[0].numpy(), expected, atol=1e-12, rtol=1e-12)
        assert torch.autograd.gradcheck(lambda value:layer(value[None])[0], (h,), fast_mode=True)
        loss = layer(h[None]).square().mean()
        gradient = torch.autograd.grad(loss, layer.proj.weight)[0]
        assert torch.isfinite(gradient).all() and gradient.norm() > 0
        direction = torch.sin(torch.arange(layer.proj.weight.numel(), dtype=torch.double)).reshape_as(layer.proj.weight)
        direction /= direction.norm()
        initial = layer.proj.weight.detach().clone()
        values = []
        with torch.no_grad():
            for sign in (1, -1):
                layer.proj.weight.copy_(initial+sign*1e-5*direction)
                values.append(float(layer(h[None]).square().mean()))
            layer.proj.weight.copy_(initial)
        finite = (values[0]-values[1])/2e-5
        analytic = float((gradient*direction).sum())
        assert np.isclose(finite, analytic, atol=1e-7, rtol=1e-5)
        counts = []
        for inspected in [reference_layer, layer]:
            with patch('torch.nn.functional.linear', wraps=torch.nn.functional.linear) as linear:
                inspected(h[None])
            counts.append(linear.call_count)
        assert counts == [8, 8]
        with torch.no_grad():
            refined = layer.solve(h[None], 64)
            errors = [float((layer.solve(h[None], steps)-refined).norm()) for steps in [2, 4, 8]]
            initial_map_difference = float((layer(h[None])-reference_layer(h[None])).norm())
        assert errors[0] > 8*errors[1] and errors[1] > 8*errors[2]
        rows.append(dict(dimension=dimension, parameters=candidate.parameter_count(),
            initialized_tensors_buffers_rng_exact=True, numpy_rk4_formula=True,
            full_input_gradient=True, projection_directional_gradient_error=abs(finite-analytic),
            matrix_applications=counts, refinement_errors=errors,
            initial_local_map_difference=initial_map_difference))
    base.dump(out/'kernels.json', dict(passed=True, rows=rows))
    continuations = []
    for cell in json.loads((HERE/'rk4-grid.json').read_text())['cells']:
        def folder(root):
            return root/TAG/cell['dataset']/cell['id']/'seed000'
        def run(root, steps, label, resume=None):
            command = [sys.executable, str(HERE/'rk4_trial.py'), '--cell', cell['id'],
                       '--data-root', args.data_root, '--output', str(root), '--device', 'cpu', '--steps', str(steps)]
            if resume:
                command += ['--resume', str(resume)]
            with (out/(cell['id']+label+'.log')).open('x') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        run(out/'resume', 8, '-8')
        run(out/'resume', 16, '-resume16', folder(out/'resume')/'checkpoints/step_0000008.pt')
        run(out/'continuous', 16, '-continuous16')
        left = torch.load(folder(out/'resume')/'checkpoints/step_0000016.pt', weights_only=False)
        right = torch.load(folder(out/'continuous')/'checkpoints/step_0000016.pt', weights_only=False)
        helpers.numerical(left, right)
        old_id = cell['id'].replace('state_rk4_2', 'state_midpoint4')
        command = [sys.executable, str(HERE/'state_trial.py'), '--cell', old_id,
                   '--data-root', args.data_root, '--output', str(out/'midpoint-control'), '--device', 'cpu', '--steps', '8']
        with (out/(cell['id']+'-midpoint8.log')).open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        old_folder = out/'midpoint-control'/'state_flow_20261116'/cell['dataset']/old_id/'seed000'
        old8 = torch.load(old_folder/'checkpoints/step_0000008.pt', weights_only=False)
        new8 = torch.load(folder(out/'resume')/'checkpoints/step_0000008.pt', weights_only=False)
        assert torch.equal(old8['data_rng'], new8['data_rng'])
        assert old8['dataset_manifest_sha256'] == new8['dataset_manifest_sha256']
        command = [sys.executable, str(HERE/'rk4_evaluate.py'), '--folder', str(folder(out/'resume')),
                   '--data-root', args.data_root, '--assets', args.assets, '--output', str(out/('evaluation-'+cell['dataset'])),
                   '--device', 'cpu', '--steps', '16', '--witness']
        with (out/(cell['id']+'-eval.log')).open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        continuations.append(dict(dataset=cell['dataset'], native_resume_matches=True,
            training_data_rng_matches_midpoint=True, native_evaluator_passed=True))
    base.dump(out/'PASS.json', dict(passed=True, kernels=rows, continuations=continuations,
        native_continuations=2, real_evaluator_witnesses=2, source_sha256=base.digest(__file__)))
    print(json.dumps(dict(passed=True, output=str(out))), flush=True)


if __name__ == '__main__':
    main()
