"""Measure native hidden-solve error against the same continuous ODE.

Reuse the original fixed inputs, hooks and two-tolerance reference. No training,
replacement outputs, new features or generation surrogate is introduced.
"""
import argparse
import os
from pathlib import Path
import sys
import time
import numpy as np
from rk4_trial import HERE, base, torch, build_model
from coefficient_evaluate import evaluator

GEOMETRY = HERE.parent/'target_geometry_20261116'
sys.path.insert(0, str(GEOMETRY))
from geometry import read, dump, sha
from frozen_probe import tensor_hashes
from hidden_energy_probe import fixed_inputs, capture
from hidden_energy import reference, response, TOLERANCES
from check_hidden_energy import potential


def independent_endpoint(entrance: np.ndarray, matrix: np.ndarray,
                         family: str, steps: int) -> np.ndarray:
    """Separate NumPy algebra checks the actual trained matrix and entrances."""
    y, dt = entrance.copy(), 1/steps
    def f(value: np.ndarray) -> np.ndarray:
        return response(value)@matrix.T
    for _ in range(steps):
        if family == 'state_midpoint4':
            y = y+dt*f(y+.5*dt*f(y))
        elif family == 'state_rk4_2':
            a = f(y)
            b = f(y+.5*dt*a)
            c = f(y+.5*dt*b)
            d = f(y+dt*c)
            y = y+dt*(a+2*b+2*c+d)/6
        else:
            raise ValueError(f'Unregistered family: {family}')
    return y


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ['folder', 'evaluation', 'data-root', 'assets', 'output']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--step', type=int, default=20000)
    parser.add_argument('--witness', action='store_true')
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    assert args.witness or args.step == 20000
    folder, evaluation, out = map(Path, [args.folder, args.evaluation, args.output])
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    base.verify_sources()
    cfg = read(folder/'config.json')
    cell = cfg['cell']
    assert cfg['steps'] == 20000 and cfg['seeds'] == [0]
    assert cell['family'] in ('state_midpoint4', 'state_rk4_2')
    assert (folder/'ADAPTER_COMPLETE').exists()
    cache = base.verify_data(args.data_root, cell)
    manifest = cache/'manifest.json'
    checkpoint = folder/'checkpoints'/f'step_{args.step:07d}.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['step'] == args.step
    assert state['config_sha256'] == sha(folder/'config.json')
    assert state['dataset_manifest_sha256'] == sha(manifest)
    assert read(evaluation/'COMPLETE.json')['passed']
    summary = next(r for r in read(evaluation/'summary.json') if r['step'] == args.step)
    assert summary['checkpoint_sha256'] == sha(checkpoint)
    spec = read(manifest)['datasets'][cell['dataset']]
    torch.manual_seed(0)
    model = build_model(spec['dim'], cell['family']).cpu().eval()
    base.train.restore_model(model, state)
    arrays, _, binding = evaluator.assets(args.assets, cell['dataset'], 'selection', 'cpu')
    x = torch.from_numpy(arrays['field_x'][:128])
    t = torch.from_numpy(arrays['field_t'][:128])
    with torch.no_grad():
        fp32 = model(x, t).numpy()
    saved = np.load(evaluation/f'field-{args.step}.npy')[:128]
    assert np.allclose(fp32, saved, atol=2e-5, rtol=2e-5)
    model.double()
    snapshot = tensor_hashes(model)
    with torch.no_grad():
        fp64 = model(x.double(), t.double()).numpy()
    assert np.allclose(fp32, fp64, atol=3e-5, rtol=3e-5)
    inputs = fixed_inputs(arrays, args.witness)
    entrances, outputs, prediction = capture(model, inputs)
    np.savez_compressed(out/'inputs.npz', **inputs, native_prediction=prediction)
    records, before = [], time.monotonic()
    for index, (name, layer) in enumerate(model.ode_layers()):
        h = entrances[name]
        levels = [layer.substeps*factor for factor in (1, 2, 4)]
        with torch.no_grad():
            matrix = layer.operator().numpy().copy()
            endpoints = np.stack([layer.solve(h, n).numpy() for n in levels])
        assert np.array_equal(endpoints[0], outputs[name].numpy())
        independent = np.stack([independent_endpoint(h.numpy(), matrix, cell['family'], n)
                                for n in levels])
        assert np.allclose(endpoints, independent, atol=1e-11, rtol=1e-11)
        weights = np.repeat(matrix[None], 4, axis=0)
        refs = [reference(h.numpy(), weights, *tol) for tol in TOLERANCES]
        reference_y = np.stack([r[0] for r in refs])
        work = np.stack([r[1] for r in refs])
        initial_energy = potential(h.numpy())
        final_energy = potential(reference_y[-1])
        closure = final_energy-initial_energy-work[-1]
        qualified = np.all(np.abs(reference_y[0]-reference_y[1]) <=
                           1e-9+1e-9*np.abs(reference_y[1]), axis=1)
        qualified &= np.abs(work[0]-work[1]) <= 1e-8*(1+np.abs(work[1]))
        qualified &= np.abs(closure) <= 1e-8*(1+np.abs(initial_energy)+
                                            np.abs(final_energy)+np.abs(work[1]))
        error2 = np.sum((endpoints-reference_y[-1])**2, axis=2)
        displacement2 = np.sum((reference_y[-1]-h.numpy())**2, axis=1)
        file = out/f'layer{index}.npz'
        np.savez_compressed(file, entrance=h.numpy(), matrix=matrix, levels=levels,
            endpoints=endpoints, independent_endpoints=independent,
            references=reference_y, reference_work=work, qualified=qualified,
            error_squared=error2, displacement_squared=displacement2, closure=closure)
        records.append(dict(layer=name, file=file.name, sha256=sha(file), points=len(h),
            qualified_points=int(qualified.sum()), levels=levels,
            native_numpy_max_error=float(np.max(np.abs(endpoints-independent))),
            reference_evaluations=[r[2] for r in refs],
            zero_displacement_points=int(np.sum(displacement2 == 0)),
            normalized_errors=[float(e.sum()/displacement2.sum())
                               if displacement2.sum() else None for e in error2]))
        dump(out/'progress.json', dict(pid=os.getpid(), completed_layers=index+1,
            layers=6, qualified_points=int(qualified.sum()), seconds=time.monotonic()-before))
        print(cell['id'], name, int(qualified.sum()), '/', len(h), flush=True)
    assert snapshot == tensor_hashes(model)
    result = dict(id=cell['id'], dataset=cell['dataset'], family=cell['family'],
        step=args.step, witness=args.witness, layers=records,
        checkpoint_sha256=sha(checkpoint), config_sha256=sha(folder/'config.json'),
        dataset_manifest_sha256=sha(manifest), evaluation_summary_sha256=sha(evaluation/'summary.json'),
        assets_file=binding['file'], assets_sha256=binding['sha256'],
        inputs_sha256=sha(out/'inputs.npz'), parameters_and_buffers_unchanged=True,
        native_replay_max_error=float(np.max(np.abs(fp32-saved))),
        all_points_qualified=all(r['points'] == r['qualified_points'] for r in records),
        seconds=time.monotonic()-before)
    dump(out/'rows.json', result)
    dump(out/'COMPLETE.json', dict(passed=True, witness=args.witness,
        all_points_qualified=result['all_points_qualified'], rows_sha256=sha(out/'rows.json'),
        source_sha256={str(p):sha(p) for p in [Path(__file__), HERE/'rk4_trial.py',
            HERE/'state_trial.py', GEOMETRY/'hidden_energy.py', GEOMETRY/'hidden_energy_probe.py']}))
    print(result['id'], 'complete', result['all_points_qualified'], flush=True)


if __name__ == '__main__':
    main()
