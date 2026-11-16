"""Complete frozen-field solver substitutions, at every actual layer entrance."""
import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import pickle
import time
from unittest.mock import patch
import hashlib
import numpy as np
from rk4_fidelity import (HERE, GEOMETRY, base, torch, evaluator, read, dump, sha,
                          tensor_hashes, reference, potential)
from state_trial import StateFlowLayer
from rk4_trial import StateRK4Layer

MODES = ('original', 'identity', 'midpoint', 'rk4', 'continuous_loose', 'continuous_tight')
TOLS = {'continuous_loose':(1e-10, 1e-11), 'continuous_tight':(1e-12, 1e-13)}


def rng_hashes() -> dict:
    return dict(torch=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                numpy=hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest())


def complete_forward(model, x: np.ndarray, t: np.ndarray, mode: str,
                     output: Path, progress: Path) -> dict:
    """Patch only solve methods; all six layers execute the native full model."""
    layers = list(model.ode_layers())
    assert len(layers) == 6
    count, width = len(x), layers[0][1].proj.weight.shape[0]
    entrances = np.empty((6, count, width))
    endpoints = np.empty_like(entrances)
    work = np.zeros((6, count))
    predictions = np.empty((count, x.shape[1]))
    matrices = np.stack([layer.operator().detach().numpy() for _,layer in layers])
    evaluations, calls = [0]*6, [0]*6
    start, stop = 0, 0
    before = time.monotonic()
    with ExitStack() as stack:
        for index, (name, layer) in enumerate(layers):
            original_solve = layer.solve
            def pre_hook(module, args, i=index):
                calls[i] += 1
                entrances[i, start:stop] = args[0].detach().numpy()
            def post_hook(module, args, result, i=index):
                value = result[0] if isinstance(result, tuple) else result
                endpoints[i, start:stop] = value.detach().numpy()
            handles = [layer.register_forward_pre_hook(pre_hook),
                       layer.register_forward_hook(post_hook)]
            for handle in handles:
                stack.callback(handle.remove)
            if mode != 'original':
                def solve(value, steps, i=index, module=layer, original=original_solve):
                    if mode == 'identity':
                        return original(value, steps)
                    if mode == 'midpoint':
                        return StateFlowLayer.solve(module, value, 4)
                    if mode == 'rk4':
                        return StateRK4Layer.solve(module, value, 2)
                    assert mode in TOLS
                    initial = value.detach().numpy()
                    weights = np.repeat(matrices[i][None], 4, axis=0)
                    result, integral, nfev = reference(initial, weights, *TOLS[mode])
                    work[i, start:stop] = integral
                    evaluations[i] += nfev
                    return torch.from_numpy(result)
                stack.enter_context(patch.object(layer, 'solve', side_effect=solve))
        for start in range(0, count, 128):
            stop = min(count, start+128)
            previous = calls.copy()
            with torch.no_grad():
                predictions[start:stop] = model(torch.from_numpy(x[start:stop]),
                                                torch.from_numpy(t[start:stop])).numpy()
            assert calls == [v+1 for v in previous]
            dump(progress, dict(pid=os.getpid(), mode=mode, processed=stop, total=count,
                                seconds=time.monotonic()-before))
            print(mode, stop, '/', count, flush=True)
    np.savez_compressed(output, entrances=entrances, endpoints=endpoints,
        matrices=matrices, work=work, predictions=predictions)
    return dict(mode=mode, file=output.name, sha256=sha(output), calls=calls,
        reference_evaluations=evaluations, seconds=time.monotonic()-before,
        has_reference_work=mode in TOLS)


def risk_accounting(numerical: np.ndarray, continuous: np.ndarray, oracle: np.ndarray) -> dict:
    error, delta = numerical-oracle, continuous-numerical
    r = np.sum(error*error, axis=1)
    s = np.sum(delta*delta, axis=1)
    c = np.sum(error*delta, axis=1)
    actual = np.sum((continuous-oracle)**2, axis=1)-r
    identity_error = float(np.max(np.abs(actual-(2*c+s))))
    assert identity_error <= 1e-12*(1+float(r.max())+float(s.max()))
    R, S, C = map(float, (r.mean(), s.mean(), c.mean()))
    return dict(R=R, S=S, C=C, continuous_R=float(np.mean(np.sum((continuous-oracle)**2, axis=1))),
        risk_change=float(actual.mean()), exposure=S/R if R else None,
        alignment=C/np.sqrt(R*S) if R*S else None,
        absolute_change_bound=float(2*np.sqrt(R*S)+S),
        identity_max_error=identity_error)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ['root', 'data-root', 'output']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--dataset', required=True,
        choices=['gmm_ring8_2d', 'rotated_product_bimodal_16d'])
    parser.add_argument('--witness', action='store_true')
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    torch.set_num_threads(1)
    base.verify_sources()
    root, out = Path(args.root), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    protocol = HERE.parents[2]/'docs/research-assessments/2026-11-16-ode-mechanism'
    binding = read(protocol/'rk4-control-bindings.json')
    terminal = read(root/'ops/TERMINAL.json')
    assert terminal['passed']
    old = next(r for r in binding['references'] if
               r['dataset']==args.dataset and r['family']=='state_midpoint4')
    new = next(r for r in terminal['cases'] if r['dataset']==args.dataset)
    cases = [dict(id=old['id'], folder=old['folder'], family='state_midpoint4',
                  evaluation=str(Path(old['evaluations']['selection']['summary_file']).parent)),
             dict(id=new['id'], folder=new['folder'], family='state_rk4_2',
                  evaluation=str(root/'analysis/selection'/new['id']))]
    arrays, _, asset = evaluator.assets(binding['assets_root'], args.dataset, 'selection', 'cpu')
    assert arrays['field_x'].shape[0] == 4096
    n = 8 if args.witness else 4096
    x, t, oracle = [arrays['field_'+key][:n].astype(float) for key in ['x','t','oracle']]
    np.savez_compressed(out/'inputs.npz', x=x, time=t, oracle=oracle,
                        original_indices=np.arange(n))
    rows, began = [], time.monotonic()
    for case in cases:
        folder, evaluation = Path(case['folder']), Path(case['evaluation'])
        assert (folder/'ADAPTER_COMPLETE').exists()
        cfg = read(folder/'config.json')
        assert cfg['steps'] == 20000 and cfg['seeds'] == [0]
        assert cfg['cell']['family'] == case['family']
        cache = base.verify_data(args.data_root, cfg['cell'])
        manifest = cache/'manifest.json'
        assert asset['data_manifest_sha256'] == sha(manifest)
        checkpoint = folder/'checkpoints/step_0020000.pt'
        state = torch.load(checkpoint, weights_only=False, map_location='cpu')
        assert state['step']==20000 and state['config_sha256']==sha(folder/'config.json')
        assert state['dataset_manifest_sha256']==sha(manifest)
        summary = next(r for r in read(evaluation/'summary.json') if r['step']==20000)
        assert summary['checkpoint_sha256']==sha(checkpoint)
        from rk4_trial import build_model
        torch.manual_seed(0)
        model = build_model(x.shape[1], case['family']).cpu().eval()
        base.train.restore_model(model, state)
        fp32 = []
        for start in range(0, n, 1024):
            with torch.no_grad():
                fp32.append(model(torch.from_numpy(arrays['field_x'][start:min(n,start+1024)]),
                                  torch.from_numpy(arrays['field_t'][start:min(n,start+1024)])).numpy())
        fp32 = np.concatenate(fp32)
        saved = np.load(evaluation/'field-20000.npy')[:n]
        assert np.allclose(fp32, saved, atol=2e-5, rtol=2e-5)
        model.double()
        snapshot, rng = tensor_hashes(model), rng_hashes()
        target = out/case['id']
        target.mkdir()
        modes = [complete_forward(model, x, t, mode, target/(mode+'.npz'), out/'progress.json')
                 for mode in MODES]
        with np.load(target/'original.npz') as z:
            original = z['predictions'].copy()
        assert np.allclose(original, fp32, atol=3e-5, rtol=3e-5)
        with np.load(target/'identity.npz') as z:
            assert np.array_equal(original, z['predictions'])
        own = 'midpoint' if case['family']=='state_midpoint4' else 'rk4'
        with np.load(target/(own+'.npz')) as z:
            assert np.array_equal(original, z['predictions'])
        with np.load(target/'continuous_loose.npz') as lo, np.load(target/'continuous_tight.npz') as hi:
            qualified = np.all(np.abs(lo['predictions']-hi['predictions']) <=
                               1e-9+1e-9*np.abs(hi['predictions']), axis=1)
            for key in ['entrances','endpoints']:
                qualified &= np.all(np.abs(lo[key]-hi[key]) <= 1e-9+1e-9*np.abs(hi[key]), axis=(0,2))
            for z in [lo, hi]:
                initial, final = potential(z['entrances']), potential(z['endpoints'])
                closure = final-initial-z['work']
                qualified &= np.all(np.abs(closure) <=
                    1e-8*(1+np.abs(initial)+np.abs(final)+np.abs(z['work'])), axis=0)
            continuous = hi['predictions'].copy()
            floor = hi['predictions']-lo['predictions']
            floor_energy = float(np.mean(np.sum(floor*floor,axis=1)))
            continuous_risk = float(np.mean(np.sum((continuous-oracle)**2,axis=1)))
            resolution = 2*np.sqrt(continuous_risk*floor_energy)+floor_energy
        accounting = {}
        for mode in ['midpoint','rk4']:
            with np.load(target/(mode+'.npz')) as z:
                accounting[mode] = risk_accounting(z['predictions'], continuous, oracle)
        assert tensor_hashes(model)==snapshot and rng_hashes()==rng
        np.save(target/'qualified.npy', qualified, allow_pickle=False)
        row = dict(**case, dataset=args.dataset, checkpoint_sha256=sha(checkpoint),
            config_sha256=sha(folder/'config.json'), manifest_sha256=sha(manifest),
            asset_sha256=asset['sha256'], modes=modes, points=n,
            qualified_points=int(qualified.sum()), parameters_and_rng_unchanged=True,
            original_identity_exact=True, original_own_solver_exact=True,
            native_fp32_replay_max_error=float(np.max(np.abs(fp32-saved))),
            conversion_max_error=float(np.max(np.abs(original-fp32))),
            risk_accounting=accounting, continuous_tolerance_risk_resolution=float(resolution),
            qualification_sha256=sha(target/'qualified.npy'))
        rows.append(row)
        dump(out/'rows.json', dict(rows=rows))
    dump(out/'COMPLETE.json', dict(passed=True, witness=args.witness, cases=2, points_per_case=n,
        modes=MODES, layers_per_case=6, all_points_qualified=all(r['points']==r['qualified_points'] for r in rows),
        rows_sha256=sha(out/'rows.json'), inputs_sha256=sha(out/'inputs.npz'),
        seconds=time.monotonic()-began, source_sha256={str(p):sha(p) for p in
        [Path(__file__), HERE/'rk4_trial.py', HERE/'state_trial.py', GEOMETRY/'hidden_energy.py']}))
    print('field exposure complete', args.dataset, flush=True)


if __name__ == '__main__':
    main()
