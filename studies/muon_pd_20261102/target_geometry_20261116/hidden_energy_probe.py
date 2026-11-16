"""Frozen learned-state energy audit, preserving native feature maps and inputs."""
import argparse
import os
from pathlib import Path
import time
import numpy as np
from geometry import read, dump, sha, TIMES
from frozen_probe import restore, tensor_hashes, assets, torch, base
from hidden_energy import LEVELS, TOLERANCES, midpoint, reference, response, observables, summarize

FAMILIES = ('state_midpoint4', 'palindrome_operator_midpoint4', 'alternating_operator_midpoint4')


def fixed_inputs(arrays: dict, witness: bool) -> dict:
    count, clocks = (1, (.5,)) if witness else (2, TIMES)
    source = np.concatenate([arrays[f'source_{b}'][:count] for b in range(8)]).astype(float)
    target = np.concatenate([arrays[f'target_{b}'][:count] for b in range(8)]).astype(float)
    return dict(source=source, target=target,
        x=np.concatenate([(1-t)*source+t*target for t in clocks]),
        time=np.repeat(np.asarray(clocks), len(source))[:, None],
        bank=np.tile(np.repeat(np.arange(8), count), len(clocks)),
        pair=np.tile(np.arange(count), 8*len(clocks)))


def capture(model, inputs: dict) -> tuple:
    entrances, outputs, handles = {}, {}, []
    for name, layer in model.ode_layers():
        def before(module, args, key=name):
            assert key not in entrances
            entrances[key] = args[0].detach().clone()
        def after(module, args, output, key=name):
            value = output[0] if isinstance(output, tuple) else output
            outputs[key] = value.detach().clone()
        handles += [layer.register_forward_pre_hook(before), layer.register_forward_hook(after)]
    try:
        with torch.no_grad():
            prediction = model(torch.from_numpy(inputs['x']), torch.from_numpy(inputs['time'])).numpy()
    finally:
        for handle in handles:
            handle.remove()
    assert len(entrances) == len(outputs) == 6
    return entrances, outputs, prediction


def main() -> None:
    p = argparse.ArgumentParser()
    for key in ['study', 'data-root', 'output']:
        p.add_argument('--'+key, required=True)
    p.add_argument('--dataset', choices=['gmm_ring8_2d', 'rotated_product_bimodal_16d'])
    p.add_argument('--witness', action='store_true')
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    study, data_root, out = map(Path, [a.study, a.data_root, a.output])
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    base.verify_sources()
    anchors = read(study/'anchors.json')
    bound = [dict(r, family=read(Path(r['folder'])/'config.json')['cell']['family'])
             for r in anchors['references']+anchors['new_cases']]
    rows = [r for r in bound if
            r['family'] in FAMILIES and (a.dataset is None or r['dataset'] == a.dataset)]
    assert len(rows) == (3 if a.dataset else 6)
    started = time.monotonic()
    dump(out/'RUNNING.json', dict(pid=os.getpid(), cases=len(rows), witness=a.witness,
        check_only=a.check_only, started_unix=time.time()))
    results = []
    for row in rows:
        model, law, cell, evaluation, identity = restore(study, anchors, row, data_root)
        arrays, _, binding = assets(anchors['assets'], cell['dataset'], 'selection', 'cpu')
        x32, t32 = torch.from_numpy(arrays['field_x'][:128]), torch.from_numpy(arrays['field_t'][:128])
        with torch.no_grad():
            fp32 = model(x32, t32).numpy()
        saved = np.load(evaluation/'field-20000.npy')[:128]
        assert np.allclose(fp32, saved, atol=2e-5, rtol=2e-5)
        model.double()
        snapshot = tensor_hashes(model)
        with torch.no_grad():
            fp64 = model(x32.double(), t32.double()).numpy()
        assert np.allclose(fp64, fp32, atol=3e-5, rtol=3e-5)
        inputs = fixed_inputs(arrays, a.witness)
        entrance, output, prediction = capture(model, inputs)
        folder = out/row['id']
        folder.mkdir()
        np.savez_compressed(folder/'inputs.npz', **inputs, native_prediction=prediction)
        layers = []
        for index, (name, layer) in enumerate(model.ode_layers()):
            before = time.monotonic()
            h = entrance[name]
            with torch.no_grad():
                if hasattr(layer, 'stage_weights'):
                    weights = layer.stage_weights().numpy()
                else:
                    weights = np.repeat(layer.operator().numpy()[None], 4, axis=0)
                native_response = (h*layer.step_size(h)).numpy()
                native = [layer.solve(h, level).numpy() for level in LEVELS]
            assert weights.shape == (4, h.shape[1], h.shape[1])
            assert np.allclose(native_response, response(h.numpy()), atol=1e-14, rtol=1e-14)
            assert np.array_equal(native[0], output[name].numpy())
            values = [midpoint(h.numpy(), weights, level) for level in LEVELS]
            endpoints, work = np.stack([r[0] for r in values]), np.stack([r[1] for r in values])
            errors = [float(np.max(np.abs(value-expected))) for value, expected in zip(endpoints, native)]
            assert all(np.allclose(value, expected, atol=1e-11, rtol=1e-11)
                       for value, expected in zip(endpoints, native)), (row['id'], name, errors)
            raw = dict(entrance=h.numpy(), native_forward=output[name].numpy(),
                weights=weights, steps=np.asarray(LEVELS), endpoints=endpoints, work=work,
                native_endpoints=np.stack(native))
            details, nfev = None, []
            if not a.check_only:
                references = [reference(h.numpy(), weights, *tol) for tol in TOLERANCES]
                raw.update(references=np.stack([r[0] for r in references]),
                    reference_work=np.stack([r[1] for r in references]), tolerances=np.asarray(TOLERANCES))
                nfev = [r[2] for r in references]
                details = summarize(observables(raw))
            file = folder/f'layer{index}.npz'
            np.savez_compressed(file, **raw)
            layers.append(dict(layer=name, file=str(file.relative_to(out)), sha256=sha(file),
                points=len(h), native_numpy_max_errors=errors, reference_nfev=nfev,
                summary=details, seconds=time.monotonic()-before))
            dump(out/'progress.json', dict(pid=os.getpid(), id=row['id'], layer=index+1,
                layers_per_case=6, completed_cases=len(results), cases=len(rows),
                elapsed_seconds=time.monotonic()-started))
            print(f"{row['id']} layer{index+1}/6", flush=True)
        assert snapshot == tensor_hashes(model)
        results.append(dict(id=row['id'], dataset=cell['dataset'], family=cell['family'],
            **identity, assets_file=binding['file'], assets_sha256=binding['sha256'],
            inputs_file=str((folder/'inputs.npz').relative_to(out)), inputs_sha256=sha(folder/'inputs.npz'),
            native_fp32_replay_max_error=float(np.max(np.abs(fp32-saved))),
            parameters_and_buffers_unchanged=True, layers=layers))
        dump(out/'rows.json', dict(rows=results))
    qualified = None if a.check_only else all(layer['summary']['qualified_points'] == layer['points']
        for row in results for layer in row['layers'])
    dump(out/'COMPLETE.json', dict(passed=True, witness=a.witness, check_only=a.check_only,
        cases=len(results), layers_per_case=6, points_per_layer=8 if a.witness else 80,
        all_qualified=qualified, rows_sha256=sha(out/'rows.json'), anchors_sha256=sha(study/'anchors.json'),
        seconds=time.monotonic()-started, device='cpu', torch=torch.__version__, numpy=np.__version__,
        source_sha256={f.name: sha(f) for f in [Path(__file__), Path(__file__).with_name('hidden_energy.py'),
            Path(__file__).with_name('check_hidden_energy.py'), Path(__file__).with_name('frozen_probe.py')]}))


if __name__ == '__main__':
    main()
