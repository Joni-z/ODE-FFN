"""One native layer intervention at a time, on bound original inputs."""
import argparse
import os
from pathlib import Path
import time
import numpy as np
from geometry import read, dump, sha
from frozen_probe import restore, tensor_hashes, assets, torch, base
from check_hidden_energy import potential, response

MODES = ('identity', 'reference', 'normal', 'tangential', 'ray')


def candidates(native: np.ndarray, reference: np.ndarray) -> dict:
    difference = native-reference
    q = response(reference)
    denominator = np.sum(q*q, axis=1)
    coefficient = np.divide(np.sum(difference*q, axis=1), denominator,
                            out=np.zeros(len(native)), where=denominator != 0)
    normal = coefficient[:, None]*q
    tangent = difference-normal
    target = potential(reference)
    zero = np.sum(native*native, axis=1) == 0
    qualified = ~(zero & (target != 0))
    low, high = np.zeros(len(native)), np.ones(len(native))
    for _ in range(64):
        too_small = (potential(high[:, None]*native) < target) & qualified
        if not too_small.any():
            break
        high[too_small] *= 2
    assert not ((potential(high[:, None]*native) < target)&qualified).any()
    for _ in range(64):
        middle = (low+high)/2
        below = potential(middle[:, None]*native) < target
        low = np.where(below, middle, low)
        high = np.where(below, high, middle)
    scale = np.where(target == 0, 0., (low+high)/2)
    scale = np.where(qualified, scale, 1.)
    ray = scale[:, None]*native
    assert np.all(np.abs(potential(ray)[qualified]-target[qualified]) <= 1e-11*(1+np.abs(target[qualified])))
    return dict(local_outputs=np.stack([native, reference, native-normal, native-tangent, ray]),
        ray_scale=scale, ray_qualified=qualified, zero_reference_gradient=denominator == 0)


def replace_one(model, layer, inputs: dict, expected_input: np.ndarray,
                expected_output: np.ndarray, replacement: np.ndarray) -> np.ndarray:
    calls = 0
    def hook(module, args, output):
        nonlocal calls
        calls += 1
        original = output[0] if isinstance(output, tuple) else output
        assert torch.equal(args[0], torch.from_numpy(expected_input))
        assert torch.equal(original, torch.from_numpy(expected_output))
        value = torch.from_numpy(replacement)
        return (value, *output[1:]) if isinstance(output, tuple) else value
    handle = layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            output = model(torch.from_numpy(inputs['x']), torch.from_numpy(inputs['time'])).numpy()
    finally:
        handle.remove()
    assert calls == 1
    return output


def main() -> None:
    p = argparse.ArgumentParser()
    for key in ['study', 'energy', 'data-root', 'output']:
        p.add_argument('--'+key, required=True)
    p.add_argument('--dataset', choices=['gmm_ring8_2d', 'rotated_product_bimodal_16d'])
    p.add_argument('--witness', action='store_true')
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    torch.set_num_threads(1)
    study, energy, data_root, out = map(Path, [a.study, a.energy, a.data_root, a.output])
    out.mkdir(parents=True, exist_ok=False)
    base.verify_sources()
    anchors = read(study/'anchors.json')
    terminal = read(energy/'terminal-receipts.json')
    assert terminal['passed'] and all(v == 0 for v in terminal['native_exits'].values())
    assert all(v == 0 for v in terminal['independent_exits'].values())
    original_rows = {r['id']: r for r in anchors['references']+anchors['new_cases']}
    sources = []
    for name, task in [('ring', 'gmm_ring8_2d'), ('product', 'rotated_product_bimodal_16d')]:
        if a.dataset is not None and task != a.dataset:
            continue
        folder, verified = energy/'formal'/name, energy/'formal'/(name+'-verified')
        complete, proof = read(folder/'COMPLETE.json'), read(verified/'COMPLETE.json')
        assert complete['passed'] and complete['all_qualified'] and not complete['witness']
        assert sha(folder/'rows.json') == complete['rows_sha256'] and proof['passed']
        assert sha(verified/'summary.json') == proof['summary_sha256']
        assert read(verified/'summary.json')['input_complete_sha256'] == sha(folder/'COMPLETE.json')
        sources += [(folder, row) for row in read(folder/'rows.json')['rows']]
    assert len(sources) == (3 if a.dataset else 6)
    started, results = time.monotonic(), []
    dump(out/'RUNNING.json', dict(pid=os.getpid(), cases=len(sources), witness=a.witness,
        check_only=a.check_only, started_unix=time.time()))
    for source_folder, row in sources:
        model, law, cell, evaluation, identity = restore(study, anchors, original_rows[row['id']], data_root)
        assert identity['checkpoint_sha256'] == row['checkpoint_sha256']
        arrays, _, binding = assets(anchors['assets'], cell['dataset'], 'selection', 'cpu')
        with torch.no_grad():
            fp32 = model(torch.from_numpy(arrays['field_x'][:128]), torch.from_numpy(arrays['field_t'][:128])).numpy()
        saved = np.load(evaluation/'field-20000.npy')[:128]
        assert np.allclose(fp32, saved, atol=2e-5, rtol=2e-5)
        model.double()
        snapshot = tensor_hashes(model)
        source_inputs = source_folder/row['inputs_file']
        assert sha(source_inputs) == row['inputs_sha256']
        with np.load(source_inputs) as z:
            inputs = {k: z[k].copy() for k in z.files}
        with torch.no_grad():
            baseline = model(torch.from_numpy(inputs['x']), torch.from_numpy(inputs['time'])).numpy()
        assert np.array_equal(baseline, inputs['native_prediction'])
        oracle = np.zeros_like(baseline)
        for t in np.unique(inputs['time']):
            mask = inputs['time'][:, 0] == t
            oracle[mask] = law.velocity(inputs['x'][mask], float(t))
        folder = out/row['id']
        folder.mkdir()
        np.savez_compressed(folder/'inputs.npz', **inputs, baseline=baseline, oracle=oracle)
        modules = dict(model.ode_layers())
        selected = row['layers'][:1] if a.witness else row['layers']
        layers = []
        for i, record in enumerate(selected):
            source_file = source_folder/record['file']
            assert sha(source_file) == record['sha256']
            with np.load(source_file) as z:
                entrance, native, reference = z['entrance'].copy(), z['native_forward'].copy(), z['references'][-1].copy()
            values = candidates(native, reference)
            layer = modules[record['layer']]
            fields = [replace_one(model, layer, inputs, entrance, native, values['local_outputs'][0])]
            assert np.array_equal(fields[0], baseline)
            if not a.check_only:
                fields += [replace_one(model, layer, inputs, entrance, native, value)
                           for value in values['local_outputs'][1:]]
            file = folder/f'layer{i}.npz'
            np.savez_compressed(file, **values, fields=np.stack(fields))
            layers.append(dict(layer=record['layer'], file=str(file.relative_to(out)), sha256=sha(file),
                source_layer=str(source_file), source_layer_sha256=sha(source_file),
                ray_qualified_points=int(values['ray_qualified'].sum()),
                exact_entrance=True, exact_original_layer_output=True, exact_identity_output=True))
            print(f"{row['id']} one-layer intervention {i+1}/{len(selected)}", flush=True)
        assert snapshot == tensor_hashes(model)
        assert all(len(layer._forward_hooks) == 0 for layer in modules.values())
        with torch.no_grad():
            roundtrip = model(torch.from_numpy(inputs['x']), torch.from_numpy(inputs['time'])).numpy()
        assert np.array_equal(roundtrip, baseline)
        results.append(dict(id=row['id'], dataset=cell['dataset'], family=cell['family'], **identity,
            layers=layers, inputs_file=str((folder/'inputs.npz').relative_to(out)), inputs_sha256=sha(folder/'inputs.npz'),
            source_inputs=str(source_inputs), source_inputs_sha256=sha(source_inputs),
            parameters_and_buffers_unchanged=True, native_roundtrip_exact=True))
        dump(out/'rows.json', dict(rows=results))
        dump(out/'progress.json', dict(pid=os.getpid(), completed=len(results), cases=len(sources), seconds=time.monotonic()-started))
    dump(out/'COMPLETE.json', dict(passed=True, witness=a.witness, check_only=a.check_only, cases=len(results),
        inputs_per_case=80, layers_per_case=1 if a.witness else 6, modes=list(MODES),
        rows_sha256=sha(out/'rows.json'), anchors_sha256=sha(study/'anchors.json'),
        original_energy_terminal_sha256=sha(energy/'terminal-receipts.json'),
        seconds=time.monotonic()-started, source_sha256={f.name: sha(f) for f in
            [Path(__file__), Path(__file__).with_name('check_hidden_energy.py'), Path(__file__).with_name('frozen_probe.py')]}))


if __name__ == '__main__':
    main()
