"""Finite one-layer normal/complement interventions in complete fixed frames."""
import argparse
import os
from pathlib import Path
import time
import numpy as np
from geometry import read, dump, sha
from frozen_probe import restore, tensor_hashes, assets, torch, base
from hidden_readout_probe import replace_one


def main() -> None:
    parser = argparse.ArgumentParser()
    for key in ['study', 'frames', 'data-root', 'output']:
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--dataset', required=True, choices=['gmm_ring8_2d', 'rotated_product_bimodal_16d'])
    parser.add_argument('--witness', action='store_true')
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    torch.set_num_threads(1)
    study, frame_root, data_root, out = map(Path, [args.study, args.frames, args.data_root, args.output])
    out.mkdir(parents=True, exist_ok=False)
    base.verify_sources()
    manifest = read(frame_root/'manifest.json')
    assert manifest['passed'] and manifest['evaluation_rng'] == 0 and manifest['frames'] == 36
    anchors = read(study/'anchors.json')
    originals = {row['id']:row for row in anchors['references']+anchors['new_cases']}
    rows = [row for row in manifest['rows'] if row['dataset'] == args.dataset]
    assert len(rows) == 3
    started, results = time.monotonic(), []
    dump(out/'RUNNING.json', dict(pid=os.getpid(), cases=3, witness=args.witness, directions=64))
    for row in rows:
        source = Path(row['source_readout'])
        assert sha(source/'rows.json') == row['source_rows_sha256']
        assert sha(source/'COMPLETE.json') == row['source_complete_sha256']
        old = next(item for item in read(source/'rows.json')['rows'] if item['id'] == row['id'])
        assert sha(Path(row['file'])) == row['sha256']
        with np.load(row['file']) as bundle:
            frames = bundle['frames'].copy()
        assert frames.shape == (6, 64, 64)
        model, _, cell, evaluation, identity = restore(study, anchors, originals[row['id']], data_root)
        assert identity['checkpoint_sha256'] == old['checkpoint_sha256']
        arrays = assets(anchors['assets'], cell['dataset'], 'selection', 'cpu')[0]
        with torch.no_grad():
            native_fp32 = model(torch.from_numpy(arrays['field_x'][:128]), torch.from_numpy(arrays['field_t'][:128])).numpy()
        assert np.allclose(native_fp32, np.load(evaluation/'field-20000.npy')[:128], atol=2e-5, rtol=2e-5)
        model.double()
        before = tensor_hashes(model)
        inputs_file = source/old['inputs_file']
        assert sha(inputs_file) == old['inputs_sha256']
        with np.load(inputs_file) as bundle:
            inputs = {key:bundle[key].copy() for key in bundle.files}
        with torch.no_grad():
            baseline = model(torch.from_numpy(inputs['x']), torch.from_numpy(inputs['time'])).numpy()
        assert np.array_equal(baseline, inputs['baseline'])
        folder = out/row['id']
        folder.mkdir()
        modules, layer_records = dict(model.ode_layers()), []
        selected = old['layers'][:1] if args.witness else old['layers']
        for index, record in enumerate(selected):
            original_file = source/record['file']
            energy_file = Path(record['source_layer'])
            assert sha(original_file) == record['sha256'] and sha(energy_file) == record['source_layer_sha256']
            assert record['layer'] == row['layers'][index]
            with np.load(original_file) as bundle:
                actual_local, actual_fields = bundle['local_outputs'].copy(), bundle['fields'].copy()
            with np.load(energy_file) as bundle:
                entrance = bundle['entrance'].copy()
            native, reference = actual_local[:2]
            delta, frame = native-reference, frames[index]
            assert np.allclose(frame@frame.T, np.eye(64), atol=1e-12, rtol=0)
            coefficients = delta@frame.T
            projections = coefficients.T[:, :, None]*frame[:, None, :]
            locals_ = np.stack([native[None]-projections, reference[None]+projections])
            layer = modules[record['layer']]
            identity_field = replace_one(model, layer, inputs, entrance, native, native)
            assert np.array_equal(identity_field, baseline)
            for mode in [2, 3]:
                prediction = replace_one(model, layer, inputs, entrance, native, actual_local[mode])
                assert np.array_equal(prediction, actual_fields[mode])
            fields = np.empty((2, 64, 80, baseline.shape[1]))
            for direction in range(64):
                for mode in range(2):
                    fields[mode, direction] = replace_one(model, layer, inputs, entrance, native, locals_[mode, direction])
                if direction % 16 == 15:
                    dump(out/'progress.json', dict(pid=os.getpid(), completed_cases=len(results), cases=3,
                        current=row['id'], completed_layers=index, layers=len(selected),
                        completed_directions=direction+1, directions=64, seconds=time.monotonic()-started))
            assert np.isfinite(fields).all() and np.isfinite(locals_).all()
            expected = np.sum(delta*delta, axis=1)/64
            measured = np.mean(np.sum(projections*projections, axis=-1), axis=0)
            assert np.allclose(measured, expected, atol=1e-11, rtol=1e-11)
            file = folder/f'layer{index}.npz'
            np.savez_compressed(file, frame=frame, local_outputs=locals_, fields=fields)
            layer_records.append(dict(layer=record['layer'], file=str(file.relative_to(out)), sha256=sha(file),
                source_readout=str(original_file), source_readout_sha256=sha(original_file),
                source_energy=str(energy_file), source_energy_sha256=sha(energy_file),
                frame_index=index, max_frame_identity_error=float(np.max(np.abs(measured-expected))),
                exact_upstream=True, exact_native_output=True, actual_component_outputs_exact=True))
            print(row['id'], 'complete frame', index+1, '/', len(selected), flush=True)
        assert before == tensor_hashes(model)
        assert all(not layer._forward_hooks for layer in modules.values())
        with torch.no_grad():
            final = model(torch.from_numpy(inputs['x']), torch.from_numpy(inputs['time'])).numpy()
        assert np.array_equal(final, baseline)
        results.append(dict(id=row['id'], dataset=cell['dataset'], family=cell['family'], **identity,
            source_inputs=str(inputs_file), source_inputs_sha256=sha(inputs_file),
            frame_file=row['file'], frame_sha256=row['sha256'], layers=layer_records,
            parameters_and_buffers_unchanged=True, native_roundtrip_exact=True))
        dump(out/'rows.json', dict(rows=results))
    dump(out/'COMPLETE.json', dict(passed=True, witness=args.witness, cases=3, inputs_per_case=80,
        layers_per_case=1 if args.witness else 6, directions=64, modes=['random_1d', 'random_complement'],
        rows_sha256=sha(out/'rows.json'), frames_manifest_sha256=sha(frame_root/'manifest.json'),
        anchors_sha256=sha(study/'anchors.json'), seconds=time.monotonic()-started,
        source_sha256={name:sha(Path(__file__).with_name(name)) for name in
            ['hidden_null_probe.py','hidden_readout_probe.py','frozen_probe.py']}))


if __name__ == '__main__':
    main()
