"""Qualify fixed native refinements against completed continuous interventions."""
import argparse
import os
from pathlib import Path
import time
import numpy as np
from geometry import read, dump, sha
from frozen_probe import restore, tensor_hashes, assets, torch, base
from hidden_energy import midpoint
from hidden_readout_probe import candidates, replace_one, MODES

LEVELS = (16, 32, 64, 128, 256)


def energy_sums(fields: np.ndarray, banks: np.ndarray) -> tuple:
    displacement = np.sum((fields[0]-fields[1])**2, axis=-1)
    residual = np.sum((fields[2:]-fields[1])**2, axis=-1)
    return (np.bincount(banks, weights=displacement, minlength=8),
            np.stack([np.bincount(banks, weights=value, minlength=8) for value in residual]))


def fractions(displacement: np.ndarray, residual: np.ndarray) -> np.ndarray:
    return 1-residual/displacement if np.all(displacement > 0) else np.full(residual.shape, np.nan)


def compare(local: np.ndarray, fields: np.ndarray, old: dict) -> dict:
    local_scale = 1e-9+1e-3*np.linalg.norm(old['local_outputs'][0]-old['local_outputs'][1], axis=-1)
    output_scale = 1e-10+1e-3*np.linalg.norm(old['fields'][0]-old['fields'][1], axis=-1)
    local_ratio = np.linalg.norm(local[1:]-old['local_outputs'][1:], axis=-1)/local_scale
    output_ratio = np.linalg.norm(fields[1:]-old['fields'][1:], axis=-1)/output_scale
    return dict(local_max_limit_ratio=float(local_ratio.max()), output_max_limit_ratio=float(output_ratio.max()),
        local_violations_by_mode=np.sum(local_ratio > 1, axis=1).tolist(),
        output_violations_by_mode=np.sum(output_ratio > 1, axis=1).tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    for key in ['study', 'energy', 'readout', 'data-root', 'output']:
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--dataset', required=True, choices=['gmm_ring8_2d', 'rotated_product_bimodal_16d'])
    parser.add_argument('--witness', action='store_true')
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    torch.set_num_threads(1)
    study, energy, readout, data_root, out = map(Path,
        [args.study, args.energy, args.readout, args.data_root, args.output])
    out.mkdir(parents=True, exist_ok=False)
    base.verify_sources()
    for root in [energy, readout]:
        terminal = read(root/'terminal-receipts.json')
        assert terminal['passed']
        assert all(value == 0 for value in terminal['native_exits'].values())
        assert all(value == 0 for value in terminal['independent_exits'].values())
    task = 'ring' if args.dataset == 'gmm_ring8_2d' else 'product'
    source_root = readout/'formal'/task
    complete = read(source_root/'COMPLETE.json')
    assert complete['passed'] and not complete['witness'] and not complete['check_only']
    assert sha(source_root/'rows.json') == complete['rows_sha256']
    rows = read(source_root/'rows.json')['rows']
    assert len(rows) == 3 and all(row['dataset'] == args.dataset for row in rows)
    anchors = read(study/'anchors.json')
    originals = {row['id']: row for row in anchors['references']+anchors['new_cases']}
    started, results = time.monotonic(), []
    dump(out/'RUNNING.json', dict(pid=os.getpid(), cases=3, witness=args.witness, levels=LEVELS))
    for row in rows:
        model, law, cell, evaluation, identity = restore(study, anchors, originals[row['id']], data_root)
        assert identity['checkpoint_sha256'] == row['checkpoint_sha256']
        arrays = assets(anchors['assets'], cell['dataset'], 'selection', 'cpu')[0]
        with torch.no_grad():
            native = model(torch.from_numpy(arrays['field_x'][:128]), torch.from_numpy(arrays['field_t'][:128])).numpy()
        assert np.allclose(native, np.load(evaluation/'field-20000.npy')[:128], atol=2e-5, rtol=2e-5)
        model.double()
        snapshot = tensor_hashes(model)
        source_inputs = source_root/row['inputs_file']
        assert sha(source_inputs) == row['inputs_sha256']
        with np.load(source_inputs) as bundle:
            inputs = {key: bundle[key].copy() for key in bundle.files}
        with torch.no_grad():
            baseline = model(torch.from_numpy(inputs['x']), torch.from_numpy(inputs['time'])).numpy()
        assert np.array_equal(baseline, inputs['baseline'])
        banks, modules = inputs['bank'], dict(model.ode_layers())
        records = row['layers'][:1] if args.witness else row['layers']
        folder = out/row['id']
        folder.mkdir()
        level_rows = []
        for level in LEVELS:
            level_start = time.monotonic()
            directory = folder/f'level{level}'
            directory.mkdir()
            layers = []
            denominator, numerator = np.zeros(8), np.zeros((3, 8))
            old_denominator, old_numerator = np.zeros(8), np.zeros((3, 8))
            for index, record in enumerate(records):
                old_file, energy_file = source_root/record['file'], Path(record['source_layer'])
                assert sha(old_file) == record['sha256']
                assert sha(energy_file) == record['source_layer_sha256']
                with np.load(old_file) as bundle:
                    old = {key: bundle[key].copy() for key in bundle.files}
                with np.load(energy_file) as bundle:
                    entrance, weights, h4 = [bundle[key].copy() for key in ['entrance', 'weights', 'native_forward']]
                    assert np.array_equal(bundle['references'][-1], old['local_outputs'][1])
                layer = modules[record['layer']]
                with torch.no_grad():
                    refined = layer.solve(torch.from_numpy(entrance), level).numpy()
                independent, _ = midpoint(entrance, weights, level)
                assert np.allclose(refined, independent, atol=1e-11, rtol=1e-11)
                values = candidates(h4, refined)
                assert values['ray_qualified'].all()
                fields = np.stack([replace_one(model, layer, inputs, entrance, h4, candidate)
                                   for candidate in values['local_outputs']])
                assert np.array_equal(fields[0], baseline)
                file = directory/f'layer{index}.npz'
                np.savez_compressed(file, **values, fields=fields,
                                    native_refined=refined, numpy_refined=independent)
                d, n = energy_sums(fields, banks)
                od, on = energy_sums(old['fields'], banks)
                difference = np.abs(fractions(d.sum(), n.sum(1))-fractions(od.sum(), on.sum(1)))
                metrics = compare(values['local_outputs'], fields, old)
                fraction_resolved = bool(np.isfinite(difference).all())
                layers.append(dict(layer=record['layer'], file=str(file.relative_to(out)), sha256=sha(file),
                    source_readout=str(old_file), source_readout_sha256=sha(old_file),
                    source_energy=str(energy_file), source_energy_sha256=sha(energy_file),
                    **metrics, fraction_resolved=fraction_resolved,
                    fraction_max_error=float(difference.max()) if fraction_resolved else None,
                    native_numpy_max_error=float(np.max(np.abs(refined-independent))),
                    qualified=bool(metrics['local_max_limit_ratio'] <= 1 and metrics['output_max_limit_ratio'] <= 1
                                   and fraction_resolved and difference.max() <= .0005)))
                denominator += d
                numerator += n
                old_denominator += od
                old_numerator += on
            bank_f = fractions(denominator[None], numerator)
            old_bank_f = fractions(old_denominator[None], old_numerator)
            model_f = fractions(denominator.sum(), numerator.sum(1))
            old_model_f = fractions(old_denominator.sum(), old_numerator.sum(1))
            resolved = bool(np.isfinite(bank_f).all() and np.isfinite(old_bank_f).all()
                            and np.isfinite(model_f).all() and np.isfinite(old_model_f).all())
            bank_error = float(np.max(np.abs(bank_f-old_bank_f))) if resolved else None
            model_error = float(np.max(np.abs(model_f-old_model_f))) if resolved else None
            level_rows.append(dict(steps=level, layers=layers, fractions_resolved=resolved,
                bank_fraction_max_error=bank_error, model_fraction_max_error=model_error,
                fractions=model_f.tolist() if resolved else None,
                qualified=bool(all(item['qualified'] for item in layers) and resolved
                               and bank_error <= .0005 and model_error <= .0005),
                one_layer_matrix_work_multiplier=(2*level+40)/48,
                seconds=time.monotonic()-level_start))
            dump(out/'progress.json', dict(pid=os.getpid(), completed_cases=len(results), current=row['id'],
                completed_levels=len(level_rows), levels=len(LEVELS), seconds=time.monotonic()-started))
            print(row['id'], level, 'qualified', level_rows[-1]['qualified'], flush=True)
        assert snapshot == tensor_hashes(model)
        assert all(not layer._forward_hooks for layer in modules.values())
        with torch.no_grad():
            final = model(torch.from_numpy(inputs['x']), torch.from_numpy(inputs['time'])).numpy()
        assert np.array_equal(final, baseline)
        results.append(dict(id=row['id'], dataset=cell['dataset'], family=cell['family'], **identity,
            source_inputs=str(source_inputs), source_inputs_sha256=sha(source_inputs), levels=level_rows,
            native_roundtrip_exact=True, parameters_and_buffers_unchanged=True))
        dump(out/'rows.json', dict(rows=results))
    qualified = [level for level in LEVELS if all(next(item for item in row['levels']
                  if item['steps'] == level)['qualified'] for row in results)]
    dump(out/'COMPLETE.json', dict(passed=True, witness=args.witness, cases=3, inputs_per_case=80,
        layers_per_case=len(records), levels=LEVELS, modes=list(MODES), qualified_levels=qualified,
        rows_sha256=sha(out/'rows.json'), anchors_sha256=sha(study/'anchors.json'),
        source_complete_sha256=sha(source_root/'COMPLETE.json'), seconds=time.monotonic()-started,
        environment_spec_hash='04f11f75', source_sha256={name: sha(Path(__file__).with_name(name)) for name in
            ['hidden_surrogate_probe.py', 'hidden_readout_probe.py', 'hidden_energy.py', 'frozen_probe.py']}))


if __name__ == '__main__':
    main()
