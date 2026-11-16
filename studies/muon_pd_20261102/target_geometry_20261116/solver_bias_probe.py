"""Predict midpoint endpoint errors from the frozen hidden field, without fitting."""
import argparse
import os
from pathlib import Path
import time
import numpy as np
from scipy.integrate import solve_ivp
from geometry import read, dump, sha
from check_solver_bias import components

LEVELS = (4, 8, 16, 32, 64, 128, 256)
TOLERANCES = ((1e-10, 1e-11), (1e-12, 1e-13))


def augmented_reference(entrance: np.ndarray, weights: np.ndarray, rtol: float, atol: float) -> tuple:
    shape = (3, *entrance.shape)
    value = np.stack([entrance, np.zeros_like(entrance), np.zeros_like(entrance)]).ravel()
    evaluations = 0
    for stage, matrix in enumerate(weights):
        def drift(time: float, packed: np.ndarray) -> np.ndarray:
            y, z_h, z_j = packed.reshape(shape)
            f, first, curvature, chain = components(y, matrix)
            return np.stack([f, (first*z_h)@matrix.T+curvature, (first*z_j)@matrix.T+chain]).ravel()
        result = solve_ivp(drift, (stage/4, (stage+1)/4), value, method='DOP853', rtol=rtol, atol=atol)
        assert result.success and np.isfinite(result.y).all(), result.message
        value = result.y[:, -1]
        evaluations += result.nfev
    return value.reshape(shape), evaluations


def qualification(augmented: np.ndarray, original: np.ndarray) -> np.ndarray:
    qualified = np.all(np.abs(augmented[0, 0]-augmented[1, 0]) <= 1e-9+1e-9*np.abs(augmented[1, 0]), axis=1)
    qualified &= np.all(np.abs(augmented[:, 0]-original[-1]) <= 1e-9+1e-9*np.abs(original[-1]), axis=(0, 2))
    qualified &= np.all(np.abs(augmented[0, 1:]-augmented[1, 1:]) <= 1e-8+1e-8*np.abs(augmented[1, 1:]), axis=(0, 2))
    return qualified


def main() -> None:
    parser = argparse.ArgumentParser()
    for key in ['energy', 'surrogate', 'output']:
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--dataset', required=True, choices=['gmm_ring8_2d', 'rotated_product_bimodal_16d'])
    parser.add_argument('--witness', action='store_true')
    args = parser.parse_args()
    assert os.environ.get('OMP_NUM_THREADS') == '1' and os.environ.get('OPENBLAS_NUM_THREADS') == '1'
    energy, surrogate, out = map(Path, [args.energy, args.surrogate, args.output])
    out.mkdir(parents=True, exist_ok=False)
    for source in [energy, surrogate]:
        terminal = read(source/'terminal-receipts.json')
        assert terminal['passed'] and all(value == 0 for value in terminal['native_exits'].values())
        assert all(value == 0 for value in terminal['independent_exits'].values())
    task = 'ring' if args.dataset == 'gmm_ring8_2d' else 'product'
    source, refined = energy/'formal'/task, surrogate/'formal'/task
    complete, refined_complete = read(source/'COMPLETE.json'), read(refined/'COMPLETE.json')
    assert complete['passed'] and complete['all_qualified'] and not complete['witness']
    assert refined_complete['passed'] and not refined_complete['witness']
    assert sha(source/'rows.json') == complete['rows_sha256']
    assert sha(refined/'rows.json') == refined_complete['rows_sha256']
    refined_rows = {row['id']:row for row in read(refined/'rows.json')['rows']}
    rows = read(source/'rows.json')['rows']
    assert len(rows) == 3 and all(row['dataset'] == args.dataset for row in rows)
    started, results = time.monotonic(), []
    dump(out/'RUNNING.json', dict(pid=os.getpid(), cases=3, witness=args.witness, levels=LEVELS))
    for row in rows:
        fine_row = refined_rows[row['id']]
        assert row['checkpoint_sha256'] == fine_row['checkpoint_sha256']
        source_inputs = source/row['inputs_file']
        assert sha(source_inputs) == row['inputs_sha256']
        folder = out/row['id']
        folder.mkdir()
        records = []
        for index, layer in enumerate(row['layers'][:1] if args.witness else row['layers']):
            file = source/layer['file']
            assert sha(file) == layer['sha256']
            with np.load(file) as bundle:
                entrance, weights, original, old_steps, old_endpoints = [bundle[key].copy() for key in
                    ['entrance','weights','references','steps','native_endpoints']]
            endpoints = {int(step):endpoint for step, endpoint in zip(old_steps, old_endpoints)}
            refined_bindings = []
            for level in fine_row['levels']:
                record = level['layers'][index]
                assert record['layer'] == layer['layer'] and Path(record['source_energy']) == file
                refined_file = refined/record['file']
                assert sha(refined_file) == record['sha256']
                with np.load(refined_file) as bundle:
                    endpoint = bundle['native_refined'].copy()
                if level['steps'] in endpoints:
                    assert np.allclose(endpoint, endpoints[level['steps']], atol=1e-11, rtol=1e-11)
                endpoints[level['steps']] = endpoint
                refined_bindings.append(dict(steps=level['steps'], file=str(refined_file), sha256=sha(refined_file)))
            references = [augmented_reference(entrance, weights, rtol, atol) for rtol, atol in TOLERANCES]
            augmented = np.stack([value[0] for value in references])
            qualified = qualification(augmented, original)
            target = folder/f'layer{index}.npz'
            np.savez_compressed(target, entrance=entrance, weights=weights,
                original_references=original, levels=np.asarray(LEVELS),
                native_endpoints=np.stack([endpoints[level] for level in LEVELS]),
                augmented=augmented, qualified=qualified)
            records.append(dict(layer=layer['layer'], file=str(target.relative_to(out)), sha256=sha(target),
                source_energy=str(file), source_energy_sha256=sha(file), refined_sources=refined_bindings,
                qualified_points=int(qualified.sum()), total_points=len(qualified),
                reference_evaluations=[value[1] for value in references]))
            dump(out/'progress.json', dict(pid=os.getpid(), completed_cases=len(results), cases=3,
                current=row['id'], completed_layers=len(records), layers=1 if args.witness else 6,
                qualified_points=int(qualified.sum()), seconds=time.monotonic()-started))
            print(row['id'], layer['layer'], 'qualified', int(qualified.sum()), '/', len(qualified), flush=True)
        results.append(dict(id=row['id'], dataset=row['dataset'], family=row['family'],
            checkpoint_sha256=row['checkpoint_sha256'], source_inputs=str(source_inputs),
            source_inputs_sha256=sha(source_inputs), layers=records))
        dump(out/'rows.json', dict(rows=results))
    dump(out/'COMPLETE.json', dict(passed=True, witness=args.witness, cases=3, inputs_per_case=80,
        layers_per_case=1 if args.witness else 6, levels=LEVELS,
        all_reference_points_qualified=all(record['qualified_points'] == record['total_points'] for row in results for record in row['layers']),
        rows_sha256=sha(out/'rows.json'), source_energy_complete_sha256=sha(source/'COMPLETE.json'),
        source_surrogate_complete_sha256=sha(refined/'COMPLETE.json'), seconds=time.monotonic()-started,
        source_sha256={name:sha(Path(__file__).with_name(name)) for name in ['solver_bias_probe.py','check_solver_bias.py']}))


if __name__ == '__main__':
    main()
