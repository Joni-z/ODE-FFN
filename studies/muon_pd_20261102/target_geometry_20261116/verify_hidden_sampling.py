"""Reconstruct finite-direction distribution effects and numerical pilot gates."""
import argparse
from pathlib import Path
import numpy as np
from geometry import read, dump, sha
from verify_hidden_surrogate import solve

MODES = ('identity', 'reference', 'normal', 'tangential', 'ray')


def energy(y: np.ndarray) -> np.ndarray:
    return np.sum(y*y/(np.sqrt(1+1.2*y*y)+1), axis=-1)


def quantiles(x: np.ndarray, directions: np.ndarray) -> np.ndarray:
    return np.sort(x@directions.T, axis=0)


def sw2(x: np.ndarray, y: np.ndarray, directions: np.ndarray) -> float:
    return float(np.sqrt(np.mean((quantiles(x, directions)-quantiles(y, directions))**2)))


def fraction(native: np.ndarray, reference: np.ndarray, value: np.ndarray, directions: np.ndarray):
    denominator = np.mean((quantiles(native, directions)-quantiles(reference, directions))**2)
    return float(1-np.mean((quantiles(value, directions)-quantiles(reference, directions))**2)/denominator) if denominator > 0 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    complete = read(root/'COMPLETE.json')
    assert complete['passed'] and not complete['check_only']
    assert complete['first_layer'] == 'blocks.0.ffn.ode1'
    assert complete['inner_steps'] == [256, 512] and complete['heun_steps'] == [128, 129]
    assert sha(root/'rows.json') == complete['rows_sha256']
    rows, max_reconstruction = [], 0.
    for row in read(root/'rows.json')['rows']:
        inputs_file = root/row['inputs_file']
        assert sha(inputs_file) == row['inputs_sha256']
        with np.load(inputs_file) as bundle:
            source, target, banks, directions = [bundle[key].copy() for key in ['source', 'target', 'bank', 'directions']]
        assert len(source) == 8*complete['per_bank']
        assert np.array_equal(np.bincount(banks), np.full(8, complete['per_bank']))
        assert sha(Path(row['assets_file'])) == row['assets_sha256']
        with np.load(row['assets_file']) as bundle:
            for key, measured in [('source', source), ('target', target)]:
                expected = np.concatenate([bundle[f'{key}_{bank}'][:complete['per_bank']] for bank in range(8)]).astype(float)
                assert np.array_equal(measured, expected)
            assert np.array_equal(directions, bundle['directions'])
        endpoints, arm_rows = {}, []
        assert len(row['arms']) == (18 if row['focal'] else 2)
        for arm in row['arms']:
            file = root/arm['file']
            assert sha(file) == arm['sha256']
            with np.load(file) as bundle:
                z = {key: bundle[key].copy() for key in bundle.files}
            value = z['endpoints']
            assert value.shape == source.shape and np.isfinite(value).all()
            key = arm['heun_steps'], arm['inner_steps'], arm['mode']
            assert key not in endpoints
            endpoints[key] = value
            snapshot_metrics = {}
            if row['focal']:
                h, coarse, refs, work = [z['snapshot_'+key] for key in ['entrance', 'native', 'references', 'reference_work']]
                local, fields = z['snapshot_dop_local'], z['snapshot_dop_fields']
                actual_local, actual_field = z['snapshot_local'], z['snapshot_field']
                assert h.shape == (5, len(source), 64) and refs.shape == (5, 2, len(source), 64)
                assert np.array_equal(z['snapshot_call'], [0, 64, 128, 192, 2*arm['heun_steps']-1])
                assert np.array_equal(z['snapshot_x'][0], source)
                for index, call in enumerate(z['snapshot_call']):
                    t = (call//2+call % 2)/arm['heun_steps']
                    assert np.allclose(z['snapshot_time'][index], t, atol=1e-15, rtol=0)
                independent = solve(h.reshape(-1, 64), z['weights'], 4).reshape(h.shape)
                assert np.allclose(independent, coarse, atol=1e-11, rtol=1e-11)
                if arm['inner_steps']:
                    independent = solve(h.reshape(-1, 64), z['weights'], arm['inner_steps']).reshape(h.shape)
                    assert np.allclose(independent, z['snapshot_refined'], atol=1e-11, rtol=1e-11)
                assert np.array_equal(local[:, 0], coarse) and np.array_equal(local[:, 1], refs[:, -1])
                gradient = refs[:, -1]/np.sqrt(1+1.2*refs[:, -1]**2)
                diff = coarse-refs[:, -1]
                norm2 = np.sum(gradient*gradient, axis=-1)
                coeff = np.divide(np.sum(diff*gradient, axis=-1), norm2, out=np.zeros_like(norm2), where=norm2 != 0)
                normal = coeff[..., None]*gradient
                for saved, expected in [(local[:, 2], coarse-normal), (local[:, 3], refs[:, -1]+normal)]:
                    error = float(np.max(np.abs(saved-expected)))
                    max_reconstruction = max(max_reconstruction, error)
                    assert np.allclose(saved, expected, atol=1e-11, rtol=1e-11)
                ray = local[:, 4]
                scale = np.divide(np.sum(ray*coarse, axis=-1), np.sum(coarse*coarse, axis=-1),
                                  out=np.zeros(coarse.shape[:-1]), where=np.sum(coarse*coarse, axis=-1) != 0)
                assert np.all(scale >= -1e-12) and np.allclose(ray, scale[..., None]*coarse, atol=1e-10, rtol=1e-10)
                assert np.all(np.abs(energy(ray)-energy(refs[:, -1])) <= 1e-11*(1+np.abs(energy(refs[:, -1]))))
                closure = energy(refs[:, -1])-energy(h)-work[:, -1]
                qualified = np.all(np.abs(refs[:, 0]-refs[:, 1]) <= 1e-9+1e-9*np.abs(refs[:, 1]), axis=-1)
                qualified &= np.abs(work[:, 0]-work[:, 1]) <= 1e-8*(1+np.abs(work[:, 1]))
                qualified &= np.abs(closure) <= 1e-8*(1+np.abs(energy(h))+np.abs(energy(refs[:, -1]))+np.abs(work[:, -1]))
                assert np.array_equal(qualified, z['snapshot_reference_qualified'])
                index = MODES.index(arm['mode'])
                local_ratio = np.linalg.norm(actual_local-local[:, index], axis=-1)/(1e-9+1e-3*np.linalg.norm(coarse-refs[:, -1], axis=-1))
                output_ratio = np.linalg.norm(actual_field-fields[:, index], axis=-1)/(1e-10+1e-3*np.linalg.norm(fields[:, 0]-fields[:, 1], axis=-1))
                assert np.allclose(local_ratio, z['snapshot_local_limit_ratio'], atol=1e-10, rtol=1e-10)
                assert np.allclose(output_ratio, z['snapshot_output_limit_ratio'], atol=1e-10, rtol=1e-10)
                snapshot_metrics = dict(reference_failures=int((~qualified).sum()),
                    local_failures=int((local_ratio > 1).sum()), output_failures=int((output_ratio > 1).sum()),
                    snapshot_qualified=bool(qualified.all() and (local_ratio <= 1).all() and (output_ratio <= 1).all()),
                    local_max_limit_ratio=float(local_ratio.max()), output_max_limit_ratio=float(output_ratio.max()))
                for metric, measured in snapshot_metrics.items():
                    if isinstance(measured, float):
                        assert np.isclose(measured, arm[metric], atol=1e-9, rtol=1e-9)
                    else:
                        assert measured == arm[metric]
            arm_rows.append(dict(mode=arm['mode'], inner_steps=arm['inner_steps'], heun_steps=arm['heun_steps'],
                target_sw2=sw2(value, target, directions),
                bank_target_sw2=[sw2(value[banks == bank], target[banks == bank], directions) for bank in range(8)],
                **snapshot_metrics, seconds=arm['seconds']))
        comparisons = []
        if row['focal']:
            for outer in (128, 129):
                native = endpoints[outer, 0, 'identity']
                numerical_points, g_difference = [], []
                fractions_by_level = {}
                for inner in (256, 512):
                    ref = endpoints[outer, inner, 'reference']
                    fractions_by_level[inner] = []
                    for mask in [np.ones(len(source), dtype=bool)]+[banks == bank for bank in range(8)]:
                        fractions_by_level[inner].append([fraction(native[mask], ref[mask], endpoints[outer, inner, mode][mask], directions)
                                                        for mode in MODES[2:]])
                valid = all(value is not None for values in fractions_by_level.values() for group in values for value in group)
                if valid:
                    g_difference = np.abs(np.asarray(fractions_by_level[256])-np.asarray(fractions_by_level[512]))
                for mode in MODES[1:]:
                    fine = endpoints[outer, 512, mode]
                    change = np.linalg.norm(endpoints[outer, 256, mode]-fine, axis=-1)
                    scale = 1e-7+.01*np.linalg.norm(fine-native, axis=-1)
                    numerical_points.append(change/scale)
                numerical_ratio = float(np.max(numerical_points))
                snapshots_ok = all(arm['snapshot_qualified'] for arm in arm_rows if arm['heun_steps'] == outer)
                # Compare the measured refinement to its numerical shift, preserving unresolved cases.
                ref256, ref512 = endpoints[outer, 256, 'reference'], endpoints[outer, 512, 'reference']
                effect_resolved = all(sw2(native[mask], ref512[mask], directions) >
                    1e-7+100*sw2(ref256[mask], ref512[mask], directions)
                    for mask in [np.ones(len(source), dtype=bool)]+[banks == bank for bank in range(8)])
                qualified = bool(valid and snapshots_ok and numerical_ratio <= 1 and np.max(g_difference) <= .005 and effect_resolved)
                point = fractions_by_level[512][0]
                support = bool(valid and point[1] > 0 and point[1] > point[0] and point[1] > point[2])
                comparisons.append(dict(heun_steps=outer, fractions_by_level={str(k):v for k,v in fractions_by_level.items()},
                    pointwise_endpoint_max_limit_ratio=numerical_ratio,
                    distribution_fraction_max_resolution_change=float(np.max(g_difference)) if valid else None,
                    snapshot_gates_pass=snapshots_ok, effect_resolved=effect_resolved,
                    numerically_qualified=qualified, direction_hypothesis_point_support=support))
        assert row['parameters_and_buffers_unchanged'] and row['native_roundtrip_exact']
        rows.append(dict(id=row['id'], dataset=row['dataset'], family=row['family'], focal=row['focal'],
            arms=arm_rows, comparisons=comparisons, sources=len(source)))
    result = dict(passed=True, witness=complete['witness'], cases=len(rows), rows=rows,
        max_candidate_reconstruction_error=max_reconstruction,
        input_complete_sha256=sha(root/'COMPLETE.json'), source_sha256=sha(Path(__file__)))
    dump(out/'summary.json', result)
    dump(out/'COMPLETE.json', dict(passed=True, witness=complete['witness'], cases=len(rows),
        summary_sha256=sha(out/'summary.json'), max_error=max_reconstruction))
    print('PASS independent first-layer sampling:', len(rows), 'cases, candidate error', max_reconstruction, flush=True)


if __name__ == '__main__':
    main()
