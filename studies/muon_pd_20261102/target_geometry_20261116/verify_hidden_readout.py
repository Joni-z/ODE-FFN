"""Independent candidate/field-metric/bank-interval audit of single-layer probes."""
import argparse
import itertools
import math
from pathlib import Path
import numpy as np
from geometry import read, dump, sha


def potential(value: np.ndarray) -> np.ndarray:
    return np.sum(value*value/(1+np.sqrt(1+1.2*value*value)), axis=-1)


def intervals(denominator: np.ndarray, numerator: np.ndarray) -> dict:
    if np.any(denominator <= 0):
        return dict(resolved=False, zero_energy_banks=np.flatnonzero(denominator <= 0).tolist())
    values, weights = [], []
    for sample in itertools.combinations_with_replacement(range(8), 8):
        count = np.bincount(sample, minlength=8)
        d = count@denominator
        n, t, r = numerator@count
        values.append([1-t/d, (n-t)/d, (r-t)/d])
        weights.append(math.factorial(8)//math.prod(math.factorial(int(c)) for c in count))
    assert len(values) == 6435 and sum(weights) == 8**8
    values, weights = np.asarray(values), np.asarray(weights, dtype=np.int64)
    result = {}
    for i, name in enumerate(['tangential_fraction', 'tangential_minus_normal', 'tangential_minus_ray']):
        order = np.argsort(values[:, i])
        cdf = np.cumsum(weights[order])
        def at(q):
            return float(values[order[np.searchsorted(cdf, math.ceil(q*8**8))], i])
        result[name] = dict(conditional95=[at(.025), at(.975)],
            nominal_eighteen_contrast_adjusted=[at(.05/36), at(1-.05/36)])
    return dict(resolved=True, ordered_resamples=8**8, intervals=result)


def main() -> None:
    p = argparse.ArgumentParser()
    for key in ['input', 'output']:
        p.add_argument('--'+key, required=True)
    a = p.parse_args()
    root, out = Path(a.input), Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    complete, data = read(root/'COMPLETE.json'), read(root/'rows.json')
    assert complete['passed'] and not complete['check_only'] and complete['cases'] == 3
    assert sha(root/'rows.json') == complete['rows_sha256']
    assert complete['modes'] == ['identity', 'reference', 'normal', 'tangential', 'ray']
    expected_layers = 1 if complete['witness'] else 6
    rows, max_error = [], 0.
    for row in data['rows']:
        file = root/row['inputs_file']
        assert sha(file) == row['inputs_sha256']
        assert sha(Path(row['source_inputs'])) == row['source_inputs_sha256']
        with np.load(file) as z:
            inputs = {k: z[k].copy() for k in z.files}
        with np.load(row['source_inputs']) as z:
            for key in z.files:
                assert np.array_equal(inputs[key], z[key]), key
        assert np.array_equal(inputs['baseline'], inputs['native_prediction'])
        bank = inputs['bank']
        assert len(bank) == 80 and np.array_equal(np.bincount(bank, minlength=8), np.full(8, 10))
        baseline, oracle = inputs['baseline'], inputs['oracle']
        assert baseline.shape == oracle.shape and np.isfinite(oracle).all()
        denominator_bank = np.zeros(8)
        numerator_bank = np.zeros((3, 8))
        layer_rows, unresolved_ray = [], 0
        assert len(row['layers']) == expected_layers
        for record in row['layers']:
            file, original = root/record['file'], Path(record['source_layer'])
            assert sha(file) == record['sha256'] and sha(original) == record['source_layer_sha256']
            with np.load(file) as z:
                local, fields, scales, ray_qualified = [z[k].copy() for k in ['local_outputs', 'fields', 'ray_scale', 'ray_qualified']]
            with np.load(original) as z:
                native, ref = z['native_forward'].copy(), z['references'][-1].copy()
            assert local.shape == (5, 80, 64) and fields.shape == (5, *baseline.shape)
            assert np.isfinite(local).all() and np.isfinite(fields).all() and np.isfinite(scales).all()
            assert np.array_equal(local[0], native) and np.array_equal(local[1], ref)
            assert np.array_equal(fields[0], baseline)
            gradient = ref*np.power(1+1.2*ref*ref, -.5)
            norm2 = np.einsum('nd,nd->n', gradient, gradient)
            difference = native-ref
            dot = np.einsum('nd,nd->n', difference, gradient)
            coefficient = np.zeros(80)
            coefficient[norm2 != 0] = dot[norm2 != 0]/norm2[norm2 != 0]
            normal = gradient*coefficient[:, None]
            expected = [native-normal, ref+normal]
            for measured, target in zip(local[2:4], expected):
                error = float(np.max(np.abs(measured-target)))
                max_error = max(max_error, error)
                assert np.allclose(measured, target, atol=1e-11, rtol=1e-11)
            assert np.all(scales >= 0) and np.array_equal(local[4], scales[:, None]*native)
            target_energy, ray_energy = potential(ref), potential(local[4])
            impossible = (np.square(native).sum(1) == 0)&(target_energy != 0)
            assert np.array_equal(ray_qualified, ~impossible)
            assert np.all(np.abs(ray_energy[ray_qualified]-target_energy[ray_qualified]) <= 1e-11*(1+np.abs(target_energy[ray_qualified])))
            unresolved_ray += int((~ray_qualified).sum())
            displacement = np.sum((fields[0]-fields[1])**2, axis=1)
            residual = np.sum((fields[2:]-fields[1][None])**2, axis=2)
            for b in range(8):
                denominator_bank[b] += displacement[bank == b].sum()
                numerator_bank[:, b] += residual[:, bank == b].sum(1)
            denominator = float(displacement.sum())
            fractions = (1-residual.sum(1)/denominator).tolist() if denominator else [None]*3
            layer_rows.append(dict(layer=record['layer'], fractions=dict(zip(['normal', 'tangential', 'ray'], fractions)),
                response_mse=float(displacement.mean()/baseline.shape[1]),
                oracle_mse_by_mode=[float(np.mean((value-oracle)**2)) for value in fields],
                ray_potential_max_error=float(np.max(np.abs(ray_energy[ray_qualified]-target_energy[ray_qualified]))) if ray_qualified.any() else None,
                input_sha256=sha(file)))
        denominator = float(denominator_bank.sum())
        fractions = (1-numerator_bank.sum(1)/denominator).tolist() if denominator else [None]*3
        condition = denominator > 0 and unresolved_ray == 0
        point_support = condition and fractions[1] > 0 and fractions[1] > fractions[0] and fractions[1] > fractions[2]
        rows.append(dict(id=row['id'], dataset=row['dataset'], family=row['family'],
            layers=layer_rows, fractions=dict(zip(['normal', 'tangential', 'ray'], fractions)),
            response_mse=denominator/(expected_layers*80*baseline.shape[1]),
            bank_displacement_sums=denominator_bank.tolist(), bank_residual_sums=numerator_bank.tolist(),
            uncertainty=intervals(denominator_bank, numerator_bank), unresolved_ray_points=unresolved_ray,
            direction_hypothesis_point_support=bool(point_support), native_oracle_mse=float(np.mean((baseline-oracle)**2)),
            baseline_and_roundtrip_exact=row['native_roundtrip_exact'], parameters_and_buffers_unchanged=row['parameters_and_buffers_unchanged']))
    result = dict(passed=True, witness=complete['witness'], cases=3, rows=rows,
        all_source_arrays_exact=True, max_candidate_reconstruction_error=max_error,
        input_complete_sha256=sha(root/'COMPLETE.json'), source_sha256=sha(Path(__file__)))
    dump(out/'summary.json', result)
    dump(out/'COMPLETE.json', dict(passed=True, witness=complete['witness'], cases=3,
        summary_sha256=sha(out/'summary.json'), max_error=max_error))
    print('PASS independent hidden readout:', len(rows), 'cases, max candidate error', max_error, flush=True)


if __name__ == '__main__':
    main()
