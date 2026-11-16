"""Independent full-frame geometry, finite-response metrics and bank contrasts."""
import argparse
import itertools
import math
from pathlib import Path
import numpy as np
from geometry import read, dump, sha


def conditional_intervals(denominator: np.ndarray, difference: np.ndarray) -> dict:
    if np.any(denominator <= 0):
        return dict(resolved=False, zero_energy_banks=np.flatnonzero(denominator <= 0).tolist())
    samples, weights = [], []
    for sample in itertools.combinations_with_replacement(range(8), 8):
        counts = np.bincount(sample, minlength=8)
        samples.append((difference@counts)/(denominator@counts))
        weights.append(math.factorial(8)//math.prod(math.factorial(int(count)) for count in counts))
    samples, weights = np.asarray(samples), np.asarray(weights, dtype=np.int64)
    assert len(samples) == 6435 and weights.sum() == 8**8
    intervals = {}
    for index, name in enumerate(['normal_minus_null', 'tangent_minus_null']):
        order = np.argsort(samples[:, index])
        cdf = np.cumsum(weights[order])
        def quantile(value: float) -> float:
            return float(samples[order[np.searchsorted(cdf, math.ceil(value*8**8))], index])
        intervals[name] = dict(conditional95=[quantile(.025), quantile(.975)],
            nominal_twelve_contrast_adjusted=[quantile(.05/24), quantile(1-.05/24)])
    return dict(resolved=True, ordered_resamples=8**8, intervals=intervals)


def ratio(numerator: np.ndarray, denominator: np.ndarray) -> list:
    return [float(n/d) if d > 0 else None for n, d in zip(numerator, denominator)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    complete, native = read(root/'COMPLETE.json'), read(root/'rows.json')
    assert complete['passed'] and complete['cases'] == 3 and complete['directions'] == 64
    assert sha(root/'rows.json') == complete['rows_sha256']
    expected_layers = 1 if complete['witness'] else 6
    frame_manifest = Path(native['rows'][0]['frame_file']).parent/'manifest.json'
    assert sha(frame_manifest) == complete['frames_manifest_sha256']
    rng = np.random.default_rng(0)
    for row in read(frame_manifest)['rows']:
        assert sha(Path(row['file'])) == row['sha256']
        with np.load(row['file']) as bundle:
            frames = bundle['frames'].copy()
        for frame in frames:
            q, r = np.linalg.qr(rng.standard_normal((64, 64)))
            expected = (q*np.where(np.diag(r) >= 0, 1., -1.)).T
            assert np.allclose(frame, expected, atol=1e-14, rtol=1e-14)
    rows, maximum = [], 0.
    for row in native['rows']:
        assert sha(Path(row['source_inputs'])) == row['source_inputs_sha256']
        with np.load(row['source_inputs']) as bundle:
            inputs = {key:bundle[key].copy() for key in bundle.files}
        banks, baseline = inputs['bank'], inputs['baseline']
        assert len(banks) == 80 and np.array_equal(np.bincount(banks), np.full(8, 10))
        with np.load(row['frame_file']) as bundle:
            bound_frames = bundle['frames'].copy()
        assert sha(Path(row['frame_file'])) == row['frame_sha256']
        den_bank = np.zeros(8)
        actual_residual, null_residual = np.zeros((2, 8)), np.zeros((2, 8))
        actual_response, null_response = np.zeros(2), np.zeros(2)
        actual_feature, null_feature = np.zeros(2), np.zeros(2)
        layers = []
        assert len(row['layers']) == expected_layers
        for index, item in enumerate(row['layers']):
            file, source = root/item['file'], Path(item['source_readout'])
            assert sha(file) == item['sha256'] and sha(source) == item['source_readout_sha256']
            assert sha(Path(item['source_energy'])) == item['source_energy_sha256']
            with np.load(file) as bundle:
                local, fields, frame = [bundle[key].copy() for key in ['local_outputs', 'fields', 'frame']]
            with np.load(source) as bundle:
                actual, old_fields = bundle['local_outputs'].copy(), bundle['fields'].copy()
            assert local.shape == (2, 64, 80, 64) and fields.shape == (2, 64, *baseline.shape)
            assert np.isfinite(local).all() and np.isfinite(fields).all()
            assert np.array_equal(frame, bound_frames[index]) and item['frame_index'] == index
            assert np.allclose(frame.T@frame, np.eye(64), atol=1e-12, rtol=0)
            assert np.array_equal(old_fields[0], baseline)
            coarse, reference = actual[:2]
            delta = coarse-reference
            projection = (frame@delta.T)[:, :, None]*frame[:, None, :]
            for measured, expected in [(local[0], coarse[None]-projection), (local[1], reference[None]+projection)]:
                error = float(np.max(np.abs(measured-expected)))
                maximum = max(maximum, error)
                assert np.allclose(measured, expected, atol=1e-11, rtol=1e-11)
            feature_squared = np.sum((coarse[None, None]-local)**2, axis=-1)
            target_feature = np.stack([np.sum(delta*delta, axis=1)/64, 63*np.sum(delta*delta, axis=1)/64])
            assert np.allclose(feature_squared.mean(axis=1), target_feature, atol=1e-11, rtol=1e-11)
            displacement = np.sum((old_fields[0]-old_fields[1])**2, axis=1)
            actual_r = np.sum((old_fields[2:4]-old_fields[1])**2, axis=-1)
            null_r = np.sum((fields-old_fields[1])**2, axis=-1)
            d = float(displacement.sum())
            actual_f = (1-actual_r.sum(axis=1)/d).tolist() if d else [None]*2
            null_f = (1-null_r.mean(axis=1).sum(axis=1)/d).tolist() if d else [None]*2
            direction_f = (1-null_r.sum(axis=2)/d).tolist() if d else [[None]*64]*2
            layers.append(dict(layer=item['layer'], actual_fractions=actual_f, null_mean_fractions=null_f,
                null_direction_fractions=direction_f, response_energy=d,
                exact_frame_feature_error=float(np.max(np.abs(feature_squared.mean(axis=1)-target_feature)))))
            den_bank += np.bincount(banks, weights=displacement, minlength=8)
            for mode in range(2):
                actual_residual[mode] += np.bincount(banks, weights=actual_r[mode], minlength=8)
                null_residual[mode] += np.bincount(banks, weights=null_r[mode].mean(axis=0), minlength=8)
            actual_response += np.sum((old_fields[2:4]-old_fields[0])**2, axis=(1, 2))
            null_response += np.mean(np.sum((fields-old_fields[0])**2, axis=(2, 3)), axis=1)
            actual_feature += np.sum((coarse[None]-actual[2:4])**2, axis=(1, 2))
            null_feature += feature_squared.mean(axis=1).sum(axis=1)
            assert item['exact_upstream'] and item['exact_native_output'] and item['actual_component_outputs_exact']
        denominator = float(den_bank.sum())
        resolved = denominator > 0
        actual_f = (1-actual_residual.sum(axis=1)/denominator).tolist() if resolved else [None]*2
        null_f = (1-null_residual.sum(axis=1)/denominator).tolist() if resolved else [None]*2
        contrasts = ((null_residual-actual_residual).sum(axis=1)/denominator).tolist() if resolved else [None]*2
        assert row['parameters_and_buffers_unchanged'] and row['native_roundtrip_exact']
        rows.append(dict(id=row['id'], dataset=row['dataset'], family=row['family'], layers=layers,
            actual_fractions=actual_f, null_mean_fractions=null_f, contrasts=contrasts,
            normal_specificity_point=bool(resolved and contrasts[0] > 0),
            tangential_specificity_point=bool(resolved and contrasts[1] > 0),
            actual_finite_response_gain=ratio(actual_response, actual_feature),
            null_finite_response_gain=ratio(null_response, null_feature),
            response_energy_by_bank=den_bank.tolist(), actual_residual_by_bank=actual_residual.tolist(),
            null_residual_by_bank=null_residual.tolist(),
            uncertainty=conditional_intervals(den_bank, null_residual-actual_residual)))
    summary = dict(passed=True, witness=complete['witness'], cases=3, rows=rows,
        frames_reconstruct=True, max_candidate_reconstruction_error=maximum,
        input_complete_sha256=sha(root/'COMPLETE.json'), source_sha256=sha(Path(__file__)))
    dump(out/'summary.json', summary)
    dump(out/'COMPLETE.json', dict(passed=True, witness=complete['witness'], cases=3,
        summary_sha256=sha(out/'summary.json'), max_error=maximum))
    print('PASS complete-frame audit:', len(rows), 'cases; max candidate error', maximum, flush=True)


if __name__ == '__main__':
    main()
