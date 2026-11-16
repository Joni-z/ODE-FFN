"""Independent NumPy summary reconstruction; does not import the metric implementation."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    complete, data = read(root/'COMPLETE.json'), read(root/'rows.json')
    assert complete['passed'] and complete['cases'] == 12
    assert complete['rows_sha256'] == sha(root/'rows.json')
    summaries, per_pair_errors, error = [], {}, 0.
    for case in data['cases']:
        file = root/case['raw_file']
        assert sha(file) == case['raw_sha256']
        with np.load(file) as z:
            arrays = {k: z[k].copy() for k in z.files}
        assert len(arrays['source']) == complete['points_per_time']
        by_time, point_errors = [], []
        for i, t in enumerate(arrays['time']):
            pred, oracle = arrays['prediction'][i], arrays['oracle'][i]
            j, truth = arrays['jacobian'][i], arrays['oracle_jacobian'][i]
            dim = pred.shape[1]
            assert np.array_equal(arrays['x'][i], (1-t)*arrays['source']+t*arrays['target'])
            assert np.max(np.abs(arrays['oracle_image_jacobian'][i]-(np.eye(dim)+(1-t)*truth))) < 1e-10
            image_j = np.eye(dim)+(1-t)*j
            energy = np.einsum('nij,nij->n', j-truth, j-truth)/dim
            point_errors.append(energy)
            antisymmetry = (j-np.swapaxes(j, 1, 2))/2
            skew_energy = np.einsum('nij,nij->n', antisymmetry, antisymmetry)/dim
            eigenvalues = np.linalg.eigvalsh((image_j+np.swapaxes(image_j, 1, 2))/2)
            difference = pred-oracle
            measured = dict(field_mse=float(np.mean(difference**2)),
                velocity_jacobian_error=float(energy.mean()), image_jacobian_error=float(((1-t)**2*energy).mean()),
                velocity_skew_energy=float(skew_energy.mean()), image_skew_energy=float(((1-t)**2*skew_energy).mean()),
                image_symmetric_negative_energy=float(np.square(np.clip(eigenvalues, None, 0)).sum()/len(pred)/dim),
                image_symmetric_negative_fraction=float(np.count_nonzero(eigenvalues < -1e-10)/eigenvalues.size),
                image_symmetric_min_eigenvalue=float(eigenvalues.min()),
                image_symmetric_eigenvalue_quantiles=np.percentile(eigenvalues, [5, 50, 95]).tolist(),
                mean_field_bias_energy=float(np.square(difference.sum(axis=0)/len(pred)).mean()))
            original = next(r for r in data['rows'] if r['id'] == case['id'] and r['time'] == t)
            for key, value in measured.items():
                current = float(np.max(np.abs(np.asarray(value)-np.asarray(original[key]))))
                error = max(error, current)
                assert np.allclose(value, original[key], atol=1e-11, rtol=1e-11), (case['id'], key, current)
            by_time.append(dict(time=float(t), **measured))
        summary = {key: float(np.mean([r[key] for r in by_time])) for key in measured
                   if key not in ['image_symmetric_eigenvalue_quantiles', 'image_symmetric_min_eigenvalue']}
        summaries.append(dict(id=case['id'], dataset=case['dataset'], family=case['family'],
            confirmation_pooled_sw2=case['confirmation_pooled_sw2'], **summary, by_time=by_time))
        per_pair_errors[case['id']] = np.asarray(point_errors).mean(axis=0)
    pairs = []
    for dataset in ['gmm_ring8_2d', 'rotated_product_bimodal_16d']:
        pal = next(r for r in summaries if r['dataset'] == dataset and r['family'] == 'palindrome_operator_midpoint4')
        alt = next(r for r in summaries if r['dataset'] == dataset and r['family'] == 'alternating_operator_midpoint4')
        gap = pal['velocity_jacobian_error']-alt['velocity_jacobian_error']
        expected_sign = -1 if dataset == 'gmm_ring8_2d' else 1
        difference = per_pair_errors[pal['id']]-per_pair_errors[alt['id']]
        assert abs(difference.mean()-gap) < 1e-11
        rng = np.random.default_rng(0)
        bootstrap = difference[rng.integers(0, len(difference), (10000, len(difference)))].mean(axis=1)
        pairs.append(dict(dataset=dataset, palindrome=pal['velocity_jacobian_error'],
            alternating=alt['velocity_jacobian_error'], primary_gap=gap, expected_gap_sign=expected_sign,
            predicted_direction_holds=bool(expected_sign*gap > 0),
            point_pair_difference_sd=float(difference.std(ddof=1)),
            conditional_pair_bootstrap95=np.quantile(bootstrap, [.025, .975]).tolist(),
            conditional_two_contrast_bonferroni=np.quantile(bootstrap, [.0125, .9875]).tolist(),
            sampling_gap=pal['confirmation_pooled_sw2']-alt['confirmation_pooled_sw2']))
    output = dict(passed=True, witness=complete['witness'], rows=summaries, matched_pairs=pairs,
        primary_explanation_passes_both_tasks=all(r['predicted_direction_holds'] for r in pairs),
        reconstruction_max_error=error, input_complete_sha256=sha(root/'COMPLETE.json'),
        input_rows_sha256=sha(root/'rows.json'), source_sha256=sha(Path(__file__)))
    (out/'summary.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    (out/'COMPLETE.json').write_text(json.dumps(dict(passed=True, cases=12, rows=complete['rows'],
        witness=complete['witness'], summary_sha256=sha(out/'summary.json'), max_error=error), indent=2)+'\n')
    print(json.dumps(dict(passed=True, rows=complete['rows'], reconstruction_max_error=error,
        matched_pairs=pairs)), flush=True)


if __name__ == '__main__':
    main()
