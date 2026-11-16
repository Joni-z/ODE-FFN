"""Independent state-only perturbation check and complete prediction metrics."""
import argparse
import itertools
import math
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from geometry import read, dump, sha


def perturbed_flow(entrance: np.ndarray, weights: np.ndarray, epsilon: float) -> np.ndarray:
    shape, state = entrance.shape, entrance.ravel().copy()
    for stage, matrix in enumerate(weights):
        def derivative(time: float, packed: np.ndarray) -> np.ndarray:
            y = packed.reshape(shape)
            root = np.sqrt(1+1.2*y*y)
            first, second = root**-3, -3.6*y*root**-5
            apply = lambda value:np.einsum('ij,bj->bi', matrix, value, optimize=True)
            field = apply(y/root)
            defect = -apply(second*field*field)/24-apply(first*apply(first*field))/6
            return (field+epsilon*defect).ravel()
        result = solve_ivp(derivative, (stage/4, (stage+1)/4), state,
                           method='DOP853', rtol=1e-12, atol=1e-13)
        assert result.success and np.isfinite(result.y).all(), result.message
        state = result.y[:, -1]
    return state.reshape(shape)


def intervals(denominator: np.ndarray, residual: np.ndarray) -> dict:
    if np.any(denominator <= 0):
        return dict(resolved=False)
    values, weights = [], []
    for sample in itertools.combinations_with_replacement(range(8), 8):
        counts = np.bincount(sample, minlength=8)
        values.append(1-(residual@counts)/(denominator@counts))
        weights.append(math.factorial(8)//math.prod(math.factorial(int(count)) for count in counts))
    values, weights = np.asarray(values), np.asarray(weights, dtype=np.int64)
    assert len(values) == 6435 and weights.sum() == 8**8
    result = {}
    for index, name in enumerate(['trained_total_fraction', 'trained_normal_fraction']):
        order = np.argsort(values[:, index])
        cdf = np.cumsum(weights[order])
        def at(q: float) -> float:
            return float(values[order[np.searchsorted(cdf, math.ceil(q*8**8))], index])
        result[name] = dict(conditional95=[at(.025), at(.975)],
            nominal_twelve_contrast_adjusted=[at(.05/24), at(1-.05/24)])
    return dict(resolved=True, ordered_resamples=8**8, intervals=result)


def fraction(denominator: float, residual: float):
    return float(1-residual/denominator) if denominator > 0 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    complete = read(root/'COMPLETE.json')
    assert complete['passed'] and complete['cases'] == 3 and complete['inputs_per_case'] == 80
    assert complete['levels'] == [4, 8, 16, 32, 64, 128, 256]
    assert sha(root/'rows.json') == complete['rows_sha256']
    rows = []
    for row in read(root/'rows.json')['rows']:
        assert sha(Path(row['source_inputs'])) == row['source_inputs_sha256']
        with np.load(row['source_inputs']) as bundle:
            banks = bundle['bank'].copy()
        assert len(banks) == 80 and np.array_equal(np.bincount(banks), np.full(8, 10))
        layers = []
        denominator_bank, residual_bank = np.zeros((7, 2, 8)), np.zeros((7, 2, 8))
        predicted_normal, predicted_total = np.zeros(7), np.zeros(7)
        curvature_residual, chain_residual = np.zeros(7), np.zeros(7)
        max_fd_ratio, qualified_points, total_points = 0., 0, 0
        folder = out/row['id']
        folder.mkdir()
        assert len(row['layers']) == (1 if complete['witness'] else 6)
        for index, item in enumerate(row['layers']):
            file = root/item['file']
            assert sha(file) == item['sha256'] and sha(Path(item['source_energy'])) == item['source_energy_sha256']
            with np.load(file) as bundle:
                z = {key:bundle[key].copy() for key in bundle.files}
            h, weights, augmented = z['entrance'], z['weights'], z['augmented']
            assert h.shape == (80, 64) and augmented.shape == (2, 3, 80, 64)
            with np.load(item['source_energy']) as bundle:
                assert np.array_equal(h, bundle['entrance']) and np.array_equal(weights, bundle['weights'])
                assert np.array_equal(z['original_references'], bundle['references'])
                for step in [4, 8]:
                    position = list(bundle['steps']).index(step)
                    assert np.array_equal(z['native_endpoints'][list(z['levels']).index(step)], bundle['native_endpoints'][position])
            for record in item['refined_sources']:
                assert sha(Path(record['file'])) == record['sha256']
                with np.load(record['file']) as bundle:
                    assert np.array_equal(z['native_endpoints'][list(z['levels']).index(record['steps'])], bundle['native_refined'])
            reference_ok = np.all(np.abs(augmented[0, 0]-augmented[1, 0]) <= 1e-9+1e-9*np.abs(augmented[1, 0]), axis=1)
            reference_ok &= np.all(np.abs(augmented[:, 0]-z['original_references'][-1]) <= 1e-9+1e-9*np.abs(z['original_references'][-1]), axis=(0, 2))
            reference_ok &= np.all(np.abs(augmented[0, 1:]-augmented[1, 1:]) <= 1e-8+1e-8*np.abs(augmented[1, 1:]), axis=(0, 2))
            assert np.array_equal(reference_ok, z['qualified'])
            coefficient = augmented[-1, 1:].sum(axis=0)
            finite, perturbations = [], []
            for epsilon in [1e-4, 1e-5]:
                positive = perturbed_flow(h, weights, epsilon)
                negative = perturbed_flow(h, weights, -epsilon)
                finite.append((positive-negative)/(2*epsilon))
                perturbations.append(np.stack([positive, negative]))
            finite = np.stack(finite)
            fd_ratio = np.abs(finite-coefficient)/(2e-6+2e-5*np.abs(coefficient))
            fd_ok = np.all(fd_ratio <= 1, axis=(0, 2))
            qualified = reference_ok & fd_ok
            auxiliary = folder/f'layer{index}-perturbation.npz'
            np.savez_compressed(auxiliary, epsilon=np.array([1e-4, 1e-5]),
                endpoints=np.stack(perturbations), finite_derivative=finite, ratio=fd_ratio, qualified=qualified)
            max_fd_ratio = max(max_fd_ratio, float(fd_ratio.max()))
            qualified_points += int(qualified.sum())
            total_points += len(qualified)
            native, reference = z['native_endpoints'], z['original_references'][-1]
            actual = native-reference[None]
            steps = z['levels'].astype(float)
            prediction = coefficient[None]/steps[:, None, None]**2
            q = reference/np.sqrt(1+1.2*reference*reference)
            qnorm2 = np.sum(q*q, axis=1)
            actual_dot, prediction_dot = np.einsum('knd,nd->kn', actual, q), np.einsum('knd,nd->kn', prediction, q)
            actual_normal = np.divide(actual_dot**2, qnorm2, out=np.zeros_like(actual_dot), where=qnorm2 != 0)
            residual_normal = np.divide((actual_dot-prediction_dot)**2, qnorm2, out=np.zeros_like(actual_dot), where=qnorm2 != 0)
            prediction_normal = np.divide(prediction_dot**2, qnorm2, out=np.zeros_like(actual_dot), where=qnorm2 != 0)
            actual_energy = np.sum(actual*actual, axis=-1)
            residual_energy = np.sum((actual-prediction)**2, axis=-1)
            predicted_normal += prediction_normal.sum(axis=1)
            predicted_total += np.sum(prediction*prediction, axis=(1, 2))
            curvature_residual += np.sum((actual-augmented[-1, 1][None]/steps[:, None, None]**2)**2, axis=(1, 2))
            chain_residual += np.sum((actual-augmented[-1, 2][None]/steps[:, None, None]**2)**2, axis=(1, 2))
            level_rows = []
            for k, step in enumerate(z['levels']):
                for component, (den, res) in enumerate([(actual_energy[k], residual_energy[k]), (actual_normal[k], residual_normal[k])]):
                    denominator_bank[k, component] += np.bincount(banks, weights=den, minlength=8)
                    residual_bank[k, component] += np.bincount(banks, weights=res, minlength=8)
                level_rows.append(dict(steps=int(step), total_fraction=fraction(actual_energy[k].sum(), residual_energy[k].sum()),
                    normal_fraction=fraction(actual_normal[k].sum(), residual_normal[k].sum()),
                    actual_normal_share=float(actual_normal[k].sum()/actual_energy[k].sum()) if actual_energy[k].sum() else None))
            layers.append(dict(layer=item['layer'], levels=level_rows, qualified_points=int(qualified.sum()),
                points=len(qualified), fd_max_limit_ratio=float(fd_ratio.max()),
                perturbation_file=str(auxiliary.relative_to(out)), perturbation_sha256=sha(auxiliary),
                source_native=str(file), source_native_sha256=sha(file)))
            print(row['id'], item['layer'], 'independent qualified', int(qualified.sum()), '/', len(qualified), flush=True)
        denominator, residual = denominator_bank.sum(axis=2), residual_bank.sum(axis=2)
        levels = []
        for index, step in enumerate(complete['levels']):
            levels.append(dict(steps=step, total_fraction=fraction(denominator[index, 0], residual[index, 0]),
                normal_fraction=fraction(denominator[index, 1], residual[index, 1]),
                relative_total_residual=float(np.sqrt(residual[index, 0]/denominator[index, 0])) if denominator[index, 0] else None,
                actual_normal_share=float(denominator[index, 1]/denominator[index, 0]) if denominator[index, 0] else None,
                predicted_normal_share=float(predicted_normal[index]/predicted_total[index]) if predicted_total[index] else None,
                curvature_only_fraction=fraction(denominator[index, 0], curvature_residual[index]),
                chain_only_fraction=fraction(denominator[index, 0], chain_residual[index])))
        asymptotic = all(levels[i]['relative_total_residual'] is not None and levels[i+1]['relative_total_residual'] is not None
                         and levels[i]['relative_total_residual'] > levels[i+1]['relative_total_residual'] for i in [4, 5])
        trained = levels[0]
        rows.append(dict(id=row['id'], dataset=row['dataset'], family=row['family'], layers=layers, levels=levels,
            qualified_points=qualified_points, total_points=total_points, all_qualified=qualified_points == total_points,
            fd_max_limit_ratio=max_fd_ratio, asymptotic_point_support=bool(asymptotic),
            trained_total_point_support=trained['total_fraction'] is not None and trained['total_fraction'] > 0,
            trained_normal_point_support=trained['normal_fraction'] is not None and trained['normal_fraction'] > 0,
            trained_actual_energy_by_bank=denominator_bank[0].tolist(),
            trained_residual_energy_by_bank=residual_bank[0].tolist(),
            uncertainty=intervals(denominator_bank[0], residual_bank[0])))
    summary = dict(passed=True, witness=complete['witness'], cases=3, rows=rows,
        all_independent_points_qualified=all(row['all_qualified'] for row in rows),
        input_complete_sha256=sha(root/'COMPLETE.json'), source_sha256=sha(Path(__file__)))
    dump(out/'summary.json', summary)
    dump(out/'COMPLETE.json', dict(passed=True, witness=complete['witness'], cases=3,
        all_independent_points_qualified=summary['all_independent_points_qualified'], summary_sha256=sha(out/'summary.json')))
    print('PASS independent hidden-bias prediction:', len(rows), 'cases; all qualified', summary['all_independent_points_qualified'], flush=True)


if __name__ == '__main__':
    main()
