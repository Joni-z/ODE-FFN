"""Independent NumPy endpoint, candidate and qualification reconstruction."""
import argparse
from pathlib import Path
import numpy as np
from geometry import read, dump, sha


def energy(y: np.ndarray) -> np.ndarray:
    return np.sum(y*y/(1+np.sqrt(1+1.2*y*y)), axis=-1)


def solve(y0: np.ndarray, matrices: np.ndarray, level: int) -> np.ndarray:
    y = y0.copy()
    for step in range(level):
        p = matrices[step//(level//4)]
        q = y/np.sqrt(1+1.2*y*y)
        middle = y+(q@p.T)/(2*level)
        y += (middle/np.sqrt(1+1.2*middle*middle)@p.T)/level
    return y


def sums(fields: np.ndarray, banks: np.ndarray) -> tuple:
    d = np.sum(np.square(fields[1]-fields[0]), axis=-1)
    n = np.sum(np.square(fields[2:]-fields[1]), axis=-1)
    return (np.array([d[banks == bank].sum() for bank in range(8)]),
            np.array([[value[banks == bank].sum() for bank in range(8)] for value in n]))


def f(d: np.ndarray, n: np.ndarray) -> np.ndarray:
    return 1-n/d if np.all(d > 0) else np.full(n.shape, np.nan)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    complete = read(root/'COMPLETE.json')
    assert complete['passed'] and complete['cases'] == 3
    assert complete['levels'] == [16, 32, 64, 128, 256]
    assert sha(root/'rows.json') == complete['rows_sha256']
    rows, max_error = [], 0.
    for row in read(root/'rows.json')['rows']:
        assert sha(Path(row['source_inputs'])) == row['source_inputs_sha256']
        with np.load(row['source_inputs']) as bundle:
            inputs = {key: bundle[key].copy() for key in bundle.files}
        banks, baseline = inputs['bank'], inputs['baseline']
        assert len(banks) == 80 and np.array_equal(np.bincount(banks), np.full(8, 10))
        levels = []
        for level in row['levels']:
            assert len(level['layers']) == complete['layers_per_case']
            den, num, old_den, old_num = np.zeros(8), np.zeros((3, 8)), np.zeros(8), np.zeros((3, 8))
            layers = []
            for item in level['layers']:
                file = root/item['file']
                assert sha(file) == item['sha256']
                for key in ['readout', 'energy']:
                    assert sha(Path(item['source_'+key])) == item['source_'+key+'_sha256']
                with np.load(file) as bundle:
                    local, fields, native_ref, stored_np, ray_ok = [bundle[key].copy() for key in
                        ['local_outputs', 'fields', 'native_refined', 'numpy_refined', 'ray_qualified']]
                with np.load(item['source_energy']) as bundle:
                    entrance, matrices, native = [bundle[key].copy() for key in ['entrance', 'weights', 'native_forward']]
                with np.load(item['source_readout']) as bundle:
                    old_local, old_fields = bundle['local_outputs'].copy(), bundle['fields'].copy()
                assert local.shape == (5, 80, 64) and fields.shape == (5, *baseline.shape)
                assert np.isfinite(local).all() and np.isfinite(fields).all()
                assert np.array_equal(fields[0], baseline) and np.array_equal(local[0], native)
                refined = solve(entrance, matrices, level['steps'])
                for saved in [native_ref, stored_np, local[1]]:
                    error = float(np.max(np.abs(refined-saved)))
                    max_error = max(max_error, error)
                    assert np.allclose(refined, saved, atol=1e-11, rtol=1e-11)
                gradient = local[1]/np.sqrt(1+1.2*local[1]**2)
                denominator = np.sum(gradient*gradient, axis=1)
                coeff = np.divide(np.sum((native-local[1])*gradient, axis=1), denominator,
                                  out=np.zeros(80), where=denominator != 0)
                normal = coeff[:, None]*gradient
                for saved, expected in [(local[2], native-normal), (local[3], local[1]+normal)]:
                    error = float(np.max(np.abs(saved-expected)))
                    max_error = max(max_error, error)
                    assert np.allclose(saved, expected, atol=1e-11, rtol=1e-11)
                assert ray_ok.all()
                with np.load(file) as bundle:
                    scales = bundle['ray_scale'].copy()
                assert np.all(scales >= 0)
                assert np.allclose(local[4], scales[:, None]*native, atol=1e-11, rtol=1e-11)
                assert np.all(np.abs(energy(local[4])-energy(local[1])) <= 1e-11*(1+np.abs(energy(local[1]))))
                local_ratio = np.linalg.norm(local[1:]-old_local[1:], axis=-1)/(1e-9+1e-3*np.linalg.norm(old_local[0]-old_local[1], axis=-1))
                output_ratio = np.linalg.norm(fields[1:]-old_fields[1:], axis=-1)/(1e-10+1e-3*np.linalg.norm(old_fields[0]-old_fields[1], axis=-1))
                d, n = sums(fields, banks)
                od, on = sums(old_fields, banks)
                difference = np.abs(f(d.sum(), n.sum(1))-f(od.sum(), on.sum(1)))
                resolved = bool(np.isfinite(difference).all())
                metrics = dict(local_max_limit_ratio=float(local_ratio.max()), output_max_limit_ratio=float(output_ratio.max()),
                    local_violations_by_mode=np.sum(local_ratio > 1, axis=1).tolist(),
                    output_violations_by_mode=np.sum(output_ratio > 1, axis=1).tolist(), fraction_resolved=resolved,
                    fraction_max_error=float(difference.max()) if resolved else None,
                    qualified=bool(local_ratio.max() <= 1 and output_ratio.max() <= 1 and resolved and difference.max() <= .0005))
                for key, value in metrics.items():
                    if isinstance(value, float):
                        assert np.isclose(value, item[key], atol=1e-9, rtol=1e-9), (key, value, item[key])
                    else:
                        assert value == item[key], (key, value, item[key])
                layers.append(dict(layer=item['layer'], **metrics))
                den += d
                num += n
                old_den += od
                old_num += on
            bank_f, old_bank_f = f(den[None], num), f(old_den[None], old_num)
            model_f, old_model_f = f(den.sum(), num.sum(1)), f(old_den.sum(), old_num.sum(1))
            resolved = bool(all(np.isfinite(v).all() for v in [bank_f, old_bank_f, model_f, old_model_f]))
            bank_error = float(np.max(np.abs(bank_f-old_bank_f))) if resolved else None
            model_error = float(np.max(np.abs(model_f-old_model_f))) if resolved else None
            qualified = bool(all(layer['qualified'] for layer in layers) and resolved and bank_error <= .0005 and model_error <= .0005)
            assert qualified == level['qualified'] and resolved == level['fractions_resolved']
            if resolved:
                assert np.isclose(bank_error, level['bank_fraction_max_error'], atol=1e-10)
                assert np.isclose(model_error, level['model_fraction_max_error'], atol=1e-10)
            levels.append(dict(steps=level['steps'], qualified=qualified, layers=layers,
                bank_fraction_max_error=bank_error, model_fraction_max_error=model_error,
                fractions=model_f.tolist() if resolved else None, seconds=level['seconds'],
                one_layer_matrix_work_multiplier=(2*level['steps']+40)/48))
        assert row['parameters_and_buffers_unchanged'] and row['native_roundtrip_exact']
        rows.append(dict(id=row['id'], dataset=row['dataset'], family=row['family'], levels=levels))
    qualified = [level for level in complete['levels'] if all(next(item for item in row['levels']
                  if item['steps'] == level)['qualified'] for row in rows)]
    assert qualified == complete['qualified_levels']
    result = dict(passed=True, witness=complete['witness'], cases=3, rows=rows,
        qualified_levels=qualified, max_reconstruction_error=max_error,
        input_complete_sha256=sha(root/'COMPLETE.json'), source_sha256=sha(Path(__file__)))
    dump(out/'summary.json', result)
    dump(out/'COMPLETE.json', dict(passed=True, witness=complete['witness'], cases=3,
        qualified_levels=qualified, summary_sha256=sha(out/'summary.json'), max_error=max_error))
    print('PASS independent native reference:', qualified, 'max reconstruction error', max_error, flush=True)


if __name__ == '__main__':
    main()
