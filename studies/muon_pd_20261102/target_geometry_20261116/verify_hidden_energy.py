"""Independent reconstruction of hidden maps, work, geometry and coverage."""
import argparse
from pathlib import Path
import numpy as np
from geometry import read, dump, sha


def energy(value: np.ndarray) -> np.ndarray:
    return np.sum(np.square(value)/(1+np.sqrt(1+1.2*np.square(value))), axis=-1)


def reconstruct(z: dict) -> tuple:
    h, end, weights = z['entrance'], z['endpoints'], z['weights']
    state, flux = [], []
    for steps in [4, 8, 16, 32]:
        y, work = h.copy(), np.zeros(len(h))
        for k in range(steps):
            p = weights[k//(steps//4)]
            s = (p+p.T)/2
            start_q = y*np.power(1+1.2*y*y, -.5)
            middle = y+(start_q@p.T)/(2*steps)
            q = middle*np.power(1+1.2*middle*middle, -.5)
            work += np.einsum('bi,ij,bj->b', q, s, q)/steps
            y += (q@p.T)/steps
        state.append(y)
        flux.append(work)
    endpoint, work = np.stack(state), np.stack(flux)
    maximum = max(float(np.max(np.abs(endpoint-end))), float(np.max(np.abs(work-z['work']))))
    assert np.allclose(endpoint, end, atol=1e-10, rtol=1e-11)
    assert np.allclose(work, z['work'], atol=1e-10, rtol=1e-11)
    assert np.allclose(end, z['native_endpoints'], atol=1e-11, rtol=1e-11)
    assert np.array_equal(z['native_forward'], z['native_endpoints'][0])
    loose, ref = z['references']
    wref = z['reference_work'][-1]
    e0, eref = energy(h), energy(ref)
    closure = eref-e0-wref
    valid = np.all(np.abs(loose-ref) <= 1e-9*(1+np.abs(ref)), axis=1)
    valid &= np.abs(z['reference_work'][0]-wref) <= 1e-8*(1+np.abs(wref))
    valid &= np.abs(closure) <= 1e-8*(1+np.abs(e0)+np.abs(eref)+np.abs(wref))
    difference = end-ref[None]
    error2 = np.einsum('knd,knd->kn', difference, difference)
    gradient = ref/np.sqrt(1+1.2*ref**2)
    norm2 = np.einsum('nd,nd->n', gradient, gradient)
    normal_work = np.einsum('knd,nd->kn', difference, gradient)
    normal2 = np.zeros_like(error2)
    nonzero = norm2 != 0
    normal2[:, nonzero] = np.square(normal_work[:, nonzero])/norm2[None, nonzero]
    energy_gap = energy(end)-eref[None]
    remainder = energy_gap-normal_work
    rounding = 64*np.finfo(float).eps*(1+np.abs(energy(end))+np.abs(eref)[None])
    assert np.all(np.abs(remainder) <= .5*error2+rounding)
    feature_floor = np.sqrt(np.sum((loose-ref)**2, axis=1))+64*np.finfo(float).eps*(1+np.sqrt(np.sum(ref**2, axis=1)))
    energy_floor = np.abs(energy(loose)-eref)+64*np.finfo(float).eps*(1+np.abs(energy(loose))+np.abs(eref))
    stats = []
    for i, steps in enumerate([4, 8, 16, 32]):
        stats.append(dict(steps=steps, feature_error_rms=float(np.sqrt(error2[i].mean())),
            feature_error_max_norm=float(np.sqrt(error2[i].max())),
            normal_squared_error_share=float(normal2[i].sum()/error2[i].sum()) if error2[i].sum() else None,
            energy_gap_rms=float(np.sqrt(np.mean(energy_gap[i]**2))),
            balance_defect_rms=float(np.sqrt(np.mean((energy(end[i])-e0-z['work'][i])**2))),
            work_change_rms=float(np.sqrt(np.mean((z['work'][i]-wref)**2))),
            resolved_feature_points=int(np.sum(np.sqrt(error2[i]) > feature_floor)),
            energy_at_floor_points=int(np.sum(np.abs(energy_gap[i]) <= energy_floor)),
            directional_only_points=int(np.sum((np.sqrt(error2[i]) > feature_floor)&(np.abs(energy_gap[i]) <= energy_floor)))))
    details = dict(points=len(h), qualified_points=int(valid.sum()), zero_reference_gradient_points=int((~nonzero).sum()),
                   reference_closure_max=float(np.max(np.abs(closure))), levels=stats)
    arrays = dict(feature_error_squared=error2, normal_error_squared=normal2, energy_gap=energy_gap,
        balance_defect=energy(end)-e0[None]-z['work'], work_change=z['work']-wref[None],
        qualified=valid, feature_floor=feature_floor, energy_floor=energy_floor)
    return details, arrays, maximum


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
    count, times = (1, [.5]) if complete['witness'] else (2, [.1, .3, .5, .7, .9])
    n = 8*count*len(times)
    assert complete['points_per_layer'] == n
    results, max_error = [], 0.
    for row in data['rows']:
        assert sha(Path(row['assets_file'])) == row['assets_sha256']
        file = root/row['inputs_file']
        assert sha(file) == row['inputs_sha256']
        with np.load(row['assets_file']) as z:
            source = np.concatenate([z[f'source_{b}'][:count] for b in range(8)]).astype(float)
            target = np.concatenate([z[f'target_{b}'][:count] for b in range(8)]).astype(float)
        with np.load(file) as z:
            assert np.array_equal(source, z['source']) and np.array_equal(target, z['target'])
            assert np.array_equal(z['x'], np.concatenate([(1-t)*source+t*target for t in times]))
            assert np.array_equal(z['time'][:, 0], np.repeat(times, len(source)))
            assert np.array_equal(z['bank'], np.tile(np.repeat(np.arange(8), count), len(times)))
            assert np.array_equal(z['pair'], np.tile(np.arange(count), 8*len(times)))
        layers, full = [], []
        assert len(row['layers']) == 6
        for layer in row['layers']:
            file = root/layer['file']
            assert sha(file) == layer['sha256'] and layer['points'] == n
            with np.load(file) as archive:
                z = {k: archive[k].copy() for k in archive.files}
            assert all(np.isfinite(v).all() for v in z.values())
            assert z['entrance'].shape == (n, 64) and z['weights'].shape == (4, 64, 64)
            assert z['endpoints'].shape == (4, n, 64) and z['references'].shape == (2, n, 64)
            assert np.array_equal(z['steps'], [4, 8, 16, 32])
            assert np.array_equal(z['tolerances'], [[1e-10, 1e-11], [1e-12, 1e-13]])
            detail, derived, maximum = reconstruct(z)
            max_error = max(max_error, maximum)
            original = layer['summary']
            for key in ['points', 'qualified_points', 'zero_reference_gradient_points']:
                assert detail[key] == original[key]
            assert abs(detail['reference_closure_max']-original['reference_closure_max']) < 1e-9
            for measured, recorded in zip(detail['levels'], original['levels']):
                for key in measured:
                    if measured[key] is None:
                        assert recorded[key] is None
                    else:
                        difference = abs(measured[key]-recorded[key])
                        assert difference < 1e-9, (row['id'], layer['layer'], key, difference)
                        max_error = max(max_error, difference)
            full.append(derived)
            layers.append(dict(layer=layer['layer'], **detail, input_sha256=sha(file)))
        combined = {key: np.concatenate([r[key] for r in full], axis=1)
                    for key in ['feature_error_squared', 'normal_error_squared', 'energy_gap', 'balance_defect', 'work_change']}
        aggregate = []
        for i, steps in enumerate([4, 8, 16, 32]):
            e2, normal2 = combined['feature_error_squared'][i], combined['normal_error_squared'][i]
            aggregate.append(dict(steps=steps, feature_error_rms=float(np.sqrt(e2.mean())),
                normal_squared_error_share=float(normal2.sum()/e2.sum()) if e2.sum() else None,
                energy_gap_rms=float(np.sqrt(np.mean(combined['energy_gap'][i]**2))),
                balance_defect_rms=float(np.sqrt(np.mean(combined['balance_defect'][i]**2))),
                work_change_rms=float(np.sqrt(np.mean(combined['work_change'][i]**2)))))
        results.append(dict(id=row['id'], dataset=row['dataset'], family=row['family'], layers=layers, aggregate=aggregate,
            points_per_layer=n, qualified_points=sum(v['qualified_points'] for v in layers),
            total_points=6*n, all_qualified=all(v['qualified_points']==n for v in layers)))
    result = dict(passed=True, witness=complete['witness'], cases=3, rows=results,
        all_qualified=all(r['all_qualified'] for r in results), max_reconstruction_error=max_error,
        all_source_arrays_exact=True, input_complete_sha256=sha(root/'COMPLETE.json'), source_sha256=sha(Path(__file__)))
    dump(out/'summary.json', result)
    dump(out/'COMPLETE.json', dict(passed=True, witness=complete['witness'], cases=3,
        summary_sha256=sha(out/'summary.json'), max_error=max_error))
    print('PASS independent hidden energy:', len(results), 'cases, max error', max_error, flush=True)


if __name__ == '__main__':
    main()
