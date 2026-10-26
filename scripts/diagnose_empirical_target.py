"""Compare the exact cached-target-bank teacher with the analytic population.

No training is performed. All target atoms are included in each posterior sum.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def empirical_teacher(atoms: np.ndarray, x: np.ndarray, time: float,
                      second_features: np.ndarray | None = None) -> dict[str, np.ndarray]:
    if not 0 < time < 1:
        raise ValueError('posterior identity requires an interior time')
    if atoms.ndim != 2 or x.ndim != 2 or atoms.shape[1] != x.shape[1] or len(atoms) == 0:
        raise ValueError('incompatible atoms and query shapes')
    a = 1-time
    atoms = np.asarray(atoms, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    dimension = atoms.shape[1]
    if second_features is None:
        second_features = np.einsum('ni,nj->nij', atoms, atoms).reshape(len(atoms), -1)
    squared_norm = (atoms**2).sum(axis=1)
    result = {k: [] for k in ['velocity', 'jacobian', 'noise_covariance', 'ess', 'max_weight']}
    for start in range(0, len(x), 16):
        points = x[start:start+16]
        logits = time/a**2*(points@atoms.T) - time**2/(2*a**2)*squared_norm
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= weights.sum(axis=1, keepdims=True)
        mean = weights@atoms
        second = (weights@second_features).reshape(len(points), dimension, dimension)
        covariance = second-mean[:, :, None]*mean[:, None, :]
        result['velocity'].append((mean-points)/a)
        result['jacobian'].append(-np.eye(dimension)[None]/a+time/a**3*covariance)
        result['noise_covariance'].append(covariance/a**2)
        result['ess'].append(1/(weights**2).sum(axis=1))
        result['max_weight'].append(weights.max(axis=1))
    return {k: np.concatenate(v) for k, v in result.items()}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnose(study: Path, data_root: Path, output: Path, points: int = 128) -> dict:
    if points < 1:
        raise ValueError('points must be positive')
    output.mkdir(parents=True, exist_ok=False)
    dataset = 'rotated_product_bimodal_16d'
    manifest = json.loads((data_root/'manifest.json').read_text())
    provenance = json.loads((study/'provenance.json').read_text())
    assert sha256(data_root/'manifest.json') == provenance['dataset_manifest_sha256']
    path = data_root/dataset/'train.npz'
    assert sha256(path) == manifest['files'][dataset+'/train.npz']['sha256']
    with np.load(path, allow_pickle=False) as data:
        atoms = data['x1'].astype(np.float64)
    features = np.einsum('ni,nj->nij', atoms, atoms).reshape(len(atoms), -1)
    summary = json.loads((study/'diagnostics'/dataset/'scalar_k1/summary.json').read_text())
    rows, inputs = [], {}
    for time in (.1, .3, .5, .7, .9):
        path = study/'diagnostics'/dataset/'scalar_k1'/f'common-{time}.npz'
        assert sha256(path) == summary['arrays_sha256'][path.name]
        inputs[str(path)] = sha256(path)
        with np.load(path, allow_pickle=False) as bank:
            x = bank['x'][:points]
            assert len(x) == points
            population_velocity = bank['oracle'][:points]
            population_jacobian = bank['oracle_jacobian'][:points]
        empirical = empirical_teacher(atoms, x, time, features)
        population_noise = (np.eye(atoms.shape[1])[None]+(1-time)*population_jacobian)/time
        empirical_noise = empirical['noise_covariance']
        eigen_min = float(np.linalg.eigvalsh(empirical_noise).min())
        assert eigen_min > -1e-9, eigen_min
        np.testing.assert_allclose(empirical_noise,
            (np.eye(atoms.shape[1])[None]+(1-time)*empirical['jacobian'])/time, atol=1e-10, rtol=1e-10)
        rows.append(dict(time=time,points=points,target_atoms=len(atoms),
            population_to_empirical_field_mse=float(np.mean((population_velocity-empirical['velocity'])**2)),
            population_to_empirical_jacobian_error=float(np.mean(np.sum((population_jacobian-empirical['jacobian'])**2,axis=(1,2)))),
            empirical_posterior_ess_quantiles=np.quantile(empirical['ess'],[0,.25,.5,.75,1]).tolist(),
            median_largest_posterior_weight=float(np.median(empirical['max_weight'])),
            population_noise_trace_mean=float(np.trace(population_noise,axis1=-2,axis2=-1).mean()),
            empirical_noise_trace_mean=float(np.trace(empirical_noise,axis1=-2,axis2=-1).mean()),
            empirical_noise_eigen_min=eigen_min))
        np.savez_compressed(output/f'common-{time}.npz',x=x,population_velocity=population_velocity,
            population_jacobian=population_jacobian,population_noise_covariance=population_noise,**empirical)
    result=dict(state='COMPLETE',rows=rows,source_sha256=sha256(Path(__file__)),
        manifest_sha256=sha256(data_root/'manifest.json'),target_sha256=sha256(data_root/dataset/'train.npz'),
        input_arrays_sha256=inputs,files_sha256={p.name:sha256(p) for p in sorted(output.glob('*.npz'))},
        interpretation='Exact uniform empirical-target posterior versus known population on exposed validation anchors. Descriptive comparison; neither neural capacity nor optimization causation is identified.')
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('study','data-root','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=diagnose(args.study,args.data_root,args.output)
    print(json.dumps(dict(state=result['state'],rows=result['rows']),indent=2))
