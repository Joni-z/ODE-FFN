"""Read-only, preregistered PSD-denoiser diagnostic of saved Stage17 Jacobians."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DATASETS = ('gmm_ring8_2d', 'rotated_product_bimodal_16d')
TITLES = ('Ring8', 'Product16')
ARMS = ('vector_parent', 'swiglu_budget', 'swiglu')
LABELS = ('Vector ODE parent', 'SwiGLU96', 'SwiGLU128')
TIMES = (.1, .3, .5, .7, .9)
COMPONENTS = ('jacobian_error', 'skew_distance', 'negative_distance',
              'structural_distance', 'projected_error', 'projection_slack')
SCOPE = ('Saved-point Jacobian geometry only. Independent PSD projections need not '
         'be integrable, globally realizable, or trainable. No new velocity model, '
         'architecture lower bound, sampling improvement, or training experiment.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def geometry(jacobian, oracle, time):
    a = 1-time
    identity = np.eye(jacobian.shape[-1])
    actual, target = identity+a*jacobian, identity+a*oracle
    symmetric = .5*(actual+actual.swapaxes(-1, -2))
    skew = .5*(actual-actual.swapaxes(-1, -2))
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    projected = np.einsum('nik,nk,njk->nij', eigenvectors, np.maximum(eigenvalues, 0), eigenvectors)
    norm2 = lambda value: np.square(value).sum(axis=(-1, -2))
    target_skew = .5*(target-target.swapaxes(-1, -2))
    target_eigenvalues = np.linalg.eigvalsh(.5*(target+target.swapaxes(-1, -2)))
    tolerance = 1e-10*np.maximum(1, np.abs(eigenvalues).max(axis=1))
    target_tolerance = 1e-10*np.maximum(1, np.abs(target_eigenvalues).max(axis=1))
    assert np.all(np.sqrt(norm2(target_skew)) <= target_tolerance)
    assert np.all(target_eigenvalues[:, 0] >= -target_tolerance)

    denoiser = dict(jacobian_error=norm2(actual-target), skew_distance=norm2(skew),
                    negative_distance=np.square(np.minimum(eigenvalues, 0)).sum(axis=1),
                    projected_error=norm2(projected-target))
    denoiser['structural_distance'] = denoiser['skew_distance']+denoiser['negative_distance']
    denoiser['projection_slack'] = (denoiser['jacobian_error']-denoiser['structural_distance']
                                   -denoiser['projected_error'])
    scale = np.maximum(1, np.maximum(denoiser['jacobian_error'],
                                     denoiser['structural_distance']+denoiser['projected_error']))
    inequality_tolerance = 1e-10*scale
    assert np.all(denoiser['projection_slack'] >= -inequality_tolerance)
    # This independently checks the PSD-cone cross term. Oracle symmetry is
    # verified above; any residual asymmetry is bounded numerical roundoff.
    cross_term = -2*np.sum((symmetric-projected)*target, axis=(-1, -2))
    assert np.all(np.abs(denoiser['projection_slack']-cross_term) <= inequality_tolerance)
    np.testing.assert_allclose(norm2(actual-projected), denoiser['structural_distance'], rtol=1e-11, atol=1e-12)
    velocity = {name: value/(a*a) for name, value in denoiser.items()}
    np.testing.assert_allclose(velocity['jacobian_error'], norm2(jacobian-oracle), rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(velocity['projected_error'], norm2((projected-identity)/a-oracle), rtol=1e-11, atol=1e-12)
    return dict(**{f'denoiser_{k}': v for k, v in denoiser.items()},
                **{f'velocity_{k}': v for k, v in velocity.items()},
                symmetric_min_eigenvalue=eigenvalues[:, 0],
                psd_violation=eigenvalues[:, 0] < -tolerance,
                negative_eigenvalue_count=(eigenvalues < -tolerance[:, None]).sum(axis=1),
                violation_tolerance=tolerance,
                oracle_min_eigenvalue=target_eigenvalues[:, 0],
                oracle_skew_frobenius=np.sqrt(norm2(target_skew)),
                oracle_tolerance=target_tolerance,
                denoiser_inequality_tolerance=inequality_tolerance)


def aggregate(records, dataset, arm, time=None):
    row = dict(dataset=dataset, arm=arm)
    if time is not None:
        row['time'] = time
    row['anchor_records'] = len(records)
    for unit in ('velocity', 'denoiser'):
        for component in COMPONENTS:
            key = unit+'_'+component
            row['mean_'+key] = float(np.mean([r[key] for r in records]))
        total = row['mean_'+unit+'_jacobian_error']
        assert total > 0
        row[unit+'_structural_distance_fraction'] = row['mean_'+unit+'_structural_distance']/total
        row[unit+'_skew_distance_fraction'] = row['mean_'+unit+'_skew_distance']/total
        row[unit+'_negative_distance_fraction'] = row['mean_'+unit+'_negative_distance']/total
        row[unit+'_pointwise_projected_error_fraction'] = row['mean_'+unit+'_projected_error']/total
    eigen_minima = [r['symmetric_min_eigenvalue'] for r in records]
    row.update(psd_violation_fraction=float(np.mean([r['psd_violation'] for r in records])),
               mean_negative_eigenvalue_count=float(np.mean([r['negative_eigenvalue_count'] for r in records])),
               min_symmetric_eigenvalue=min(eigen_minima), median_symmetric_min_eigenvalue=float(np.median(eigen_minima)),
               p05_symmetric_min_eigenvalue=float(np.quantile(eigen_minima, .05)),
               min_oracle_eigenvalue=min(r['oracle_min_eigenvalue'] for r in records),
               max_oracle_skew_frobenius=max(r['oracle_skew_frobenius'] for r in records),
               min_velocity_projection_slack=min(r['velocity_projection_slack'] for r in records),
               min_denoiser_projection_slack=min(r['denoiser_projection_slack'] for r in records),
               max_denoiser_inequality_tolerance=max(r['denoiser_inequality_tolerance'] for r in records))
    return row


def figures(output, aggregates, panels):
    directory = output/'figures'
    directory.mkdir(exist_ok=True)
    def save(fig, name):
        for suffix in ('png', 'pdf'):
            fig.savefig(directory/f'{name}.{suffix}', dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    parts = ('skew_distance', 'negative_distance', 'projected_error', 'projection_slack')
    names = ('Skew distance', 'Negative-spectrum distance', 'Projected-target residual', 'Cone cross term')
    colors = ('#de9934', '#bb4444', '#5d8daf', '#b4b4b4')
    for ax, dataset, title in zip(axes, DATASETS, TITLES):
        rows = [next(r for r in aggregates if r['dataset'] == dataset and r['arm'] == arm) for arm in ARMS]
        bottom = np.zeros(3)
        for part, name, color in zip(parts, names, colors):
            # Any negative cone slack below numerical tolerance is only clipped
            # for rendering; all CSVs retain its original value.
            values = np.maximum(0, [r['mean_velocity_'+part] for r in rows])
            ax.bar(range(3), values, bottom=bottom, label=name, color=color)
            bottom += values
        for i, row in enumerate(rows):
            ax.text(i, bottom[i], f"{100*row['velocity_structural_distance_fraction']:.1f}% structural", ha='center', va='bottom', fontsize=8)
        ax.set_xticks(range(3), LABELS, rotation=15, ha='right')
        ax.set(title=title, ylabel='Mean squared velocity-Jacobian Frobenius error')
        ax.set_ylim(0, 1.18*bottom.max())
        ax.grid(axis='y', alpha=.15)
    axes[-1].legend(fontsize=8, frameon=False)
    fig.suptitle('All five times, equal anchor weights; denoiser distances divided by (1-t)^2 before averaging', fontsize=10)
    save(fig, 'complete-field-error-components')
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    for col, (dataset, title) in enumerate(zip(DATASETS, TITLES)):
        for arm, label, color in zip(ARMS, LABELS, ('#bb4444', '#7ba8bd', '#255f88')):
            rows = [next(r for r in panels if r['dataset'] == dataset and r['arm'] == arm and r['time'] == t) for t in TIMES]
            axes[0, col].plot(TIMES, [100*r['velocity_structural_distance_fraction'] for r in rows], marker='o', label=label, color=color)
            axes[1, col].plot(TIMES, [100*r['psd_violation_fraction'] for r in rows], marker='o', label=label, color=color)
        axes[0, col].set(title=title, ylabel='Structural distance / Jacobian error (%)')
        axes[1, col].set(ylabel='Anchors with a negative symmetric eigenvalue (%)', xlabel='Interpolation time', ylim=(-2, 102))
        for row in range(2):
            axes[row, col].set_xticks(TIMES)
            axes[row, col].grid(alpha=.15)
    axes[0, -1].legend(fontsize=8, frameon=False)
    fig.suptitle('Saved-point geometry; projected Jacobians do not define a trained or globally integrable model', fontsize=10)
    save(fig, 'complete-field-time-profiles')


def diagnose(root, output, protocol):
    root, output, protocol = root.resolve(), output.resolve(), protocol.resolve()
    assert output != root and root not in output.parents and output not in root.parents
    started = datetime.now(timezone.utc).isoformat()
    hashes, array_hashes, array_dtypes = {}, {}, {}
    def read(path):
        hashes[str(path)] = sha(path)
        return json.loads(path.read_text())
    manifest = read(root/'source-manifest.json')
    for name, digest in manifest['all_files'].items():
        assert sha(root/'source'/name) == digest, name
    protocol_hash, script_hash = sha(protocol), sha(__file__)
    all_records, panels, aggregates, replay = [], [], [], []
    target_checks = {}
    for dataset, suffix in zip(DATASETS, ('v1', 'product16')):
        study = root/'output'/f'shared_features_{suffix}_20261024_m1'
        assert read(study/'status.json')['state'] == 'COMPLETE'
        provenance = read(study/'provenance.json')
        data = Path('/Users/mr.z/ode-workspace-prep/data/synthetic')/('v1' if suffix == 'v1' else 'product16_v1')
        assert sha(data/'manifest.json') == provenance['dataset_manifest_sha256']
        hashes[str(data/'manifest.json')] = sha(data/'manifest.json')
        common = {}
        for arm in ARMS:
            directory = study/'diagnostics'/dataset/arm
            native = read(directory/'summary.json')
            cfg = read(study/'configs'/f'{arm}.json')
            assert native['dataset'] == dataset and native['split'] == 'validation'
            assert native['seed'] == 0 and native['step'] == 5000
            assert sha(study/'configs'/f'{arm}.json') == native['config_sha256']
            assert native['manifest_sha256'] == provenance['dataset_manifest_sha256']
            assert sha(data/dataset/'validation.npz') == native['targets_sha256']
            hashes[str(data/dataset/'validation.npz')] = native['targets_sha256']
            checkpoint = study/'runs'/cfg['experiment_id']/dataset/native['model']/'seed000/checkpoints/step_0005000.pt'
            assert sha(checkpoint) == native['checkpoint_sha256']
            hashes[str(checkpoint)] = native['checkpoint_sha256']
            assert set(native['arrays_sha256']) == {f'common-{t:.1f}.npz' for t in TIMES}
            arm_records, field_errors = [], []
            for time in TIMES:
                path = directory/f'common-{time:.1f}.npz'
                digest = sha(path)
                assert digest == native['arrays_sha256'][path.name]
                array_hashes[str(path)] = digest
                with np.load(path, allow_pickle=False) as archive:
                    array_dtypes[str(path)] = {key: str(archive[key].dtype) for key in ('x', 'prediction', 'jacobian', 'oracle', 'oracle_jacobian')}
                    assert set(array_dtypes[str(path)].values()) == {'float64'}, 'Review cone tolerances before using lower-precision arrays'
                    values = {key: archive[key].astype(np.float64) for key in ('x', 'prediction', 'jacobian', 'oracle', 'oracle_jacobian')}
                assert all(np.isfinite(value).all() for value in values.values())
                dimension = 2 if suffix == 'v1' else 16
                assert values['x'].shape == values['prediction'].shape == values['oracle'].shape == (512, dimension)
                assert values['jacobian'].shape == values['oracle_jacobian'].shape == (512, dimension, dimension)
                shared = {k: values[k] for k in ('x', 'oracle', 'oracle_jacobian')}
                if time in common:
                    for key in shared:
                        np.testing.assert_array_equal(shared[key], common[time][key])
                else:
                    common[time] = shared
                result = geometry(values['jacobian'], values['oracle_jacobian'], time)
                records = [dict(dataset=dataset, arm=arm, time=time, anchor=i,
                                **{key: value[i].item() for key, value in result.items()}) for i in range(512)]
                panels.append(aggregate(records, dataset, arm, time))
                arm_records.extend(records)
                field_errors.append(float(np.mean(np.square(values['prediction']-values['oracle']))))
            summary = aggregate(arm_records, dataset, arm)
            np.testing.assert_allclose(summary['mean_velocity_jacobian_error'], native['jacobian_error'], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(np.mean(field_errors), native['field_mse'], rtol=1e-12, atol=1e-12)
            aggregates.append(summary)
            all_records.extend(arm_records)
            replay.append(dict(dataset=dataset, arm=arm, native_jacobian_error=native['jacobian_error'],
                               recomputed_jacobian_error=summary['mean_velocity_jacobian_error'],
                               absolute_difference=abs(summary['mean_velocity_jacobian_error']-native['jacobian_error']),
                               native_field_mse=native['field_mse'], recomputed_field_mse=float(np.mean(field_errors))))
        target_checks[dataset] = dict(common_times=5, anchors_per_time=512, input_and_oracle_equality='EXACT')
    assert len(array_hashes) == len(panels) == 30 and len(all_records) == 15360 and len(aggregates) == 6
    output.mkdir(parents=True, exist_ok=True)
    archived = output/'source'
    archived.mkdir(exist_ok=True)
    for path in (Path(__file__).resolve(), protocol):
        if path != archived/path.name:
            shutil.copy2(path, archived/path.name)
    write_csv(output/'per-anchor.csv', all_records)
    write_csv(output/'per-panel.csv', panels)
    write_csv(output/'aggregate.csv', aggregates)
    figures(output, aggregates, panels)
    for path, digest in {**hashes, **array_hashes}.items():
        assert sha(path) == digest, 'Input changed during analysis: '+path
    for name, digest in manifest['all_files'].items():
        assert sha(root/'source'/name) == digest, name
    assert sha(protocol) == protocol_hash and sha(__file__) == script_hash
    summary = dict(state='PASS', arrays=30, model_anchor_records=15360, shared_dataset_time_anchors=5120,
                   primary_units='Velocity-Jacobian squared Frobenius distances: divide by (1-t)^2 before aggregation.',
                   secondary_units='Separately named unweighted denoiser-Jacobian distances.',
                   oracle_and_common_anchor_checks=target_checks, native_metric_reproduction=replay,
                   projection_inequalities='PASS at every saved anchor within recorded scale-aware tolerance',
                   aggregate=aggregates, scope=SCOPE)
    write(output/'summary.json', summary)
    write(output/'provenance.json', dict(state='PASS', started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(),
          script_sha256=script_hash, protocol_sha256=protocol_hash,
          protocol_mtime_utc=datetime.fromtimestamp(protocol.stat().st_mtime, timezone.utc).isoformat(),
          source_files_verified=len(manifest['all_files']), arrays_sha256=array_hashes, array_dtypes=array_dtypes,
          metadata_and_checkpoint_sha256=hashes,
          input_bytes_unchanged=True, scope=SCOPE))
    return dict(state='PASS', arrays=30, rows=15360, aggregate=aggregates, scope=SCOPE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.root, args.output, args.protocol), indent=2))
