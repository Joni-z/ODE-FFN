"""Audit the completed generic GLU control and its fixed posterior-region readout."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

DATASETS = ('gaussian_aniso_2d', 'gmm_ring8_2d', 'rotated_product_bimodal_16d')
TITLES = ('Gaussian2D', 'Ring8', 'Product16')
ARMS = ('swiglu_budget', 'swiglu', 'sigmoid_glu')
LABELS = ('SwiGLU96', 'SwiGLU128', 'Sigmoid GLU128')
COLORS = ('#8bb0c5', '#255f88', '#b94b48')
STEPS, TIMES = tuple(range(0, 5001, 1000)), (.1, .3, .5, .7, .9)
METRICS = ('validation_oracle_mse', 'jacobian_error', 'validation_sw2')
STATE_KEYS = ('model', 'optimizer', 'runtime_buffers', 'data_rng', 'torch_rng', 'numpy_rng', 'python_rng')
EDGES, REGIONS = np.array([0., .1, .5, .9, 1.]), ('[0,.1)', '[.1,.5)', '[.5,.9)', '[.9,1]')
REPLAY_SHA = '142bbe47e3ddf7f25befb1f0eda3b67d187122505805fcf94d4a5c62477640b7'
REGION_SOURCE_SHA = 'da081f91278eb8f68e3b1bef446615437fec942162ef33485c190d94125c2453'
REGION_PROTOCOL_SHA = '399c89307b462ad91698c0b3621826295af66b4ac827e207025171e01d450251'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def csv_write(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):equal(x, y)
    else:
        assert a == b


def registered_screen(rows):
    lookup = {(r['dataset'], r['label']): r for r in rows}
    details, leads = {}, []
    for dataset in DATASETS:
        best = {m: min(lookup[dataset, arm][m] for arm in ARMS[:2]) for m in METRICS}
        assert all(value > 0 for value in best.values())
        ratios = {m: lookup[dataset, 'sigmoid_glu'][m]/best[m] for m in METRICS}
        if dataset in DATASETS[1:] and ratios['validation_oracle_mse'] <= .9:
            leads.append(dataset)
        details[dataset] = dict(ratio_to_better_swiglu=ratios)
    primary = all(details[d]['ratio_to_better_swiglu']['validation_oracle_mse'] <= 1.05 for d in DATASETS)
    secondary = all(details[d]['ratio_to_better_swiglu'][m] <= 1.1 for d in DATASETS[1:] for m in METRICS[1:])
    passes = bool(leads) and primary and secondary
    return dict(state='GENERIC_BASELINE_LEAD' if passes else 'STOP_GENERIC_SETTING',
                generic_baseline_lead=passes, architecture_goal_complete=False,
                qualifying_nonlinear_leads=leads, all_primary_within_5pct=primary,
                nonlinear_Jacobian_SW2_within_10pct=secondary, details=details,
                scope='Known GLU baseline only; no ODE nomination or mechanism claim. Engineering screen on exposed validation. A lead still requires independent evaluation; failure stops this substitution without rescue.')


def posterior_regions(study, data, output, read, hashes):
    """Use the unchanged prior diagnostic's formula, bins and coordinate units."""
    spec = read(data/'manifest.json')['datasets'][DATASETS[2]]['distribution']
    qmat, mu, variance = (np.asarray(spec[k], dtype=np.float64) for k in ('rotation', 'locations', 'variances'))
    np.testing.assert_allclose(qmat.T@qmat, np.eye(16), atol=1e-12, rtol=0)
    errors, bins, entropy, probabilities, summaries, common = {}, {}, {}, {}, {}, {}
    max_field, max_jacobian, max_rotation = 0., 0., 0.
    for arm in ARMS:
        directory = study/'diagnostics'/DATASETS[2]/arm
        summary = read(directory/'summary.json'); summaries[arm] = summary
        for time in TIMES:
            path = directory/f'common-{time:.1f}.npz'
            hashes[str(path)] = sha(path)
            assert hashes[str(path)] == summary['arrays_sha256'][path.name]
            with np.load(path, allow_pickle=False) as archive:
                values = {k: archive[k] for k in ('x', 'prediction', 'oracle', 'oracle_jacobian')}
            assert all(v.dtype == np.float64 and np.isfinite(v).all() for v in values.values())
            x, prediction, oracle = (values[k] for k in ('x', 'prediction', 'oracle'))
            assert x.shape == prediction.shape == oracle.shape == (512, 16)
            assert values['oracle_jacobian'].shape == (512, 16, 16)
            if time in common:
                for key in ('x', 'oracle', 'oracle_jacobian'):
                    np.testing.assert_array_equal(values[key], common[time][key])
            else:common[time] = values
            latent, a = x@qmat, 1-time
            c = a*a+time*time*variance
            q = time*mu*latent/c
            sign = np.tanh(q)
            expected = (((time*variance-a)/c)*latent+a*mu/c*sign)@qmat.T
            diagonal = (time*variance-a)/c+time*a*mu*mu/(c*c)*(1-sign*sign)
            jacobian = np.einsum('ij,bj,kj->bik', qmat, diagonal, qmat)
            max_field = max(max_field, float(np.abs(expected-oracle).max()))
            max_jacobian = max(max_jacobian, float(np.abs(jacobian-values['oracle_jacobian']).max()))
            np.testing.assert_allclose(expected, oracle, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(jacobian, values['oracle_jacobian'], atol=1e-10, rtol=1e-11)
            probability = (1+sign)/2
            r = 2*np.abs(q); e = np.exp(-r)
            h = (np.log1p(e)+r*e/(1+e))/np.log(2.)
            assert np.all((probability >= 0)&(probability <= 1)) and np.all((h >= 0)&(h <= 1))
            region = np.searchsorted(EDGES[1:-1], h, side='right')
            if time in bins:np.testing.assert_array_equal(region, bins[time])
            bins[time], entropy[time], probabilities[time] = region, h, probability
            errors[arm, time] = ((prediction-oracle)@qmat)**2
            ambient = float(np.mean((prediction-oracle)**2)); local = float(errors[arm, time].mean())
            max_rotation = max(max_rotation, abs(local-ambient))
            np.testing.assert_allclose(local, ambient, atol=1e-12, rtol=1e-12)
            recorded = next(row['field_mse'] for row in summary['rows'] if row['time'] == time)
            np.testing.assert_allclose(local, recorded, atol=1e-12, rtol=1e-12)
        errors[arm, 'all_times'] = np.concatenate([errors[arm, t] for t in TIMES])
        np.testing.assert_allclose(errors[arm, 'all_times'].mean(), summary['field_mse'], atol=1e-12, rtol=1e-12)
    bins['all_times'] = np.concatenate([bins[t] for t in TIMES])
    rows = []
    for time in (*TIMES, 'all_times'):
        for arm in ARMS:
            err = errors[arm, time]
            for index, name in [(None, 'all'), *enumerate(REGIONS)]:
                mask = np.ones_like(bins[time], dtype=bool) if index is None else bins[time] == index
                count = int(mask.sum()); contribution = float(err[mask].sum()/err.size)
                row = dict(label=arm, time=time, entropy_bin=name, coordinate_count=count, coordinate_fraction=count/err.size,
                           conditional_mse=float(err[mask].mean()) if count else None,
                           mse_contribution=contribution, error_share=contribution/float(err.mean()), total_mse=float(err.mean()))
                for baseline in ARMS[:2]:
                    gap = float((err-errors[baseline, time])[mask].sum()/err.size)
                    total_gap = float(err.mean()-errors[baseline, time].mean())
                    row['gap_vs_'+baseline] = gap
                    row['gap_share_vs_'+baseline] = gap/total_gap if arm == 'sigmoid_glu' and total_gap != 0 else None
                rows.append(row)
            assert sum(row['coordinate_count'] for row in rows[-4:]) == err.size
            np.testing.assert_allclose(sum(row['mse_contribution'] for row in rows[-4:]), err.mean(), atol=1e-12, rtol=1e-12)
    assert len(rows) == 90
    csv_write(output/'posterior-regions.csv', rows)
    np.savez_compressed(output/'posterior-coordinate-errors.npz', arms=np.array(ARMS), times=np.array(TIMES),
                        squared_velocity_error=np.stack([np.stack([errors[arm, t] for t in TIMES]) for arm in ARMS]),
                        entropy=np.stack([entropy[t] for t in TIMES]), positive_probability=np.stack([probabilities[t] for t in TIMES]),
                        region=np.stack([bins[t] for t in TIMES]))
    receipt = dict(state='PASS', arrays=15, rows=len(rows), error_shape=[3, 5, 512, 16], entropy_edges=EDGES.tolist(),
                   maximum_oracle_reconstruction_error=max_field, maximum_oracle_Jacobian_reconstruction_error=max_jacobian,
                   maximum_rotation_mse_difference=max_rotation, native_field_mse={a: summaries[a]['field_mse'] for a in ARMS},
                   scope='All fixed entropy bins/times, descriptive velocity-coordinate error only. These networks do not predict posterior probabilities; no gate-classification or causal assignment-error interpretation.')
    write(output/'posterior-regions-audit.json', receipt)
    return rows, receipt


def make_figures(output, rows, context, curves, regions):
    directory = output/'figures'; directory.mkdir(exist_ok=True)
    lookup = {(r['dataset'], r['label']): r for r in rows}
    def save(fig, name):
        for extension in ('png', 'pdf'):fig.savefig(directory/f'{name}.{extension}', dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), layout='constrained')
    for col, (dataset, title) in enumerate(zip(DATASETS, TITLES)):
        historical = next(r for r in context if r['dataset'] == dataset)
        for row, (metric, name) in enumerate(zip(METRICS, ('Oracle MSE', 'Jacobian error', 'Validation SW2'))):
            best = min(lookup[dataset, arm][metric] for arm in ARMS[:2]); ax = axes[row, col]
            bars = ax.bar(range(4), [lookup[dataset, a][metric]/best for a in ARMS]+[historical[metric]/best], color=(*COLORS, '#b9b9b9'))
            bars[-1].set_hatch('//')
            ax.axhline(1, color='black', ls='--', lw=.8)
            ax.set_xticks(range(4), (*LABELS, 'Swi256 historical'), rotation=35, ha='right', fontsize=7)
            ax.set(ylabel=name+' / better matched SwiGLU', title=title if row == 0 else ''); ax.grid(axis='y', alpha=.15)
    fig.suptitle('Gray/hatching: existing higher-cost SwiGLU256 context, excluded from the matched-budget screen', fontsize=10)
    save(fig, 'sigmoid-glu-endpoints')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    for ax, dataset, title in zip(axes, DATASETS, TITLES):
        for arm, label, color in zip(ARMS, LABELS, COLORS):
            trace = [r for r in curves if r['dataset'] == dataset and r['label'] == arm]
            ax.plot([r['step'] for r in trace], [r['validation_oracle_mse'] for r in trace], label=label, color=color)
        ax.set(title=title, xlabel='Training update', ylabel='Validation oracle MSE', yscale='log'); ax.grid(alpha=.15)
    axes[-1].legend(fontsize=8, frameon=False); save(fig, 'sigmoid-glu-learning')
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained'); axes = axes.ravel()
    for arm, label, color in zip(ARMS, LABELS, COLORS):
        trace = [next(r for r in regions if r['label'] == arm and r['time'] == t and r['entropy_bin'] == 'all') for t in TIMES]
        axes[0].plot(TIMES, [r['total_mse'] for r in trace], marker='o', label=label, color=color)
    axes[0].set(title='Product16: all coordinates', xlabel='Time', ylabel='Velocity MSE'); axes[0].legend(fontsize=8, frameon=False)
    for ax, time in zip(axes[1:], TIMES):
        for i, (arm, label, color) in enumerate(zip(ARMS, LABELS, COLORS)):
            values = [next(r['conditional_mse'] for r in regions if r['label'] == arm and r['time'] == time and r['entropy_bin'] == name) for name in REGIONS]
            ax.bar(np.arange(4)+(i-1)*.25, [np.nan if v is None else v for v in values], width=.25, color=color, label=label)
        ax.set(title=f'Time {time}: conditional error', ylabel='Velocity MSE')
        ax.set_xticks(range(4), REGIONS, fontsize=8)
    for ax in axes:ax.grid(axis='y', alpha=.15)
    fig.suptitle('All predeclared entropy bins; empty-bin means have no bar. These are velocity errors, not gate classification errors.', fontsize=10)
    save(fig, 'sigmoid-glu-posterior-regions')


def analyze(current, output):
    current, output = current.resolve(), output.resolve(); source = current/'source'
    assert output != current and output not in current.parents
    for protected in (source, current/'output', current/'preflight'):
        assert output != protected and protected not in output.parents
    hashes = {}
    def read(path):
        hashes[str(path)] = sha(path); return json.loads(path.read_text())
    def load(path):
        hashes[str(path)] = sha(path); return torch.load(path, map_location='cpu', weights_only=False)
    def verify(root):
        manifest = read(root/'source-manifest.json')
        for name, digest in manifest['all_files'].items():assert sha(root/'source'/name) == digest, name
        return manifest
    def folder(study, cfg, dataset, kind):return study/'runs'/cfg['experiment_id']/dataset/kind/'seed000'

    manifest = verify(current)
    assert read(current/'output/sigmoid_glu_queue_20261025_m1/status.json')['state'] == 'COMPLETE', 'Partial results refused'
    for name in ('summary.json', 'full-width-initialization.json', 'all-arm-initialization.json', 'glu-trainer-resume.json',
                 'historical-baseline-trainer.json', 'historical-model-trajectories.json', 'dependency-check.json'):
        assert read(current/'preflight'/name)['state'] == 'PASS'
    studies = [(suffix, current/'output'/f'sigmoid_glu_{suffix}_20261025_m1') for suffix in ('v1', 'product16')]
    for _, study in studies:assert read(study/'status.json')['state'] == 'COMPLETE'
    oldroot = current.parent/'state_clock_20261025_m1'; verify(oldroot)
    assert Path(manifest['parent_source']).resolve() == oldroot/'source'
    assert manifest['parent_manifest_sha256'] == sha(oldroot/'source-manifest.json')
    output.mkdir(parents=True, exist_ok=True); archived = output/'source'; archived.mkdir(exist_ok=True)
    replay = Path(__file__).resolve().with_name('audit_architecture_endpoints.py')
    region_source = Path(__file__).resolve().with_name('diagnose_saved_velocity_regions.py')
    region_protocol = Path(__file__).resolve().with_name('protocol-posterior-regions.md')
    if not region_protocol.exists():region_protocol = Path(__file__).resolve().parents[1]/'docs/research-architecture-search/protocol-posterior-regions.md'
    for path, digest in ((replay, REPLAY_SHA), (region_source, REGION_SOURCE_SHA), (region_protocol, REGION_PROTOCOL_SHA)):
        assert sha(path) == digest, path
    for original in (Path(__file__).resolve(), replay, region_source, region_protocol):
        destination = archived/original.name
        if original != destination:shutil.copy2(original, destination)
        hashes[str(destination)] = sha(destination)
    sys.path.insert(0, str(source)); torch.set_num_threads(1)
    from synthetic.models import VelocityNet
    from scripts.run_architecture_study import macs
    rows, curves, initials, baselines, context, monitors, audits, replays = ([] for _ in range(8))
    rng_checks = 0
    for suffix, study in studies:
        data = Path('/Users/mr.z/ode-workspace-prep/data/synthetic')/('v1' if suffix == 'v1' else 'product16_v1')
        for name, args in (
            ('native', [source/'scripts/analyze_architecture_study.py', '--study', study, '--source', source, '--data-root', data, '--output', output/f'{suffix}-native']),
            ('replay', [archived/replay.name, '--study', study, '--source', source, '--data-root', data, '--output', output/f'{suffix}-endpoint-replay.json']),
        ):
            result = subprocess.run([sys.executable, *map(str, args)], cwd=source, capture_output=True, text=True)
            (output/f'{suffix}-{name}.log').write_text(result.stdout+result.stderr)
            assert result.returncode == 0, result.stderr
        audit, receipt = read(output/f'{suffix}-native/audit.json'), read(output/f'{suffix}-endpoint-replay.json')
        assert audit['state'] == receipt['state'] == 'PASS'; audits.append(audit); replays.append(receipt)
        rows.extend(dict(**r, study=str(study)) for r in audit['rows'])
        provenance = read(study/'provenance.json'); plan = provenance['plan']
        assert [a['label'] for a in plan['arms']] == list(ARMS)
        assert plan['seed'] == 0 and plan['steps'] == 5000 and plan['device'] == 'cpu'
        telemetry = study/'telemetry.jsonl'; hashes[str(telemetry)] = sha(telemetry)
        history = [json.loads(line) for line in telemetry.read_text().splitlines()]
        assert history and all(r['memory_available_gib'] > 0 for r in history)
        monitors.append(dict(study=study.name, rows=len(history), first=history[0], last_actual_sample=history[-1],
                             minimum_available_gib=min(r['memory_available_gib'] for r in history), terminal=read(study/'status.json')))
        oldstudy = oldroot/'output'/f'state_clock_{suffix}_20261025_m1'
        assert read(oldstudy/'status.json')['state'] == 'COMPLETE'
        assert read(oldstudy/'provenance.json')['dataset_manifest_sha256'] == provenance['dataset_manifest_sha256']
        for dataset in plan['datasets']:
            dim = 16 if dataset.startswith('rotated') else 2
            reference_rng, swi_initial = {}, None
            for arm in plan['arms']:
                label, kind = arm['label'], arm['model']; cfg = read(study/'configs'/f'{label}.json')
                run = folder(study, cfg, dataset, kind)
                reference = cfg['backbone_initialization'][dataset]
                assert sha(reference['checkpoint']) == reference['sha256']; ref = load(Path(reference['checkpoint']))
                paths = sorted((run/'checkpoints').glob('step_*.pt'))
                assert [p.name for p in paths] == [f'step_{s:07d}.pt' for s in STEPS]
                saved = {step: load(path) for step, path in zip(STEPS, paths)}; initial = saved[0]
                torch.manual_seed(0); model = VelocityNet(dim, kind, **cfg['model_options'])
                native_rng, native = torch.get_rng_state().clone(), model.state_dict(); common, ffn = [], []
                for name, value in initial['model'].items():
                    if '.ffn.' in name:equal(value, native[name]); ffn.append(name)
                    else:equal(value, ref['model'][name]); common.append(name)
                equal(initial['torch_rng'], native_rng); equal(initial['runtime_buffers'], ref['runtime_buffers'])
                receipt = read(run/'provenance.json')['backbone_initialization']
                assert receipt['source_sha256'] == reference['sha256'] and receipt['copied_state_keys'] == sorted(common)
                assert receipt['copied_runtime_buffer_keys'] == sorted(initial['runtime_buffers'])
                if label == 'swiglu':swi_initial = initial
                if label == 'sigmoid_glu':
                    assert kind == 'glu' and cfg['model_options']['ffn_hidden'] == 128
                    for key in ('model', 'runtime_buffers', 'torch_rng', 'numpy_rng', 'python_rng'):equal(initial[key], swi_initial[key])
                row = next(r for r in rows if r['dataset'] == dataset and r['label'] == label)
                assert model.parameter_count() == row['parameters'] and macs(model, dim) == row['forward_macs']
                if label == 'sigmoid_glu':
                    matched = next(r for r in rows if r['dataset'] == dataset and r['label'] == 'swiglu')
                    for key in ('parameters', 'forward_macs'):equal(row[key], matched[key])
                initials.append(dict(dataset=dataset, label=label, reference=reference, native_ffn_entries=len(ffn),
                                     non_ffn_tensors=len(common), native_ffn_rng_unchanged=True,
                                     tensors_buffers_rng_paired_to_Swi128=label == 'sigmoid_glu'))
                oldcfg = read(oldstudy/'configs'/f'{label}.json') if label in ARMS[:2] else None
                if oldcfg:
                    for key in ('steps', 'batch_size', 'learning_rate', 'weight_decay', 'grad_clip', 'eval_interval', 'sample_interval',
                                'checkpoint_interval', 'sample_count', 'sliced_directions', 'heun_steps', 'sampling_split', 'model_options', 'backbone_initialization'):
                        assert cfg[key] == oldcfg[key], (label, key)
                for step, cp in saved.items():
                    if label == 'swiglu_budget':reference_rng[step] = cp['data_rng']
                    else:equal(cp['data_rng'], reference_rng[step]); rng_checks += 1
                    if oldcfg:
                        oldcp = load(folder(oldstudy, oldcfg, dataset, kind)/'checkpoints'/f'step_{step:07d}.pt')
                        for key in STATE_KEYS:equal(cp[key], oldcp[key])
                        baselines.append(dict(dataset=dataset, label=label, step=step, state='BIT_IDENTICAL', compared_keys=list(STATE_KEYS)))
                metrics = run/'metrics.jsonl'; hashes[str(metrics)] = sha(metrics)
                trace = [json.loads(line) for line in metrics.read_text().splitlines()]
                assert [r['step'] for r in trace] == list(range(0, 5001, 500))
                for metric in trace:
                    assert not metric['ode_layers']
                    curves.append(dict(dataset=dataset, label=label, step=metric['step'], train_pair_mse=metric['train_pair_mse'],
                                       validation_pair_mse=metric['validation_pair_mse'], validation_oracle_mse=metric['validation_oracle_mse'],
                                       wall_seconds=metric['wall_seconds'], grad_norm_before_clip=metric['grad_norm_before_clip'],
                                       validation_sw2=metric.get('sample', {}).get('sliced_w2')))
            # Historical larger model: verified context, neither a matched arm
            # nor a newly replayed/independent training repetition.
            historical = dict(next(r for r in read(oldstudy/'results.json') if r['dataset'] == dataset and r['label'] == 'swiglu_compute'))
            cfg = read(oldstudy/'configs/swiglu_compute.json'); assert cfg['model_options']['ffn_hidden'] == 256
            run = folder(oldstudy, cfg, dataset, 'swiglu'); cp_path = run/'checkpoints/step_0005000.pt'; cp = load(cp_path)
            assert cp['identity'] == dict(dataset=dataset, model='swiglu', seed=0) and cp['step'] == 5000
            assert cp['config_sha256'] == sha(oldstudy/'configs/swiglu_compute.json')
            assert cp['dataset_manifest_sha256'] == provenance['dataset_manifest_sha256']
            diagnostic = read(oldstudy/'diagnostics'/dataset/'swiglu_compute/summary.json')
            assert diagnostic['checkpoint_sha256'] == sha(cp_path) and diagnostic['config_sha256'] == cp['config_sha256']
            assert diagnostic['manifest_sha256'] == provenance['dataset_manifest_sha256']
            assert diagnostic['targets_sha256'] == sha(data/dataset/'validation.npz')
            metric_path = run/'metrics.jsonl'; hashes[str(metric_path)] = sha(metric_path)
            final = json.loads(metric_path.read_text().splitlines()[-1]); assert final['step'] == 5000 and not final['ode_layers']
            for name in ('validation_oracle_mse', 'wall_seconds'):equal(historical[name], final[name])
            equal(historical['validation_sw2'], final['sample']['sliced_w2'])
            for name in ('field_mse', 'jacobian_error'):equal(historical[name], diagnostic[name])
            model = VelocityNet(dim, 'swiglu', **cfg['model_options'])
            assert model.parameter_count() == historical['parameters'] and macs(model, dim) == historical['forward_macs']
            matched = next(r for r in rows if r['dataset'] == dataset and r['label'] == 'swiglu')
            assert historical['parameters'] > matched['parameters'] and historical['forward_macs'] > matched['forward_macs']
            context.append(dict(**historical, study=str(oldstudy), context='Historical higher-cost SwiGLU256; not matched-budget or independent evidence',
                                parameters_over_Swi128=historical['parameters']/matched['parameters'], MACs_over_Swi128=historical['forward_macs']/matched['forward_macs']))

    checkpoint_count = sum(a['checkpoints_checked'] for a in audits); array_count = sum(a['arrays_checked'] for a in audits)
    replay_count = sum(a['final_checkpoints_reloaded'] for a in replays)
    assert (checkpoint_count, array_count, replay_count) == (54, 45, 9)
    assert len(rows) == len(initials) == 9 and len(curves) == 99 and len(baselines) == rng_checks == 36 and len(context) == 3
    rows.sort(key=lambda r: (DATASETS.index(r['dataset']), ARMS.index(r['label'])))
    for name, records in (('comparison', rows), ('learning-curves', curves), ('higher-cost-historical-context', context)):
        csv_write(output/(name+'.csv'), records)
    regions, region_receipt = posterior_regions(studies[1][1], Path('/Users/mr.z/ode-workspace-prep/data/synthetic/product16_v1'), output, read, hashes)
    write(output/'initialization-audit.json', dict(state='PASS', rows=initials, snapshot_data_rng_comparisons=rng_checks,
          preflight_sha256=sha(current/'preflight/full-width-initialization.json'),
          scope='GLU/Swi128 initial tensors, buffers and RNG match. Native hidden functions and output-head responses differ; zero final velocity outputs do not establish function/NTK pairing.'))
    write(output/'baseline-trajectory-replay.json', dict(state='PASS', rows=baselines, scope='Both SwiGLUs exactly reproduce Stage21 saved states; deterministic compatibility reruns, not independent repetitions.'))
    write(output/'monitoring-evidence.json', monitors)
    decision = registered_screen(rows); write(output/'screen-decision.json', decision)
    make_figures(output, rows, context, curves, regions); verify(current)
    write(output/'provenance.json', dict(state='PASS', analysis_sha256=sha(__file__), input_sha256=hashes,
          source_files_verified=len(manifest['all_files']), checkpoints_native_audited=checkpoint_count, arrays_native_audited=array_count,
          endpoint_replays=replay_count, historical_larger_context_checkpoints_verified=len(context),
          maximum_forward_replay_difference=max(a['maximum_forward_difference'] for a in replays),
          maximum_jacobian_replay_difference=max(a['maximum_derivative_difference'] for a in replays),
          entropy_protocol_sha256=REGION_PROTOCOL_SHA, entropy_reference_source_sha256=REGION_SOURCE_SHA,
          scope='New native audits cover only nine current cells. Swi256 context is from the completed prior stage, with hashed source/checkpoint/config/data/metrics; no new replay or matched-cost claim. Wall time includes concurrent training/evaluation and is not isolated latency. Exposed validation only; no ODE mechanism, fresh holdout, or generated-path replay.'))
    summary = dict(state='PASS', rows=len(rows), curve_rows=len(curves), baseline_saved_states_equal=len(baselines),
                   snapshot_rng_comparisons=rng_checks, checkpoints_native_audited=checkpoint_count, arrays_native_audited=array_count,
                   endpoint_replays=replay_count, historical_higher_cost_context_rows=len(context), posterior_region_rows=region_receipt['rows'],
                   generic_baseline_lead=decision['generic_baseline_lead'], architecture_goal_complete=False)
    write(output/'summary.json', summary); return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(analyze(args.current, args.output), indent=2))
