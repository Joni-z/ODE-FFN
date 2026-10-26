"""Audit the completed, registered five-arm forced-gate study without retraining."""
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
ARMS = ('swiglu_budget', 'swiglu', 'forced_static', 'forced_k2', 'forced_exact')
LABELS = ('SwiGLU96', 'SwiGLU128', 'Static response', 'Adaptive K2', 'Adaptive exact')
COLORS = ('#7ba8bd', '#255f88', '#9e7b53', '#bb4242', '#923b8f')
STEPS = tuple(range(0, 5001, 1000))
METRIC_STEPS = tuple(range(0, 5001, 500))
STATE_KEYS = ('model', 'optimizer', 'runtime_buffers', 'data_rng', 'torch_rng',
              'numpy_rng', 'python_rng')
METRICS = ('validation_oracle_mse', 'jacobian_error', 'validation_sw2')
SETTINGS = {'forced_static': ('exact', 'static'), 'forced_k2': ('k2', 'adaptive'),
            'forced_exact': ('exact', 'adaptive')}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def csv_write(path, rows):
    assert rows, path
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def projection_record(state, initial, dataset, label, step, block, phi1):
    """Weight summaries, not a bound on the multiplicative FFN Jacobian."""
    prefix = f'blocks.{block}.ffn'
    gate, value = state[prefix + '.w12.weight'].double().chunk(2, dim=0)
    gate0, value0 = initial[prefix + '.w12.weight'].double().chunk(2, dim=0)
    gate_bias, value_bias = state[prefix + '.w12.bias'].double().chunk(2)
    gate_bias0, value_bias0 = initial[prefix + '.w12.bias'].double().chunk(2)
    output = state[prefix + '.w3.weight'].double()
    output0 = initial[prefix + '.w3.weight'].double()
    output_bias = state[prefix + '.w3.bias'].double()
    output_bias0 = initial[prefix + '.w3.bias'].double()
    row = dict(dataset=dataset, label=label, step=step, block=block)
    norms = []
    for name, weight, start, bias, bias0 in (
        ('gate', gate, gate0, gate_bias, gate_bias0),
        ('value', value, value0, value_bias, value_bias0),
        ('output', output, output0, output_bias, output_bias0),
    ):
        norm = float(torch.linalg.matrix_norm(weight, ord=2))
        norms.append(norm)
        row.update({name + '_sigma_max': norm,
                    name + '_frobenius_norm': float(weight.norm()),
                    name + '_relative_drift': float((weight-start).norm()/start.norm()),
                    name + '_bias_norm': float(bias.norm()),
                    name + '_bias_drift_norm': float((bias-bias0).norm())})
    row['gate_value_output_norm_product'] = float(np.prod(norms))
    row['static_absorbed_output_sigma_max'] = None
    row['static_absorbed_output_relative_drift'] = None
    if label == 'forced_static':
        lam = state[prefix + '.flow.raw_eigenvalues'].double().tanh()
        effective = output * phi1(.5*lam)[None, :]
        row['static_absorbed_output_sigma_max'] = float(torch.linalg.matrix_norm(effective, ord=2))
        row['static_absorbed_output_relative_drift'] = float((effective-output0).norm()/output0.norm())
    return row


def registered_screen(rows):
    lookup = {(r['dataset'], r['label']): r for r in rows}
    screens = {}
    for label in ('forced_k2', 'forced_exact'):
        details, leads = {}, []
        for dataset in DATASETS:
            row = lookup[dataset, label]
            best = {m: min(lookup[dataset, b][m] for b in ARMS[:2]) for m in METRICS}
            assert all(value > 0 for value in best.values())
            ratios = {m: row[m]/best[m] for m in METRICS}
            static_ratio = row['validation_oracle_mse']/lookup[dataset, 'forced_static']['validation_oracle_mse']
            k2_ratio = row['validation_oracle_mse']/lookup[dataset, 'forced_k2']['validation_oracle_mse']
            if dataset in DATASETS[1:] and ratios['validation_oracle_mse'] <= .9 and static_ratio <= .95:
                leads.append(dataset)
            details[dataset] = dict(ratio_to_better_swiglu=ratios,
                                    oracle_ratio_to_static=static_ratio, oracle_ratio_to_K2=k2_ratio)
        primary = all(details[d]['ratio_to_better_swiglu']['validation_oracle_mse'] <= 1.05 for d in DATASETS)
        secondary = all(details[d]['ratio_to_better_swiglu'][m] <= 1.1
                        for d in DATASETS[1:] for m in METRICS[1:])
        passes = bool(leads) and primary and secondary
        higher_leads = [d for d in leads if details[d]['oracle_ratio_to_K2'] <= .95] if label == 'forced_exact' else []
        screens[label] = dict(passes=passes, qualifying_nonlinear_leads=leads,
                             all_primary_within_5pct=primary,
                             nonlinear_Jacobian_SW2_within_10pct=secondary,
                             higher_terms_passes=passes and bool(higher_leads),
                             higher_terms_qualifying_nonlinear_leads=higher_leads, details=details)
    nominated = [label for label in screens if screens[label]['passes']]
    return dict(state='NOMINATION' if nominated else 'STOP_FAMILY', nominated=nominated,
                comparisons=screens,
                scope='Registered engineering screen on exposed validation. A nomination is a generic adaptive-gate lead, not confirmation. Exact needs its additional K2 comparison for a higher-terms interpretation; this does not isolate solver order. Fresh holdout, transfer and practical cost/budget checks remain required. No rescue sweep.')


def make_figures(output, rows, curves, flows, projections):
    figures = output/'figures'
    figures.mkdir(exist_ok=True)
    lookup = {(r['dataset'], r['label']): r for r in rows}
    def save(fig, stem):
        for extension in ('png', 'pdf'):
            fig.savefig(figures/f'{stem}.{extension}', dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), layout='constrained')
    for col, (dataset, title) in enumerate(zip(DATASETS, TITLES)):
        for row, (metric, name) in enumerate(zip(METRICS, ('Oracle MSE', 'Jacobian error', 'Validation SW2'))):
            ax = axes[row, col]
            best = min(lookup[dataset, b][metric] for b in ARMS[:2])
            ax.bar(range(5), [lookup[dataset, a][metric]/best for a in ARMS], color=COLORS)
            ax.axhline(1, color='black', ls='--', lw=.8)
            ax.set_xticks(range(5), LABELS, rotation=40, ha='right', fontsize=7)
            ax.set(ylabel=name+' / better SwiGLU', title=title if row == 0 else '')
            ax.grid(axis='y', alpha=.15)
    save(fig, 'forced-gate-endpoints')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), layout='constrained')
    for ax, dataset, title in zip(axes, DATASETS, TITLES):
        for arm, label, color in zip(ARMS, LABELS, COLORS):
            trace = [r for r in curves if r['dataset'] == dataset and r['label'] == arm]
            ax.plot([r['step'] for r in trace], [r['validation_oracle_mse'] for r in trace], label=label, color=color)
        ax.set(title=title, xlabel='Training update', ylabel='Validation oracle MSE', yscale='log')
        ax.grid(alpha=.15)
    axes[-1].legend(fontsize=7, frameon=False)
    save(fig, 'forced-gate-learning')
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), layout='constrained')
    for col, (dataset, title) in enumerate(zip(DATASETS, TITLES)):
        for row, (metric, name) in enumerate((
            ('P_sigma_max', 'max |lambda|; dotted: conditional rate'),
            ('correction_rms_over_baseline', 'Observed correction RMS / baseline gate RMS'),
            ('gate_value_output_norm_product', 'Gate/value/output spectral-norm product'),
        )):
            ax = axes[row, col]
            entries = projections if row == 2 else flows
            for i in (range(5) if row == 2 else range(2, 5)):
                trace = [r for r in entries if r['dataset'] == dataset and r['label'] == ARMS[i]]
                steps = sorted({r['step'] for r in trace})
                values = np.array([[r[metric] for r in trace if r['step'] == s] for s in steps])
                ax.plot(steps, values.mean(1), label=LABELS[i], color=COLORS[i])
                ax.fill_between(steps, values.min(1), values.max(1), color=COLORS[i], alpha=.09)
                if row == 0:
                    rates = np.array([[r['conditional_rate_abs_max'] for r in trace if r['step'] == s] for s in steps])
                    ax.plot(steps, rates.mean(1), color=COLORS[i], ls=':', lw=1)
            ax.set(title=title if row == 0 else '', xlabel='Training update', ylabel=name)
            ax.grid(alpha=.15)
    axes[0, -1].legend(fontsize=7, frameon=False)
    axes[2, -1].legend(fontsize=7, frameon=False)
    fig.suptitle('Lines: three-block mean; shading: block range, not uncertainty. No full-network stability bound.', fontsize=9)
    save(fig, 'forced-gate-internals')


def analyze(current, output):
    current, output = current.resolve(), output.resolve()
    source = current/'source'
    assert output != current and output not in current.parents
    for protected in (source, current/'output', current/'preflight'):
        assert output != protected and protected not in output.parents, 'Output would overwrite training evidence'
    hashes = {}
    def read(path):
        hashes[str(path)] = sha(path)
        return json.loads(path.read_text())
    def load(path):
        hashes[str(path)] = sha(path)
        return torch.load(path, map_location='cpu', weights_only=False)
    def verify_manifest(root):
        manifest = read(root/'source-manifest.json')
        for name, digest in manifest['all_files'].items():
            assert sha(root/'source'/name) == digest, name
        return manifest
    def folder(study, cfg, dataset, kind):
        return study/'runs'/cfg['experiment_id']/dataset/kind/'seed000'

    manifest = verify_manifest(current)
    queue = read(current/'output/forced_gate_queue_20261024_m1/status.json')
    assert queue['state'] == 'COMPLETE', 'Refusing analysis of a partial study'
    preflight = read(current/'preflight/summary.json')
    assert preflight['state'] == 'PASS'
    assert Path(preflight['historical_source']).resolve() == current.parent/'diagonal_flow_20261024_m1/source'
    assert preflight['historical_source_files'] == 561 and preflight['prior_self_comparison_attempt_invalidated']
    paired_path = current/'preflight/full-width-initialization.json'
    assert read(paired_path)['state'] == 'PASS'
    for name in ('all-arm-initialization.json', 'forced-gate-trainer-resume.json', 'historical-baseline-trainer.json'):
        assert read(current/'preflight'/name)['state'] == 'PASS'
    studies = [(suffix, current/'output'/f'forced_gate_{suffix}_20261024_m1') for suffix in ('v1', 'product16')]
    for _, study in studies:
        assert read(study/'status.json')['state'] == 'COMPLETE', study
    oldroot = current.parent/'diagonal_flow_20261024_m1'
    verify_manifest(oldroot)
    output.mkdir(parents=True, exist_ok=True)
    archived = output/'source'
    archived.mkdir(exist_ok=True)
    # The endpoint replay is a standalone posthoc tool, not a frozen training
    # dependency. Its trusted bytes are archived alongside this analyzer.
    replay_source = Path(__file__).resolve().with_name('audit_architecture_endpoints.py')
    assert sha(replay_source) == '142bbe47e3ddf7f25befb1f0eda3b67d187122505805fcf94d4a5c62477640b7'
    for original in (Path(__file__).resolve(), replay_source):
        destination = archived/original.name
        if original != destination:
            shutil.copy2(original, destination)
        hashes[str(destination)] = sha(destination)
    sys.path.insert(0, str(source))
    torch.set_num_threads(1)
    from synthetic.models import VelocityNet
    from synthetic.forced_gate import phi1
    from scripts.run_architecture_study import macs

    rows, curves, initials, flows, channels, raw_flows, projections, baselines = ([] for _ in range(8))
    monitors, audits, replays = [], [], []
    rng_checks = 0
    for suffix, study in studies:
        data = Path('/Users/mr.z/ode-workspace-prep/data/synthetic')/('v1' if suffix == 'v1' else 'product16_v1')
        commands = (
            ('native', [source/'scripts/analyze_architecture_study.py', '--study', study, '--source', source,
                        '--data-root', data, '--output', output/f'{suffix}-native']),
            ('replay', [archived/'audit_architecture_endpoints.py', '--study', study, '--source', source,
                        '--data-root', data, '--output', output/f'{suffix}-endpoint-replay.json']),
        )
        for label, args in commands:
            result = subprocess.run([sys.executable, *map(str, args)], cwd=source, capture_output=True, text=True)
            (output/f'{suffix}-{label}.log').write_text(result.stdout+result.stderr)
            assert result.returncode == 0, f'{suffix} {label}: {result.stderr}'
        audit = read(output/f'{suffix}-native/audit.json')
        replay = read(output/f'{suffix}-endpoint-replay.json')
        assert audit['state'] == replay['state'] == 'PASS'
        audits.append(audit)
        replays.append(replay)
        rows.extend(dict(**r, study=str(study)) for r in audit['rows'])
        provenance = read(study/'provenance.json')
        plan = provenance['plan']
        assert [a['label'] for a in plan['arms']] == list(ARMS)
        assert plan['seed'] == 0 and plan['steps'] == 5000 and plan['device'] == 'cpu'
        telemetry = study/'telemetry.jsonl'
        hashes[str(telemetry)] = sha(telemetry)
        history = [json.loads(line) for line in telemetry.read_text().splitlines()]
        assert history and all(r['memory_available_gib'] > 0 for r in history)
        monitors.append(dict(study=study.name, rows=len(history), first=history[0], last_actual_sample=history[-1],
                             minimum_available_gib=min(r['memory_available_gib'] for r in history),
                             terminal=read(study/'status.json')))
        for dataset in plan['datasets']:
            dim = 16 if dataset.startswith('rotated') else 2
            reference_rng, base_initial, base_rng = {}, None, None
            for arm in plan['arms']:
                label, kind = arm['label'], arm['model']
                cfg = read(study/'configs'/f'{label}.json')
                run = folder(study, cfg, dataset, kind)
                reference = cfg['backbone_initialization'][dataset]
                assert sha(reference['checkpoint']) == reference['sha256']
                ref = load(Path(reference['checkpoint']))
                cp_paths = sorted((run/'checkpoints').glob('step_*.pt'))
                assert [p.name for p in cp_paths] == [f'step_{s:07d}.pt' for s in STEPS]
                saved = {step: load(path) for step, path in zip(STEPS, cp_paths)}
                initial = saved[0]
                torch.manual_seed(0)
                model = VelocityNet(dim, kind, **cfg['model_options'])
                native_rng = torch.get_rng_state().clone()
                native = model.state_dict()
                common, ffn = [], []
                for name, value in initial['model'].items():
                    if '.ffn.' in name:
                        equal(value, native[name])
                        ffn.append(name)
                    else:
                        equal(value, ref['model'][name])
                        common.append(name)
                equal(initial['torch_rng'], native_rng)
                equal(initial['runtime_buffers'], ref['runtime_buffers'])
                receipt = read(run/'provenance.json')['backbone_initialization']
                assert receipt['source_sha256'] == reference['sha256'] and receipt['copied_state_keys'] == sorted(common)
                assert receipt['copied_runtime_buffer_keys'] == sorted(initial['runtime_buffers'])
                if label == 'swiglu':
                    base_initial, base_rng = initial['model'], initial['torch_rng']
                if label in SETTINGS:
                    assert kind == 'ode_forced_gate'
                    for name, value in base_initial.items():
                        equal(value, initial['model'][name])
                    eta_keys = set(initial['model'])-set(base_initial)
                    assert eta_keys == {f'blocks.{i}.ffn.flow.raw_eigenvalues' for i in range(3)}
                    assert all(torch.count_nonzero(initial['model'][name]) == 0 for name in eta_keys)
                    equal(initial['torch_rng'], base_rng)
                row = next(r for r in rows if r['dataset'] == dataset and r['label'] == label)
                assert model.parameter_count() == row['parameters'] and macs(model, dim) == row['forward_macs']
                initials.append(dict(dataset=dataset, label=label, reference=reference, non_ffn_tensors=len(common),
                                     native_ffn_entries=len(ffn), native_ffn_rng_unchanged=True,
                                     SwiGLU128_projection_bias_rng_paired=label in SETTINGS,
                                     eta_zero=label in SETTINGS))
                old_saved = {}
                if label in ARMS[:2]:
                    oldstudy = oldroot/'output'/f'diagonal_flow_{suffix}_20261024_m1'
                    oldcfg = read(oldstudy/'configs'/f'{label}.json')
                    oldprov = read(oldstudy/'provenance.json')
                    assert oldprov['dataset_manifest_sha256'] == provenance['dataset_manifest_sha256']
                    for key in ('steps', 'batch_size', 'learning_rate', 'weight_decay', 'grad_clip', 'eval_interval',
                                'sample_interval', 'checkpoint_interval', 'sample_count', 'sliced_directions',
                                'heun_steps', 'sampling_split', 'model_options', 'backbone_initialization'):
                        assert cfg[key] == oldcfg[key], (label, key)
                    for step in STEPS:
                        old_saved[step] = load(folder(oldstudy, oldcfg, dataset, kind)/'checkpoints'/f'step_{step:07d}.pt')
                for step, cp in saved.items():
                    if label == 'swiglu_budget':
                        reference_rng[step] = cp['data_rng']
                    else:
                        equal(cp['data_rng'], reference_rng[step])
                        rng_checks += 1
                    if label in ARMS[:2]:
                        for key in STATE_KEYS:
                            equal(cp[key], old_saved[step][key])
                        baselines.append(dict(dataset=dataset, label=label, step=step,
                                              state='BIT_IDENTICAL', compared_keys=list(STATE_KEYS)))
                    for block in range(3):
                        projections.append(projection_record(cp['model'], initial['model'], dataset, label, step, block, phi1))
                metric_path = run/'metrics.jsonl'
                hashes[str(metric_path)] = sha(metric_path)
                trace = [json.loads(line) for line in metric_path.read_text().splitlines()]
                assert [r['step'] for r in trace] == list(METRIC_STEPS)
                for metric in trace:
                    step = metric['step']
                    curves.append(dict(dataset=dataset, label=label, step=step, train_pair_mse=metric['train_pair_mse'],
                                       validation_pair_mse=metric['validation_pair_mse'], validation_oracle_mse=metric['validation_oracle_mse'],
                                       wall_seconds=metric['wall_seconds'], grad_norm_before_clip=metric['grad_norm_before_clip'],
                                       validation_sw2=metric.get('sample', {}).get('sliced_w2')))
                    if label not in SETTINGS:
                        assert not metric['ode_layers']
                        continue
                    entries = metric['ode_layers']
                    assert {e['layer'] for e in entries} == {f'blocks.{i}.ffn.flow' for i in range(3)} and len(entries) == 3
                    for entry in entries:
                        name, activation = entry['layer'], entry['activation']
                        eta, lam = np.array(entry['eigenvalue_coordinates']), np.array(entry['P_eigenvalues'])
                        assert len(eta) == len(lam) == 85
                        assert entry['operator_type'] == 'conditional_forced_gate'
                        assert (entry['solver'], entry['clock']) == SETTINGS[label]
                        assert activation['scope'] == 'observed_gate_response_only'
                        np.testing.assert_allclose(lam, np.tanh(eta), atol=8e-8, rtol=2e-7)
                        assert np.max(np.abs(lam)) == entry['P_sigma_max']
                        assert activation['conditional_rate_min'] == entry['conditional_rate_min']
                        assert activation['conditional_rate_max'] == entry['conditional_rate_max']
                        if step in saved:
                            raw = saved[step]['model'][name+'.raw_eigenvalues']
                            equal(torch.tensor(eta, dtype=raw.dtype), raw)
                            equal(torch.tensor(lam, dtype=raw.dtype), raw.tanh())
                        raw_flows.append(dict(dataset=dataset, label=label, step=step, **entry))
                        for channel, (a, l) in enumerate(zip(eta, lam)):
                            channels.append(dict(dataset=dataset, label=label, step=step, layer=name,
                                                 channel=channel, eta=float(a), lambda_value=float(l)))
                        flows.append(dict(dataset=dataset, label=label, step=step, layer=name,
                                          eta_min=float(eta.min()), eta_max=float(eta.max()), eta_norm=float(np.linalg.norm(eta)),
                                          lambda_min=float(lam.min()), lambda_max=float(lam.max()), lambda_mean=float(lam.mean()),
                                          lambda_std=float(lam.std()), lambda_negative_fraction=float(np.mean(lam < 0)),
                                          P_sigma_max=entry['P_sigma_max'], forcing_abs_max=entry['forcing_abs_max'],
                                          conditional_rate_abs_max=max(abs(entry['conditional_rate_min']), abs(entry['conditional_rate_max'])),
                                          **activation))

    assert len(rows) == len(initials) == 15 and len(curves) == 165
    assert len(baselines) == 36 and rng_checks == 72
    assert len(flows) == len(raw_flows) == 297 and len(channels) == 25245 and len(projections) == 270
    assert sum(a['checkpoints_checked'] for a in audits) == 90
    assert sum(a['arrays_checked'] for a in audits) == 75
    assert sum(a['final_checkpoints_reloaded'] for a in replays) == 15
    rows.sort(key=lambda r: (DATASETS.index(r['dataset']), ARMS.index(r['label'])))
    for name, values in (('comparison', rows), ('learning-curves', curves), ('forced-gate-trajectories', flows),
                         ('eigenvalue-channels', channels), ('projection-trajectories', projections)):
        csv_write(output/(name+'.csv'), values)
    write(output/'raw-forced-gate-trajectory.json', raw_flows)
    write(output/'initialization-audit.json', dict(state='PASS', initializations=initials,
          formal_projection_and_bias_pairing=True, eta_zero=True, snapshot_data_rng_comparisons=rng_checks,
          nonzero_head_function_proof='Formal shared projections, biases, buffers and surrounding tensors match the preflight-paired SwiGLU128 state; three extra eta vectors are zero. See the hash-bound full-width preflight proof.',
          full_width_preflight_sha256=sha(paired_path)))
    write(output/'baseline-trajectory-replay.json', dict(state='PASS', rows=baselines,
          scope='All six saved states per SwiGLU baseline match Stage19 exactly. These are deterministic reruns, not independent repeated evidence.'))
    write(output/'monitoring-evidence.json', monitors)
    decision = registered_screen(rows)
    write(output/'screen-decision.json', decision)
    make_figures(output, rows, curves, flows, projections)
    verify_manifest(current)
    provenance = dict(state='PASS', analysis_sha256=sha(__file__), input_sha256=hashes,
                      source_files_verified=len(manifest['all_files']), checkpoints_native_audited=90,
                      arrays_native_audited=75, endpoint_replays=15,
                      maximum_forward_replay_difference=max(a['maximum_forward_difference'] for a in replays),
                      maximum_jacobian_replay_difference=max(a['maximum_derivative_difference'] for a in replays),
                      scope='Exposed validation evidence, recorded scalar gate probes and saved projection matrices. Projection norms and bounded rates/gains are not full-network Jacobian bounds. No fresh holdout, generated sampling-path replay, or isolated runtime comparison.')
    write(output/'provenance.json', provenance)
    summary = dict(state='PASS', rows=len(rows), curve_rows=len(curves), forced_gate_rows=len(flows),
                   eigenvalue_channel_rows=len(channels), projection_rows=len(projections),
                   baseline_saved_states_equal=len(baselines), snapshot_rng_comparisons=rng_checks,
                   checkpoints_native_audited=90, arrays_native_audited=75, endpoint_replays=15,
                   nominated=decision['nominated'], higher_terms_passes=decision['comparisons']['forced_exact']['higher_terms_passes'])
    write(output/'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.current, args.output), indent=2))
