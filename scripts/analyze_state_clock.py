"""Audit the completed state-clock study; never analyze partial training results."""
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
ARMS = ('state_euler', 'state_frozen', 'state_midpoint', 'swiglu_budget', 'swiglu', 'swiglu_compute')
LABELS = ('Euler', 'Frozen K2', 'Midpoint', 'SwiGLU96', 'SwiGLU128', 'SwiGLU256')
COLORS = ('#bd9940', '#9e7b53', '#bb4242', '#9fbccc', '#497f9e', '#254c73')
SOLVERS = dict(zip(ARMS[:3], ('euler', 'frozen', 'midpoint')))
LAYERS = tuple(f'blocks.{i}.ffn.ode{j}' for i in range(3) for j in (1, 2))
STEPS = tuple(range(0, 5001, 1000))
METRIC_STEPS = tuple(range(0, 5001, 500))
METRICS = ('validation_oracle_mse', 'jacobian_error', 'validation_sw2')
STATE_KEYS = ('model', 'optimizer', 'runtime_buffers', 'data_rng', 'torch_rng', 'numpy_rng', 'python_rng')
REPLAY_SHA = '142bbe47e3ddf7f25befb1f0eda3b67d187122505805fcf94d4a5c62477640b7'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


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


def matrix_record(state, initial, dataset, label, step, layer, spectrum):
    p = state[layer+'.proj.weight'].double()
    p0 = initial[layer+'.proj.weight'].double()
    raw = spectrum(p)
    row = dict(dataset=dataset, label=label, step=step, layer=layer,
               P_frobenius_norm=float(p.norm()), P_relative_drift=float((p-p0).norm()/p0.norm()))
    row.update({'P_'+key: value for key, value in raw.items() if key != 'sigma_quantiles'})
    for prefix, index in (('controller_first', 0), ('controller_last', 2)):
        weight = state[f'{layer}.t_func.{index}.weight'].double()
        bias = state[f'{layer}.t_func.{index}.bias'].double()
        weight0 = initial[f'{layer}.t_func.{index}.weight'].double()
        bias0 = initial[f'{layer}.t_func.{index}.bias'].double()
        row[prefix+'_weight_norm'] = float(weight.norm())
        row[prefix+'_weight_drift_norm'] = float((weight-weight0).norm())
        row[prefix+'_bias_norm'] = float(bias.norm())
        row[prefix+'_bias_drift_norm'] = float((bias-bias0).norm())
    row['controller_last_weight_norm_over_tau'] = row['controller_last_weight_norm']/10
    # Independently sorted spectra. This index does not track a mode across
    # checkpoints and does not pair eigenvectors with singular vectors.
    singular = np.linalg.svd(p.numpy(), compute_uv=False)
    eigenvalues = np.linalg.eigvals(p.numpy())
    eigenvalues = eigenvalues[np.lexsort((eigenvalues.imag, eigenvalues.real))]
    channels = [dict(dataset=dataset, label=label, step=step, layer=layer, sorted_index=i,
                     singular_value=float(singular[i]), eigen_real=float(eigenvalues[i].real),
                     eigen_imag=float(eigenvalues[i].imag)) for i in range(len(singular))]
    return row, channels, raw


def registered_screen(rows):
    lookup = {(r['dataset'], r['label']): r for r in rows}
    details, leads = {}, []
    for dataset in DATASETS:
        candidate = lookup[dataset, 'state_midpoint']
        best = {m: min(lookup[dataset, arm][m] for arm in ARMS[3:]) for m in METRICS}
        assert all(value > 0 for value in best.values())
        ratios = {m: candidate[m]/best[m] for m in METRICS}
        controls = {arm: candidate['validation_oracle_mse']/lookup[dataset, arm]['validation_oracle_mse']
                    for arm in ARMS[:2]}
        if dataset in DATASETS[1:] and ratios['validation_oracle_mse'] <= .9 and all(v <= .95 for v in controls.values()):
            leads.append(dataset)
        details[dataset] = dict(ratio_to_better_of_three_swiglus=ratios, oracle_ratio_to_ODE_controls=controls)
    primary = all(details[d]['ratio_to_better_of_three_swiglus']['validation_oracle_mse'] <= 1.05 for d in DATASETS)
    secondary = all(details[d]['ratio_to_better_of_three_swiglus'][m] <= 1.1 for d in DATASETS[1:] for m in METRICS[1:])
    passes = bool(leads) and primary and secondary
    return dict(state='NOMINATION' if passes else 'STOP_SETTING', candidate='state_midpoint',
                nominated=['state_midpoint'] if passes else [], qualifying_nonlinear_leads=leads,
                all_primary_within_5pct=primary, nonlinear_Jacobian_SW2_within_10pct=secondary, details=details,
                scope='Only midpoint is a candidate. Registered engineering screen on exposed validation; no significance, full-flow stability or uniquely ODE mechanism claim. A nominee still requires fresh holdout, transfer and isolated cost checks. Failure stops this setting without rescue.')


def make_figures(output, rows, curves, flows, matrices):
    directory = output/'figures'
    directory.mkdir(exist_ok=True)
    lookup = {(r['dataset'], r['label']): r for r in rows}
    def save(fig, name):
        for extension in ('png', 'pdf'):
            fig.savefig(directory/f'{name}.{extension}', dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), layout='constrained')
    for col, (dataset, title) in enumerate(zip(DATASETS, TITLES)):
        for row, (metric, name) in enumerate(zip(METRICS, ('Oracle MSE', 'Jacobian error', 'Validation SW2'))):
            ax = axes[row, col]
            best = min(lookup[dataset, arm][metric] for arm in ARMS[3:])
            ax.bar(range(6), [lookup[dataset, arm][metric]/best for arm in ARMS], color=COLORS)
            ax.axhline(1, color='black', ls='--', lw=.8)
            ax.set_xticks(range(6), LABELS, rotation=40, ha='right', fontsize=7)
            ax.set(ylabel=name+' / best of three SwiGLUs', title=title if row == 0 else '')
            ax.grid(axis='y', alpha=.15)
    save(fig, 'state-clock-endpoints')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout='constrained')
    for ax, dataset, title in zip(axes, DATASETS, TITLES):
        for arm, label, color in zip(ARMS, LABELS, COLORS):
            trace = [r for r in curves if r['dataset'] == dataset and r['label'] == arm]
            ax.plot([r['step'] for r in trace], [r['validation_oracle_mse'] for r in trace], label=label, color=color)
        ax.set(title=title, xlabel='Training update', ylabel='Validation oracle MSE', yscale='log')
        ax.grid(alpha=.15)
    axes[-1].legend(fontsize=7, frameon=False)
    save(fig, 'state-clock-learning')
    fig, axes = plt.subplots(3, 3, figsize=(14, 9), layout='constrained')
    def trace(ax, records, metric, label, color, style='-'):
        steps = sorted({r['step'] for r in records})
        values = np.array([[r[metric] for r in records if r['step'] == s] for s in steps])
        ax.plot(steps, values.mean(1), label=label, color=color, ls=style)
        ax.fill_between(steps, values.min(1), values.max(1), color=color, alpha=.08)
    for col, (dataset, title) in enumerate(zip(DATASETS, TITLES)):
        for i, arm in enumerate(ARMS[:3]):
            trace(axes[0, col], [r for r in matrices if r['dataset'] == dataset and r['label'] == arm],
                  'P_sigma_max', LABELS[i], COLORS[i])
            trace(axes[1, col], [r for r in flows if r['dataset'] == dataset and r['label'] == arm],
                  'gate_change_rms', LABELS[i]+(' (probe)' if i < 2 else ' (executed)'), COLORS[i], '--' if i < 2 else '-')
        selected = [r for r in flows if r['dataset'] == dataset and r['label'] == 'state_midpoint']
        for metric, label, color, style in (
            ('midpoint_correction_probe_rms_over_input', 'Finite midpoint correction', '#bb4242', '-'),
            ('clock_taylor_probe_rms_over_input', 'Clock Taylor diagnostic', '#923b8f', '--'),
            ('frozen_quadratic_probe_rms_over_input', 'Frozen quadratic term', '#9e7b53', ':'),
        ):
            trace(axes[2, col], selected, metric, label, color, style)
        for row, ylabel in enumerate(('Raw P largest singular value', 'Initial-to-midpoint clock change RMS', 'Midpoint-arm term RMS / input RMS')):
            axes[row, col].set(ylabel=ylabel, xlabel='Training update', title=title if row == 0 else '')
            axes[row, col].grid(alpha=.15)
    for row in range(3):
        axes[row, -1].legend(fontsize=7, frameon=False)
    fig.suptitle('Means and ranges over six solves; bands are not uncertainty. Euler/frozen midpoint probes are not executed stages.', fontsize=9)
    save(fig, 'state-clock-internals')


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
    def verify(root):
        manifest = read(root/'source-manifest.json')
        for name, digest in manifest['all_files'].items():
            assert sha(root/'source'/name) == digest, name
        return manifest
    def folder(study, cfg, dataset, kind):
        return study/'runs'/cfg['experiment_id']/dataset/kind/'seed000'

    manifest = verify(current)
    assert read(current/'output/state_clock_queue_20261025_m1/status.json')['state'] == 'COMPLETE', 'Partial study refused'
    for name in ('summary.json', 'full-width-initialization.json', 'all-arm-initialization.json',
                 'state-clock-trainer-resume.json', 'historical-baseline-trainer.json', 'forced-reference-check.json'):
        assert read(current/'preflight'/name)['state'] == 'PASS'
    paired_path = current/'preflight/full-width-initialization.json'
    studies = [(suffix, current/'output'/f'state_clock_{suffix}_20261025_m1') for suffix in ('v1', 'product16')]
    for _, study in studies:
        assert read(study/'status.json')['state'] == 'COMPLETE', study
    oldroot = current.parent/'forced_gate_20261024_m1'
    verify(oldroot)
    assert Path(manifest['parent_source']).resolve() == oldroot/'source'
    assert manifest['parent_manifest_sha256'] == sha(oldroot/'source-manifest.json')
    output.mkdir(parents=True, exist_ok=True)
    archived = output/'source'
    archived.mkdir(exist_ok=True)
    replay_source = Path(__file__).resolve().with_name('audit_architecture_endpoints.py')
    assert sha(replay_source) == REPLAY_SHA
    for original in (Path(__file__).resolve(), replay_source):
        destination = archived/original.name
        if original != destination:
            shutil.copy2(original, destination)
        hashes[str(destination)] = sha(destination)
    sys.path.insert(0, str(source))
    torch.set_num_threads(1)
    from synthetic.models import VelocityNet
    from synthetic.diagnostics import spectrum
    from scripts.run_architecture_study import macs

    rows, curves, initials, flows, raw_flows, matrices, spectral_channels, baselines = ([] for _ in range(8))
    monitors, audits, replays = [], [], []
    rng_checks = 0
    for suffix, study in studies:
        data = Path('/Users/mr.z/ode-workspace-prep/data/synthetic')/('v1' if suffix == 'v1' else 'product16_v1')
        for label, args in (
            ('native', [source/'scripts/analyze_architecture_study.py', '--study', study, '--source', source,
                        '--data-root', data, '--output', output/f'{suffix}-native']),
            ('replay', [archived/'audit_architecture_endpoints.py', '--study', study, '--source', source,
                        '--data-root', data, '--output', output/f'{suffix}-endpoint-replay.json']),
        ):
            result = subprocess.run([sys.executable, *map(str, args)], cwd=source, capture_output=True, text=True)
            (output/f'{suffix}-{label}.log').write_text(result.stdout+result.stderr)
            assert result.returncode == 0, f'{suffix} {label}: {result.stderr}'
        audit = read(output/f'{suffix}-native/audit.json')
        replay = read(output/f'{suffix}-endpoint-replay.json')
        assert audit['state'] == replay['state'] == 'PASS'
        audits.append(audit); replays.append(replay)
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
                             minimum_available_gib=min(r['memory_available_gib'] for r in history), terminal=read(study/'status.json')))
        for dataset in plan['datasets']:
            dim = 16 if dataset.startswith('rotated') else 2
            reference_rng, ode_initial, ode_rng = {}, None, None
            for arm in plan['arms']:
                label, kind = arm['label'], arm['model']
                cfg = read(study/'configs'/f'{label}.json')
                run = folder(study, cfg, dataset, kind)
                reference = cfg['backbone_initialization'][dataset]
                assert sha(reference['checkpoint']) == reference['sha256']
                ref = load(Path(reference['checkpoint']))
                paths = sorted((run/'checkpoints').glob('step_*.pt'))
                assert [p.name for p in paths] == [f'step_{s:07d}.pt' for s in STEPS]
                saved = {step: load(path) for step, path in zip(STEPS, paths)}
                initial = saved[0]
                torch.manual_seed(0)
                model = VelocityNet(dim, kind, **cfg['model_options'])
                native_rng, native = torch.get_rng_state().clone(), model.state_dict()
                common, ffn = [], []
                for name, value in initial['model'].items():
                    if '.ffn.' in name:
                        equal(value, native[name]); ffn.append(name)
                    else:
                        equal(value, ref['model'][name]); common.append(name)
                equal(initial['torch_rng'], native_rng)
                equal(initial['runtime_buffers'], ref['runtime_buffers'])
                receipt = read(run/'provenance.json')['backbone_initialization']
                assert receipt['source_sha256'] == reference['sha256'] and receipt['copied_state_keys'] == sorted(common)
                assert receipt['copied_runtime_buffer_keys'] == sorted(initial['runtime_buffers'])
                if label == 'state_euler':
                    ode_initial, ode_rng = initial['model'], initial['torch_rng']
                if label in SOLVERS:
                    assert kind == 'ode_state_clock' and cfg['model_options']['state_clock_solver'] == SOLVERS[label]
                    equal(initial['model'], ode_initial); equal(initial['torch_rng'], ode_rng)
                    for layer in LAYERS:
                        for parameter in ('weight', 'bias'):
                            assert torch.count_nonzero(initial['model'][f'{layer}.t_func.2.{parameter}']) == 0
                row = next(r for r in rows if r['dataset'] == dataset and r['label'] == label)
                assert model.parameter_count() == row['parameters'] and macs(model, dim) == row['forward_macs']
                initials.append(dict(dataset=dataset, label=label, reference=reference, non_ffn_tensors=len(common),
                                     native_ffn_entries=len(ffn), native_ffn_rng_unchanged=True,
                                     ODE_state_tensors_and_rng_paired=label in SOLVERS,
                                     controller_final_affines_zero=label in SOLVERS))
                old_saved = {}
                if label in ('swiglu_budget', 'swiglu'):
                    oldstudy = oldroot/'output'/f'forced_gate_{suffix}_20261024_m1'
                    oldcfg = read(oldstudy/'configs'/f'{label}.json')
                    assert read(oldstudy/'provenance.json')['dataset_manifest_sha256'] == provenance['dataset_manifest_sha256']
                    for key in ('steps', 'batch_size', 'learning_rate', 'weight_decay', 'grad_clip', 'eval_interval',
                                'sample_interval', 'checkpoint_interval', 'sample_count', 'sliced_directions',
                                'heun_steps', 'sampling_split', 'model_options', 'backbone_initialization'):
                        assert cfg[key] == oldcfg[key], (label, key)
                    old_saved = {step: load(folder(oldstudy, oldcfg, dataset, kind)/'checkpoints'/f'step_{step:07d}.pt') for step in STEPS}
                raw_checkpoints = {}
                for step, cp in saved.items():
                    if label == 'state_euler':
                        reference_rng[step] = cp['data_rng']
                    else:
                        equal(cp['data_rng'], reference_rng[step]); rng_checks += 1
                    if old_saved:
                        for key in STATE_KEYS:
                            equal(cp[key], old_saved[step][key])
                        baselines.append(dict(dataset=dataset, label=label, step=step, state='BIT_IDENTICAL', compared_keys=list(STATE_KEYS)))
                    if label in SOLVERS:
                        for layer in LAYERS:
                            matrix, channels, raw = matrix_record(cp['model'], initial['model'], dataset, label, step, layer, spectrum)
                            matrices.append(matrix); spectral_channels.extend(channels)
                            raw_checkpoints[step, layer] = raw
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
                    if label not in SOLVERS:
                        assert not metric['ode_layers']
                        continue
                    entries = metric['ode_layers']
                    assert len(entries) == 6 and {e['layer'] for e in entries} == set(LAYERS)
                    for entry in entries:
                        name, activation, raw = entry['layer'], entry['activation'], entry['raw']
                        assert entry['operator_type'] == 'state_clock_vector_field'
                        assert entry['solver'] == activation['solver'] == SOLVERS[label]
                        assert activation['scope'] == 'internal_state_clock_probes_only'
                        assert activation['midpoint_clock_is_executed_stage'] == (label == 'state_midpoint')
                        assert activation['projection_calls'] == (1 if label == 'state_euler' else 2)
                        assert activation['controller_calls'] == (2 if label == 'state_midpoint' else 1)
                        assert (activation['controller_tau'], activation['clock_scale'], activation['clock_shift']) == (10., .8, .2)
                        if step in saved:
                            for key in raw:
                                np.testing.assert_allclose(raw[key], raw_checkpoints[step, name][key], rtol=1e-11, atol=1e-12)
                        raw_flows.append(dict(dataset=dataset, label=label, step=step, **entry))
                        record = dict(dataset=dataset, label=label, step=step, layer=name,
                                      midpoint_probe_interpretation='executed stage' if label == 'state_midpoint' else 'counterfactual probe')
                        record.update({'P_'+key: value for key, value in raw.items() if key != 'sigma_quantiles'})
                        record.update(activation)
                        flows.append(record)

    assert len(rows) == len(initials) == 18 and len(curves) == 198
    assert len(baselines) == 36 and rng_checks == 90
    assert len(flows) == len(raw_flows) == 594 and len(matrices) == 324 and len(spectral_channels) == 20736
    checkpoint_count = sum(a['checkpoints_checked'] for a in audits)
    array_count = sum(a['arrays_checked'] for a in audits)
    replay_count = sum(a['final_checkpoints_reloaded'] for a in replays)
    assert (checkpoint_count, array_count, replay_count) == (108, 90, 18)
    rows.sort(key=lambda r: (DATASETS.index(r['dataset']), ARMS.index(r['label'])))
    for name, values in (('comparison', rows), ('learning-curves', curves), ('state-clock-trajectories', flows),
                         ('P-checkpoint-trajectories', matrices), ('P-spectral-channels', spectral_channels)):
        csv_write(output/(name+'.csv'), values)
    write(output/'raw-state-clock-trajectory.json', raw_flows)
    write(output/'initialization-audit.json', dict(state='PASS', initializations=initials, snapshot_data_rng_comparisons=rng_checks,
          full_width_preflight_sha256=sha(paired_path), scope='All three ODE arms share saved initial tensors/RNG and zero final controller affines. Only frozen/midpoint hidden functions and nonzero-head input/P gradients are paired; controller responses differ. Euler hidden features are different. Preflight receipt supplies the function/gradient proof.'))
    write(output/'baseline-trajectory-replay.json', dict(state='PASS', rows=baselines,
          scope='Both established SwiGLU arms match all six Stage20 saved states exactly. Deterministic compatibility reruns, not independent replications.'))
    write(output/'monitoring-evidence.json', monitors)
    decision = registered_screen(rows)
    write(output/'screen-decision.json', decision)
    make_figures(output, rows, curves, flows, matrices)
    verify(current)
    write(output/'provenance.json', dict(state='PASS', analysis_sha256=sha(__file__), input_sha256=hashes,
          source_files_verified=len(manifest['all_files']), checkpoints_native_audited=checkpoint_count,
          arrays_native_audited=array_count, endpoint_replays=replay_count,
          maximum_forward_replay_difference=max(a['maximum_forward_difference'] for a in replays),
          maximum_jacobian_replay_difference=max(a['maximum_derivative_difference'] for a in replays),
          scope='Exposed validation and internal probes only; Euler/frozen midpoint probes are counterfactual, Taylor probes are not executed terms. Spectral indices independently sort singular/eigenvalues and do not track modes. Raw P spectra do not bound complete flow Jacobians. No fresh holdout, sampling-path replay or isolated latency measurement.'))
    summary = dict(state='PASS', rows=len(rows), curve_rows=len(curves), state_clock_rows=len(flows), P_checkpoint_rows=len(matrices),
                   P_spectral_channel_rows=len(spectral_channels), baseline_saved_states_equal=len(baselines), snapshot_rng_comparisons=rng_checks,
                   checkpoints_native_audited=checkpoint_count, arrays_native_audited=array_count,
                   endpoint_replays=replay_count, nominated=decision['nominated'])
    write(output/'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.current, args.output), indent=2))
