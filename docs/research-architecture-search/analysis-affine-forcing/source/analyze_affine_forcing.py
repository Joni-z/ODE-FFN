"""Audit completed bias-completeness controls and a frozen affine-to-static probe."""
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
ARMS = ('vector_parent', 'static_bias', 'affine_forcing', 'swiglu_budget', 'swiglu', 'sigmoid_glu')
LABELS = ('ODE parent', 'Static bias', 'Affine forcing', 'SwiGLU96', 'SwiGLU128', 'GLU128')
COLORS = ('#b29a51', '#8359a0', '#bb4242', '#a2bfcc', '#255f88', '#549977')
MODES = dict(zip(ARMS[:3], ('none', 'static', 'conditional')))
LAYERS = tuple(f'blocks.{i}.ffn.ode{j}' for i in range(3) for j in (1, 2))
STEPS, TIMES = tuple(range(0, 5001, 1000)), (.1, .3, .5, .7, .9)
METRICS = ('validation_oracle_mse', 'jacobian_error', 'validation_sw2')
STATE_KEYS = ('model', 'optimizer', 'runtime_buffers', 'data_rng', 'torch_rng', 'numpy_rng', 'python_rng')
REPLAY_SHA = '142bbe47e3ddf7f25befb1f0eda3b67d187122505805fcf94d4a5c62477640b7'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def csv_write(path, rows):
    assert rows
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def equal(a, b):
    if isinstance(a, torch.Tensor):torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, np.ndarray):np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):equal(x, y)
    else:assert a == b


def registered_screen(rows):
    lookup = {(r['dataset'], r['label']): r for r in rows}
    decisions = {}
    for arm in ARMS[1:3]:
        details, leads = {}, []
        for dataset in DATASETS:
            row = lookup[dataset, arm]
            best = {m: min(lookup[dataset, generic][m] for generic in ARMS[3:]) for m in METRICS}
            assert all(value > 0 for value in best.values())
            ratios = {m: row[m]/best[m] for m in METRICS}
            parent = row['validation_oracle_mse']/lookup[dataset, 'vector_parent']['validation_oracle_mse']
            if dataset in DATASETS[1:] and ratios['validation_oracle_mse'] <= .9 and parent <= .9:leads.append(dataset)
            details[dataset] = dict(ratio_to_best_generic=ratios, primary_ratio_to_parent=parent)
        primary = all(details[d]['ratio_to_best_generic']['validation_oracle_mse'] <= 1.05 for d in DATASETS)
        secondary = all(details[d]['ratio_to_best_generic'][m] <= 1.1 for d in DATASETS[1:] for m in METRICS[1:])
        decisions[arm] = dict(practical_lead=bool(leads) and primary and secondary, qualifying_nonlinear_tasks=leads,
                              all_primary_within_5pct=primary, nonlinear_Jacobian_SW2_within_10pct=secondary, details=details)
    affine_static = {d: lookup[d, 'affine_forcing']['validation_oracle_mse']/lookup[d, 'static_bias']['validation_oracle_mse'] for d in DATASETS}
    conditional_tasks = [d for d in decisions['affine_forcing']['qualifying_nonlinear_tasks'] if affine_static[d] <= .95]
    conditional = decisions['affine_forcing']['practical_lead'] and bool(conditional_tasks) and all(r <= 1.05 for r in affine_static.values())
    leads = [arm for arm, decision in decisions.items() if decision['practical_lead']]
    return dict(state='BIAS_COMPLETION_LEAD' if leads else 'STOP_BIAS_COMPLETENESS_SETTING', practical_leads=leads,
                conditional_bias_evidence=conditional, conditional_qualifying_tasks=conditional_tasks,
                affine_primary_ratio_to_static=affine_static, decisions=decisions, architecture_goal_complete=False,
                scope='Engineering screen on exposed validation. A static lead is useful bias completion; conditional evidence additionally requires the registered affine/static contrast. Neither is novel ODE proof, holdout evidence or architecture-goal completion. Failure stops this fixed setting without rescue.')


def matrix_record(state, initial, dataset, label, step, name, spectrum):
    p, p0 = (s[name+'.proj.weight'].double() for s in (state, initial))
    raw = spectrum(p)
    row = dict(dataset=dataset, label=label, step=step, layer=name,
               P_frobenius_norm=float(p.norm()), P_relative_drift=float((p-p0).norm()/p0.norm()), forcing_present=name+'.forcing' in state)
    row.update({'P_'+key: value for key, value in raw.items() if key != 'sigma_quantiles'})
    channels, forcing = [], []
    singular = np.linalg.svd(p.numpy(), compute_uv=False); eigen = np.linalg.eigvals(p.numpy())
    eigen = eigen[np.lexsort((eigen.imag, eigen.real))]
    for i in range(len(singular)):
        channels.append(dict(dataset=dataset, label=label, step=step, layer=name, independently_sorted_index=i,
                             singular_value=float(singular[i]), eigen_real=float(eigen[i].real), eigen_imag=float(eigen[i].imag)))
    if row['forcing_present']:
        b = state[name+'.forcing'].double(); b0 = initial[name+'.forcing'].double()
        row.update(forcing_norm=float(b.norm()), forcing_drift_norm=float((b-b0).norm()),
                   forcing_rms=float(b.square().mean().sqrt()), forcing_mean=float(b.mean()), forcing_abs_max=float(b.abs().max()))
        forcing = [dict(dataset=dataset, label=label, step=step, layer=name, hidden_coordinate=i, forcing=float(value)) for i, value in enumerate(b)]
    for prefix, index in (('controller_first', 0), ('controller_last', 2)):
        for parameter in ('weight', 'bias'):
            key = f'{name}.t_func.{index}.{parameter}'; value = state[key].double()
            row[prefix+'_'+parameter+'_norm'] = float(value.norm())
            row[prefix+'_'+parameter+'_drift_norm'] = float((value-initial[key].double()).norm())
    return row, channels, forcing, raw


def affine_to_static(study, dataset, cfg, data, output, read, load, hashes, model_class, restore_model, jacobian):
    """Change only mode attributes on a separately restored, non-training model."""
    dim = 16 if dataset == DATASETS[2] else 2
    run = study/'runs'/cfg['experiment_id']/dataset/'ode_vector/seed000'
    checkpoint = run/'checkpoints/step_0005000.pt'; state = load(checkpoint)
    config = study/'configs/affine_forcing.json'; summary = read(study/'diagnostics'/dataset/'affine_forcing/summary.json')
    assert state['step'] == 5000 and state['identity'] == dict(dataset=dataset, model='ode_vector', seed=0)
    assert state['config_sha256'] == summary['config_sha256'] == sha(config)
    assert state['dataset_manifest_sha256'] == summary['manifest_sha256'] == sha(data/'manifest.json')
    assert summary['checkpoint_sha256'] == sha(checkpoint) and summary['targets_sha256'] == sha(data/dataset/'validation.npz')
    model = model_class(dim, 'ode_vector', **cfg['model_options']).double().eval(); restore_model(model, state)
    for parameter in model.parameters():parameter.requires_grad_(False)
    for name, value in state['runtime_buffers'].items():equal(dict(model.named_buffers())[name], value.double())
    layers = list(model.ode_layers()); assert {name for name, _ in layers} == set(LAYERS)
    assert all(layer.forcing_mode == 'conditional' for _, layer in layers)
    original_state = {name: value.clone() for name, value in model.state_dict().items()}
    original_buffers = {name: value.clone() for name, value in model.named_buffers()}
    directory = output/'affine-to-static-arrays'/dataset; directory.mkdir(parents=True, exist_ok=True)
    records, maximum_prediction, maximum_jacobian, arrays = [], 0., 0., {}
    for time in TIMES:
        path = study/'diagnostics'/dataset/'affine_forcing'/f'common-{time:.1f}.npz'
        hashes[str(path)] = sha(path); assert hashes[str(path)] == summary['arrays_sha256'][path.name]
        with np.load(path, allow_pickle=False) as archive:
            values = {key: archive[key] for key in ('x', 'prediction', 'jacobian', 'oracle', 'oracle_jacobian')}
        assert values['x'].shape == (512, dim) and all(v.dtype == np.float64 and np.isfinite(v).all() for v in values.values())
        x = torch.from_numpy(values['x']); t = torch.full((len(x), 1), time, dtype=x.dtype)
        pred, derivative = jacobian(model, x, t); pred, derivative = pred.numpy(), derivative.numpy()
        np.testing.assert_allclose(pred, values['prediction'], atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(derivative, values['jacobian'], atol=1e-9, rtol=1e-10)
        maximum_prediction = max(maximum_prediction, float(np.abs(pred-values['prediction']).max()))
        maximum_jacobian = max(maximum_jacobian, float(np.abs(derivative-values['jacobian']).max()))
        try:
            for _, layer in layers:layer.forcing_mode = 'static'
            counter, counter_j = jacobian(model, x, t); counter, counter_j = counter.numpy(), counter_j.numpy()
        finally:
            for _, layer in layers:layer.forcing_mode = 'conditional'
        assert np.isfinite(counter).all() and np.isfinite(counter_j).all()
        field = lambda prediction: float(np.mean((prediction-values['oracle'])**2))
        geometry = lambda jac: float(np.mean(np.sum((jac-values['oracle_jacobian'])**2, axis=(1, 2))))
        record = dict(dataset=dataset, time=time, anchors=len(x), original_field_mse=field(pred), staticized_field_mse=field(counter),
                      original_jacobian_error=geometry(derivative), staticized_jacobian_error=geometry(counter_j),
                      prediction_change_mse=float(np.mean((counter-pred)**2)),
                      jacobian_change_mean_squared_frobenius=float(np.mean(np.sum((counter_j-derivative)**2, axis=(1, 2)))))
        record['field_mse_change'] = record['staticized_field_mse']-record['original_field_mse']
        record['jacobian_error_change'] = record['staticized_jacobian_error']-record['original_jacobian_error']
        records.append(record)
        artifact = directory/path.name
        np.savez_compressed(artifact, x=values['x'], time=np.array(time), oracle=values['oracle'], oracle_jacobian=values['oracle_jacobian'],
                            prediction=pred, jacobian=derivative, staticized_prediction=counter, staticized_jacobian=counter_j)
        arrays[str(artifact)] = sha(artifact)
    equal(original_state, model.state_dict()); equal(original_buffers, dict(model.named_buffers()))
    aggregate = dict(dataset=dataset, time='equal_time_mean', anchors=512*len(TIMES))
    aggregate.update({key: float(np.mean([r[key] for r in records])) for key in records[0] if key not in ('dataset', 'time', 'anchors')})
    np.testing.assert_allclose([aggregate['original_field_mse'], aggregate['original_jacobian_error']],
                               [summary['field_mse'], summary['jacobian_error']], atol=1e-10, rtol=1e-10)
    receipt = dict(dataset=dataset, state='PASS', checkpoint_sha256=sha(checkpoint), config_sha256=sha(config),
                   manifest_sha256=sha(data/'manifest.json'), validation_sha256=sha(data/dataset/'validation.npz'),
                   panels=len(records), arrays_sha256=arrays, runtime_buffers_restored=True, model_tensors_unchanged=True,
                   maximum_original_prediction_replay_difference=maximum_prediction,
                   maximum_original_jacobian_replay_difference=maximum_jacobian,
                   scope='All six Db terms replaced by c*b; downstream gates and complete input Jacobians recomputed. Fixed-checkpoint dependence only, including removal of a possible mean shift. No retraining, sampling or primary uniform-time validation claim.')
    return records+[aggregate], receipt


def make_figures(output, rows, curves, flows, matrices, interventions):
    directory = output/'figures'; directory.mkdir(exist_ok=True)
    lookup = {(r['dataset'], r['label']): r for r in rows}
    def save(fig, name):
        for extension in ('png', 'pdf'):fig.savefig(directory/f'{name}.{extension}', dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), layout='constrained')
    for col, (dataset, title) in enumerate(zip(DATASETS, TITLES)):
        for row, (metric, name) in enumerate(zip(METRICS, ('Oracle MSE', 'Jacobian error', 'Validation SW2'))):
            best = min(lookup[dataset, arm][metric] for arm in ARMS[3:]); ax = axes[row, col]
            ax.bar(range(6), [lookup[dataset, arm][metric]/best for arm in ARMS], color=COLORS)
            ax.axhline(1, color='black', ls='--', lw=.8)
            ax.set_xticks(range(6), LABELS, rotation=40, ha='right', fontsize=7)
            ax.set(ylabel=name+' / best generic', title=title if row == 0 else ''); ax.grid(axis='y', alpha=.15)
    save(fig, 'affine-forcing-endpoints')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for ax, dataset, title in zip(axes, DATASETS, TITLES):
        for arm, label, color in zip(ARMS, LABELS, COLORS):
            trace = [r for r in curves if r['dataset'] == dataset and r['label'] == arm]
            ax.plot([r['step'] for r in trace], [r['validation_oracle_mse'] for r in trace], label=label, color=color)
        ax.set(title=title, xlabel='Training update', ylabel='Validation oracle MSE', yscale='log'); ax.grid(alpha=.15)
    axes[-1].legend(fontsize=7, frameon=False); save(fig, 'affine-forcing-learning')
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), layout='constrained')
    def trace(ax, records, metric, label, color, style='-'):
        steps = sorted({r['step'] for r in records})
        values = np.array([[r[metric] for r in records if r['step'] == step] for step in steps])
        ax.plot(steps, values.mean(1), label=label, color=color, ls=style)
        ax.fill_between(steps, values.min(1), values.max(1), color=color, alpha=.08)
    for col, (dataset, title) in enumerate(zip(DATASETS, TITLES)):
        for i, arm in enumerate(ARMS[:3]):
            trace(axes[0, col], [r for r in matrices if r['dataset'] == dataset and r['label'] == arm], 'P_sigma_max', LABELS[i], COLORS[i])
        selected = [r for r in flows if r['dataset'] == dataset and r['label'] == 'affine_forcing']
        for metric, label, color, style in (
            ('static_bias_rms_over_input', 'c*b', '#8359a0', ':'),
            ('conditional_bias_mean_rms_over_input', 'Mean (D-cI)b', '#bb4242', '--'),
            ('conditional_bias_centered_rms_over_input', 'Centered (D-cI)b', '#255f88', '-')):
            trace(axes[1, col], selected, metric, label, color, style)
        trace(axes[2, col], selected, 'conditional_bias_local_jacobian_frobenius_mean', 'Local diag(b) J_s', '#bb4242')
        for row, ylabel in enumerate(('Raw P largest singular value', 'Affine bias RMS / input RMS', 'Local bias Jacobian Frobenius norm')):
            axes[row, col].set(ylabel=ylabel, xlabel='Training update', title=title if row == 0 else ''); axes[row, col].grid(alpha=.15)
    for row in range(3):axes[row, -1].legend(fontsize=7, frameon=False)
    fig.suptitle('Means/ranges across six solves, not uncertainty. Raw P and local bias sensitivity do not bound the complete field.', fontsize=9)
    save(fig, 'affine-forcing-internals')
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
    for col, (dataset, title) in enumerate(zip(DATASETS, TITLES)):
        selected = [r for r in interventions if r['dataset'] == dataset and r['time'] != 'equal_time_mean']
        for row, metric in enumerate(('field_mse', 'jacobian_error')):
            for prefix, label, style in (('original_', 'Trained affine', '-'), ('staticized_', 'Same weights, Db replaced by c*b', '--')):
                axes[row, col].plot([r['time'] for r in selected], [r[prefix+metric] for r in selected], label=label, ls=style, marker='o')
            axes[row, col].set(title=title if row == 0 else '', xlabel='Time', ylabel=metric.replace('_', ' ')); axes[row, col].grid(alpha=.15)
    axes[0, -1].legend(fontsize=7, frameon=False)
    fig.suptitle('Frozen-checkpoint intervention: downstream gates recomputed; not a retrained static model or a generation evaluation', fontsize=9)
    save(fig, 'affine-forcing-staticized')


def analyze(current, output):
    current, output = current.resolve(), output.resolve(); source = current/'source'
    assert output != current and output not in current.parents
    for protected in (source, current/'output', current/'preflight'):
        assert output != protected and protected not in output.parents, 'Output would overwrite training evidence'
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
    assert len(manifest['all_files']) == 595
    assert read(current/'output/affine_forcing_queue_20261025_m1/status.json')['state'] == 'COMPLETE', 'Partial study refused'
    for name in ('summary.json', 'full-width-initialization.json', 'all-arm-initialization.json', 'affine-trainer-resume.json',
                 'historical-baseline-trainer.json', 'executed-call-counts.json', 'historical-model-trajectories.json', 'dependency-check.json'):
        assert read(current/'preflight'/name)['state'] == 'PASS'
    studies = [(suffix, current/'output'/f'affine_forcing_{suffix}_20261025_m1') for suffix in ('v1', 'product16')]
    for _, study in studies:assert read(study/'status.json')['state'] == 'COMPLETE'
    oldroot = current.parent/'sigmoid_glu_20261025_m1'; oldmanifest = verify(oldroot)
    assert len(oldmanifest['all_files']) == 587 and Path(manifest['parent_source']).resolve() == oldroot/'source'
    assert manifest['parent_manifest_sha256'] == sha(oldroot/'source-manifest.json')
    output.mkdir(parents=True, exist_ok=True); archived = output/'source'; archived.mkdir(exist_ok=True)
    replay_source = Path(__file__).resolve().with_name('audit_architecture_endpoints.py')
    assert sha(replay_source) == REPLAY_SHA
    for original in (Path(__file__).resolve(), replay_source):
        destination = archived/original.name
        if original != destination:shutil.copy2(original, destination)
        hashes[str(destination)] = sha(destination)
    sys.path.insert(0, str(source)); torch.set_num_threads(1)
    from synthetic.models import VelocityNet
    from synthetic.diagnostics import spectrum
    from synthetic.train import restore_model
    from scripts.run_architecture_study import macs
    from scripts.check_gate_geometry import jacobian

    rows, curves, initials, flows, raw_flows, matrices, spectral_channels, forcing_channels = ([] for _ in range(8))
    baselines, monitors, audits, replays, interventions, intervention_receipts = ([] for _ in range(6))
    rng_checks = 0
    for suffix, study in studies:
        data = Path('/Users/mr.z/ode-workspace-prep/data/synthetic')/('v1' if suffix == 'v1' else 'product16_v1')
        data_manifest = read(data/'manifest.json')
        for label, args in (
            ('native', [source/'scripts/analyze_architecture_study.py', '--study', study, '--source', source, '--data-root', data, '--output', output/f'{suffix}-native']),
            ('replay', [archived/replay_source.name, '--study', study, '--source', source, '--data-root', data, '--output', output/f'{suffix}-endpoint-replay.json']),
        ):
            result = subprocess.run([sys.executable, *map(str, args)], cwd=source, capture_output=True, text=True)
            (output/f'{suffix}-{label}.log').write_text(result.stdout+result.stderr)
            assert result.returncode == 0, f'{suffix} {label}: {result.stderr}'
        audit, replay = read(output/f'{suffix}-native/audit.json'), read(output/f'{suffix}-endpoint-replay.json')
        assert audit['state'] == replay['state'] == 'PASS'; audits.append(audit); replays.append(replay)
        rows.extend(dict(**row, study=str(study)) for row in audit['rows'])
        provenance = read(study/'provenance.json'); plan = provenance['plan']
        assert [arm['label'] for arm in plan['arms']] == list(ARMS)
        assert plan['seed'] == 0 and plan['steps'] == 5000 and plan['device'] == 'cpu'
        for dataset in plan['datasets']:
            for filename in ('train.npz', 'validation.npz', 'test.npz', 'probe.npz'):
                path = data/dataset/filename; hashes[str(path)] = sha(path)
                assert hashes[str(path)] == data_manifest['files'][dataset+'/'+filename]['sha256']
        hashes[str(source/plan['protocol'])] = sha(source/plan['protocol'])
        telemetry = study/'telemetry.jsonl'; hashes[str(telemetry)] = sha(telemetry)
        history = [json.loads(line) for line in telemetry.read_text().splitlines()]
        assert history and all(row['memory_available_gib'] > 0 for row in history)
        monitors.append(dict(study=study.name, rows=len(history), first=history[0], last_actual_sample=history[-1],
                             minimum_available_gib=min(row['memory_available_gib'] for row in history), terminal=read(study/'status.json')))
        oldstudy = oldroot/'output'/f'sigmoid_glu_{suffix}_20261025_m1'
        assert read(oldstudy/'status.json')['state'] == 'COMPLETE'
        assert read(oldstudy/'provenance.json')['dataset_manifest_sha256'] == provenance['dataset_manifest_sha256']
        for dataset in plan['datasets']:
            dim = 16 if dataset == DATASETS[2] else 2
            reference_rng, parent_initial, static_initial = {}, None, None
            for arm in plan['arms']:
                label, kind = arm['label'], arm['model']; cfg = read(study/'configs'/f'{label}.json')
                run = folder(study, cfg, dataset, kind); reference = cfg['backbone_initialization'][dataset]
                assert sha(reference['checkpoint']) == reference['sha256']; ref = load(Path(reference['checkpoint']))
                paths = sorted((run/'checkpoints').glob('step_*.pt'))
                assert [path.name for path in paths] == [f'step_{step:07d}.pt' for step in STEPS]
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
                if label == 'vector_parent':parent_initial = initial
                if label in MODES:
                    assert kind == 'ode_vector' and cfg['model_options']['ode_forcing_mode'] == MODES[label]
                    assert cfg['model_options']['gate_hidden'] == 32 and cfg['model_options']['orders'] == 1
                    assert cfg['model_options']['ode_solve_init'] == 'constant_paired'
                    for name, value in parent_initial['model'].items():equal(initial['model'][name], value)
                    for key in ('runtime_buffers', 'torch_rng', 'numpy_rng', 'python_rng'):equal(initial[key], parent_initial[key])
                    extra = set(initial['model'])-set(parent_initial['model'])
                    assert extra == (set() if label == 'vector_parent' else {name+'.forcing' for name in LAYERS})
                    for name in extra:assert initial['model'][name].shape == (64,) and torch.count_nonzero(initial['model'][name]) == 0
                    for name in LAYERS:
                        for parameter in ('weight', 'bias'):assert torch.count_nonzero(initial['model'][f'{name}.t_func.2.{parameter}']) == 0
                if label == 'static_bias':static_initial = initial
                if label == 'affine_forcing':equal(initial['model'], static_initial['model'])
                row = next(row for row in rows if row['dataset'] == dataset and row['label'] == label)
                assert model.parameter_count() == row['parameters'] and macs(model, dim) == row['forward_macs']
                if label in MODES:
                    parent_row = next(row for row in rows if row['dataset'] == dataset and row['label'] == 'vector_parent')
                    assert row['forward_macs'] == parent_row['forward_macs']
                    assert row['parameters']-parent_row['parameters'] == (0 if label == 'vector_parent' else 384)
                initials.append(dict(dataset=dataset, label=label, reference=reference, native_ffn_entries=len(ffn),
                                     non_ffn_tensors=len(common), native_ffn_rng_unchanged=True, ODE_common_state_paired=label in MODES))
                oldcfg = read(oldstudy/'configs'/f'{label}.json') if label in ARMS[3:] else None
                if oldcfg:
                    for key in ('steps', 'batch_size', 'learning_rate', 'weight_decay', 'grad_clip', 'eval_interval', 'sample_interval',
                                'checkpoint_interval', 'sample_count', 'sliced_directions', 'heun_steps', 'sampling_split', 'model_options', 'backbone_initialization'):
                        assert cfg[key] == oldcfg[key], (label, key)
                raw_checkpoints = {}
                for step, cp in saved.items():
                    if label == 'vector_parent':reference_rng[step] = cp['data_rng']
                    else:equal(cp['data_rng'], reference_rng[step]); rng_checks += 1
                    if oldcfg:
                        oldcp = load(folder(oldstudy, oldcfg, dataset, kind)/'checkpoints'/f'step_{step:07d}.pt')
                        for key in STATE_KEYS:equal(cp[key], oldcp[key])
                        baselines.append(dict(dataset=dataset, label=label, step=step, state='BIT_IDENTICAL', compared_keys=list(STATE_KEYS)))
                    if label in MODES:
                        for name in LAYERS:
                            matrix, channels, forcing, raw = matrix_record(cp['model'], initial['model'], dataset, label, step, name, spectrum)
                            matrices.append(matrix); spectral_channels.extend(channels); forcing_channels.extend(forcing)
                            raw_checkpoints[step, name] = raw
                metric_path = run/'metrics.jsonl'; hashes[str(metric_path)] = sha(metric_path)
                trace = [json.loads(line) for line in metric_path.read_text().splitlines()]
                assert [metric['step'] for metric in trace] == list(range(0, 5001, 500))
                for metric in trace:
                    step = metric['step']
                    curves.append(dict(dataset=dataset, label=label, step=step, train_pair_mse=metric['train_pair_mse'],
                                       validation_pair_mse=metric['validation_pair_mse'], validation_oracle_mse=metric['validation_oracle_mse'],
                                       wall_seconds=metric['wall_seconds'], grad_norm_before_clip=metric['grad_norm_before_clip'],
                                       validation_sw2=metric.get('sample', {}).get('sliced_w2')))
                    if label not in MODES:assert not metric['ode_layers']; continue
                    entries = metric['ode_layers']; assert len(entries) == 6 and {entry['layer'] for entry in entries} == set(LAYERS)
                    for entry in entries:
                        name, activation, raw = entry['layer'], entry['activation'], entry['raw']
                        assert (entry['gate_source'], entry['rate_placement'], entry['solve_return'], entry['operator_mode']) == ('input', 'after_projection', 'map', 'multiplicative')
                        equal(raw, entry['effective'])
                        if step in saved:
                            for key in raw:np.testing.assert_allclose(raw[key], raw_checkpoints[step, name][key], atol=1e-12, rtol=1e-11)
                        if label == 'vector_parent':assert 'forcing_mode' not in activation
                        else:
                            assert activation['forcing_mode'] == MODES[label]
                            for key in ('forcing_parameter_norm', 'forcing_rms_over_input', 'static_bias_rms_over_input', 'conditional_bias_rms_over_input',
                                        'conditional_bias_mean_rms_over_input', 'conditional_bias_centered_rms_over_input', 'conditional_bias_local_jacobian_frobenius_mean'):
                                assert np.isfinite(activation[key]) and activation[key] >= 0
                            if step in saved:
                                np.testing.assert_allclose(activation['forcing_parameter_norm'], float(saved[step]['model'][name+'.forcing'].norm()), atol=1e-9, rtol=1e-6)
                            np.testing.assert_allclose(activation['conditional_bias_rms_over_input']**2,
                                activation['conditional_bias_mean_rms_over_input']**2+activation['conditional_bias_centered_rms_over_input']**2,
                                atol=1e-9, rtol=2e-5)
                            if label == 'static_bias':
                                assert activation['conditional_bias_rms_over_input'] == activation['conditional_bias_local_jacobian_frobenius_mean'] == 0
                        raw_flows.append(dict(dataset=dataset, label=label, step=step, **entry))
                        record = dict(dataset=dataset, label=label, step=step, layer=name, forcing_present=label != 'vector_parent', forcing_mode=MODES[label])
                        record.update({'P_'+key: value for key, value in raw.items() if key != 'sigma_quantiles'}); record.update(activation)
                        flows.append(record)
                if label == 'affine_forcing':
                    values, receipt = affine_to_static(study, dataset, cfg, data, output, read, load, hashes, VelocityNet, restore_model, jacobian)
                    interventions.extend(values); intervention_receipts.append(receipt)

    checkpoint_count = sum(a['checkpoints_checked'] for a in audits); array_count = sum(a['arrays_checked'] for a in audits)
    replay_count = sum(a['final_checkpoints_reloaded'] for a in replays)
    assert (checkpoint_count, array_count, replay_count) == (108, 90, 18)
    assert len(rows) == len(initials) == 18 and len(curves) == 198 and len(baselines) == 54 and rng_checks == 90
    assert len(flows) == len(raw_flows) == 594 and len(matrices) == 324 and len(spectral_channels) == 20736 and len(forcing_channels) == 13824
    assert len(interventions) == 18 and len(intervention_receipts) == 3 and sum(r['panels'] for r in intervention_receipts) == 15
    rows.sort(key=lambda row: (DATASETS.index(row['dataset']), ARMS.index(row['label'])))
    for name, records in (('comparison', rows), ('learning-curves', curves), ('forcing-trajectories', flows), ('P-forcing-checkpoints', matrices),
                          ('P-spectral-channels', spectral_channels), ('forcing-coordinate-trajectories', forcing_channels), ('affine-to-static', interventions)):
        csv_write(output/(name+'.csv'), records)
    write(output/'raw-forcing-trajectories.json', raw_flows)
    write(output/'affine-to-static-audit.json', dict(state='PASS', rows=intervention_receipts, total_panels=15))
    write(output/'initialization-audit.json', dict(state='PASS', rows=initials, snapshot_data_rng_comparisons=rng_checks,
          full_width_preflight_sha256=sha(current/'preflight/full-width-initialization.json'),
          scope='All ODE common tensors/buffers/RNG match. Static/affine additional biases are zero; their full initial parameter responses are paired by preflight. Parent has no bias columns.'))
    write(output/'baseline-trajectory-replay.json', dict(state='PASS', rows=baselines,
          scope='All three generic controls reproduce six Stage22 saved states on three tasks exactly. Compatibility reruns, not independent repetitions. No Stage17 parent bit-equality is inferred.'))
    write(output/'monitoring-evidence.json', monitors)
    decision = registered_screen(rows); write(output/'screen-decision.json', decision)
    make_figures(output, rows, curves, flows, matrices, interventions); verify(current)
    write(output/'provenance.json', dict(state='PASS', analysis_sha256=sha(__file__), input_sha256=hashes,
          source_files_verified=len(manifest['all_files']), checkpoints_native_audited=checkpoint_count, arrays_native_audited=array_count,
          endpoint_replays=replay_count, intervention_final_checkpoints_restored=3, intervention_panels=15,
          maximum_forward_replay_difference=max(a['maximum_forward_difference'] for a in replays),
          maximum_jacobian_replay_difference=max(a['maximum_derivative_difference'] for a in replays),
          scope='Exposed validation only. Raw P spectra exclude bias forcing and clock derivatives; independently sorted spectra do not track modes. Conditional means and centered variation are distinct. Frozen affine-to-static intervention is not retraining or generation evaluation. Wall time includes concurrent evaluation/training, not isolated latency. No holdout or architecture-goal completion.'))
    summary = dict(state='PASS', rows=len(rows), curve_rows=len(curves), forcing_stat_rows=len(flows), P_forcing_checkpoint_rows=len(matrices),
                   P_spectral_channel_rows=len(spectral_channels), forcing_coordinate_rows=len(forcing_channels), baseline_saved_states_equal=len(baselines),
                   snapshot_rng_comparisons=rng_checks, checkpoints_native_audited=checkpoint_count, arrays_native_audited=array_count,
                   endpoint_replays=replay_count, intervention_panels=15, practical_leads=decision['practical_leads'],
                   conditional_bias_evidence=decision['conditional_bias_evidence'], architecture_goal_complete=False)
    write(output/'summary.json', summary); return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(analyze(args.current, args.output), indent=2))
