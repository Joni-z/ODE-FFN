"""Audit the completed fixed low-rank state-allocation screen; never train."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

DATASETS = ('gaussian_aniso_2d', 'gmm_ring8_2d', 'rotated_product_bimodal_16d')
TITLES = ('Gaussian2D', 'Ring8', 'Product16')
ARMS = ('vector_parent', 'lowrank_budget', 'lowrank_narrow', 'lowrank_augmented', 'swiglu', 'silu128')
HISTORICAL = ('swiglu_budget', 'sigmoid_glu')
GENERIC = ARMS[4:]+HISTORICAL
LABELS = ('ODE parent', 'Factor64/q32', 'Narrow64/q21', 'Expanded128/q21', 'SwiGLU128', 'SiLU128', 'SwiGLU96 (historical)', 'GLU128 (historical)')
COLORS = ('#a99550', '#8157a3', '#548d72', '#bb4242', '#255f88', '#669cb7', '#a8c2d0', '#8eaa80')
LAYERS = tuple(f'blocks.{i}.ffn.ode{j}' for i in range(3) for j in (1, 2))
STEPS, TIMES = tuple(range(0, 5001, 1000)), (.1, .3, .5, .7, .9)
METRICS = ('validation_oracle_mse', 'jacobian_error', 'validation_sw2')
STATE_KEYS = ('model', 'optimizer', 'runtime_buffers', 'data_rng', 'torch_rng', 'numpy_rng', 'python_rng')
REPLAY_SHA = '142bbe47e3ddf7f25befb1f0eda3b67d187122505805fcf94d4a5c62477640b7'
CONFIG_KEYS = ('steps', 'batch_size', 'learning_rate', 'weight_decay', 'grad_clip', 'eval_interval', 'sample_interval',
               'checkpoint_interval', 'sample_count', 'sliced_directions', 'heun_steps', 'sampling_split', 'model_options', 'backbone_initialization')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def csv_write(path, rows):
    assert rows
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
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
    for arm in ARMS[1:4]:
        details, tasks = {}, []
        for dataset in DATASETS:
            row = lookup[dataset, arm]
            best = {m: min(lookup[dataset, generic][m] for generic in GENERIC) for m in METRICS}
            assert all(v > 0 for v in best.values())
            ratios = {m: row[m]/best[m] for m in METRICS}
            parent = row[METRICS[0]]/lookup[dataset, 'vector_parent'][METRICS[0]]
            if dataset in DATASETS[1:] and ratios[METRICS[0]] <= .9 and parent <= .9:tasks.append(dataset)
            details[dataset] = dict(ratio_to_best_generic=ratios, primary_ratio_to_parent=parent,
                parameter_ratio_to_Swi128=row['parameters']/lookup[dataset, 'swiglu']['parameters'],
                MAC_ratio_to_Swi128=row['forward_macs']/lookup[dataset, 'swiglu']['forward_macs'])
        primary = all(details[d]['ratio_to_best_generic'][METRICS[0]] <= 1.05 for d in DATASETS)
        secondary = all(details[d]['ratio_to_best_generic'][m] <= 1.1 for d in DATASETS[1:] for m in METRICS[1:])
        savings = all(details[d][m] <= .8 for d in DATASETS for m in ('parameter_ratio_to_Swi128', 'MAC_ratio_to_Swi128'))
        passed = primary and secondary and (savings if arm == 'lowrank_narrow' else bool(tasks))
        decisions[arm] = dict(practical_lead=passed, qualifying_nonlinear_tasks=tasks, all_primary_within_5pct=primary,
            nonlinear_Jacobian_SW2_within_10pct=secondary, at_least_20pct_full_model_cost_savings=savings, details=details)
    comparisons = {d: {other: lookup[d, 'lowrank_augmented'][METRICS[0]]/lookup[d, other][METRICS[0]]
                       for other in ('lowrank_budget', 'lowrank_narrow')} for d in DATASETS}
    augmentation_tasks = [d for d in decisions['lowrank_augmented']['qualifying_nonlinear_tasks']
                          if all(r <= .9 for r in comparisons[d].values())]
    allocation = decisions['lowrank_augmented']['practical_lead'] and bool(augmentation_tasks) and all(
        r <= 1.05 for ratios in comparisons.values() for r in ratios.values())
    leads = [arm for arm, value in decisions.items() if value['practical_lead']]
    return dict(state='PHASE_A_PROVISIONAL_LEAD' if leads else 'STOP_FIXED_STATE_ALLOCATION', practical_leads=leads,
        augmentation_allocation_evidence=allocation, augmentation_qualifying_tasks=augmentation_tasks,
        augmented_primary_ratios_to_factor_controls=comparisons, decisions=decisions, architecture_goal_complete=False,
        scope='Exposed validation screen. All useful controls are retained. A narrow lead needs SwiGLU84; every lead needs separately registered timing and fresh evaluation/transfer. This is not augmentation-only causality, ODE novelty, or goal completion.')


def matrix_record(layer, initial, dataset, label, step, name, duration, spectrum):
    lowrank = label != 'vector_parent'
    p = layer.full_operator().detach() if lowrank else layer.proj.weight.detach()
    p0 = initial['P']; raw = spectrum(p)
    row = dict(dataset=dataset, label=label, step=step, layer=name, state_width=p.shape[0],
        P_frobenius_norm=float(p.double().norm()), P_relative_drift=float((p.double()-p0.double()).norm()/p0.double().norm()))
    row.update({'P_'+k: v for k, v in raw.items() if k != 'sigma_quantiles'})
    if lowrank:
        u, v = layer.left.detach().double(), layer.right.detach().double()
        row.update(rank=layer.rank, role=layer.role, left_norm=float(u.norm()), right_norm=float(v.norm()),
            left_drift_norm=float((u-initial['left'].double()).norm()), right_drift_norm=float((v-initial['right'].double()).norm()),
            factor_gram_imbalance=float((u.T@u-v@v.T).norm()), factor_norm_product=float(u.norm()*v.norm()),
            extra_left_rows_norm=float(u[layer.width:].norm()), extra_right_columns_norm=float(v[:, layer.width:].norm()),
            extra_controller_input_columns_norm=float(layer.t_func[0].weight.detach()[:, layer.width:].double().norm()))
        compressed = layer.compressed_operator(duration).detach()
        # Independently embed both factors, retaining all padded coordinates.
        full_left = torch.nn.functional.pad(layer.left.detach(), (0, 0, 0, layer.state_width-layer.output_width))
        full_right = torch.nn.functional.pad(layer.right.detach(), (0, layer.state_width-layer.input_width))
        full_duration = torch.nn.functional.pad(duration, (0, layer.state_width-layer.output_width), value=1.)
        embedded_compressed = full_right@(full_duration[:, None]*full_left)
        torch.testing.assert_close(compressed, embedded_compressed, atol=1e-6, rtol=1e-5)
        # This is a probe-mean frozen generator, not the live Jacobian.
        row.update({'mean_clock_VDU_'+k: val for k, val in spectrum(compressed).items() if k != 'sigma_quantiles'})
    for index, tag in ((0, 'controller_first'), (2, 'controller_last')):
        for parameter in ('weight', 'bias'):
            val = getattr(layer.t_func[index], parameter).detach().double()
            row[tag+'_'+parameter+'_norm'] = float(val.norm())
            row[tag+'_'+parameter+'_drift_norm'] = float((val-initial[f't_func.{index}.{parameter}'].double()).norm())
    singular = np.linalg.svd(p.double().numpy(), compute_uv=False); eigen = np.linalg.eigvals(p.double().numpy())
    eigen = eigen[np.lexsort((eigen.imag, eigen.real))]
    channels = [dict(dataset=dataset, label=label, step=step, layer=name, state_width=len(singular), independently_sorted_index=i,
                     singular_value=float(singular[i]), eigen_real=float(eigen[i].real), eigen_imag=float(eigen[i].imag)) for i in range(len(singular))]
    return row, channels, raw


def historical_guard(study, source, data, dataset, arm, fresh_study, read, load, hashes):
    """Cold-check two already audited generic histories; no independent-repeat claim."""
    cfgpath = study/'configs'/f'{arm}.json'; cfg = read(cfgpath)
    plan = read(study/'provenance.json')['plan']; kind = next(a['model'] for a in plan['arms'] if a['label'] == arm)
    run = study/'runs'/cfg['experiment_id']/dataset/kind/'seed000'
    assert (run/'COMPLETE').read_text().strip() == '5000' and sha(run/'config.json') == sha(cfgpath)
    hashes[str(run/'COMPLETE')] = sha(run/'COMPLETE'); hashes[str(run/'config.json')] = sha(run/'config.json')
    cps = []
    for step in STEPS:
        path = run/'checkpoints'/f'step_{step:07d}.pt'; cp = load(path)
        assert cp['step'] == step and cp['identity'] == dict(dataset=dataset, model=kind, seed=0)
        assert cp['config_sha256'] == sha(cfgpath) and cp['dataset_manifest_sha256'] == sha(data/'manifest.json')
        cps.append(dict(step=step, checkpoint=str(path), sha256=sha(path)))
    summary = read(study/'diagnostics'/dataset/arm/'summary.json')
    assert summary['checkpoint_sha256'] == cps[-1]['sha256'] and summary['config_sha256'] == sha(cfgpath)
    assert summary['manifest_sha256'] == sha(data/'manifest.json') and summary['targets_sha256'] == sha(data/dataset/'validation.npz')
    errors = []
    assert set(summary['arrays_sha256']) == {f'common-{t:.1f}.npz' for t in TIMES}
    for name, digest in summary['arrays_sha256'].items():
        path = study/'diagnostics'/dataset/arm/name; hashes[str(path)] = sha(path); assert sha(path) == digest
        with np.load(path, allow_pickle=False) as old, np.load(fresh_study/'diagnostics'/dataset/'swiglu'/name, allow_pickle=False) as new:
            assert all(np.isfinite(old[k]).all() for k in old.files)
            for key in ('x', 'oracle', 'oracle_jacobian'):np.testing.assert_array_equal(old[key], new[key])
            errors.append([np.mean((old['prediction']-old['oracle'])**2), np.mean(np.sum((old['jacobian']-old['oracle_jacobian'])**2, axis=(1, 2)))])
    result = next(r for r in read(study/'results.json') if r['dataset'] == dataset and r['label'] == arm)
    np.testing.assert_allclose(np.mean(errors, axis=0), [result['field_mse'], result['jacobian_error']], atol=1e-12, rtol=1e-12)
    metricpath = run/'metrics.jsonl'; hashes[str(metricpath)] = sha(metricpath)
    final = json.loads(metricpath.read_text().splitlines()[-1]); assert final['step'] == 5000
    np.testing.assert_allclose([result['validation_oracle_mse'], result['validation_sw2']], [final['validation_oracle_mse'], final['sample']['sliced_w2']], rtol=0, atol=0)
    return dict(**result, study=str(study), evidence_role='historical_generic_guard'), dict(dataset=dataset, label=arm, state='PASS',
        checkpoints=cps, arrays_checked=5, source=str(source), scope='Previously trained Stage23 reference. Hash/identity/data/metric checks; no new training or independent repeat.')


def make_figures(output, rows, curves, flows, matrices):
    directory = output/'figures'; directory.mkdir(exist_ok=True)
    lookup = {(r['dataset'], r['label']): r for r in rows}
    def save(fig, name):
        for extension in ('png', 'pdf'):fig.savefig(directory/f'{name}.{extension}', dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), layout='constrained')
    for col, (dataset, title) in enumerate(zip(DATASETS, TITLES)):
        for row, metric in enumerate(METRICS):
            ax = axes[row, col]; best = min(lookup[dataset, arm][metric] for arm in GENERIC)
            bars = ax.bar(range(8), [lookup[dataset, arm][metric]/best for arm in ARMS+HISTORICAL], color=COLORS)
            for bar in bars[-2:]:bar.set_hatch('//')
            ax.axhline(1, color='black', ls='--', lw=.8)
            ax.set_xticks(range(8), LABELS, rotation=45, ha='right', fontsize=7)
            ax.set(ylabel=metric.replace('_', ' ')+' / generic frontier', title=title if row == 0 else ''); ax.grid(axis='y', alpha=.15)
    fig.suptitle('Hatched controls are historical Stage23 runs, not additional repetitions', fontsize=10); save(fig, 'augmented-state-endpoints')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for ax, dataset, title in zip(axes, DATASETS, TITLES):
        for arm, label, color in zip(ARMS, LABELS, COLORS):
            trace = [r for r in curves if r['dataset'] == dataset and r['label'] == arm]
            ax.plot([r['step'] for r in trace], [r['validation_oracle_mse'] for r in trace], label=label, color=color)
        ax.set(title=title, xlabel='Training update', ylabel='Validation oracle MSE', yscale='log'); ax.grid(alpha=.15)
    axes[-1].legend(fontsize=7, frameon=False); save(fig, 'augmented-state-learning')
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), layout='constrained')
    for col, (dataset, title) in enumerate(zip(DATASETS, TITLES)):
        ax = axes[0, col]
        for i, arm in enumerate(ARMS+HISTORICAL):
            row = lookup[dataset, arm]
            ax.scatter(row['forward_macs']/1000, row['validation_oracle_mse'], color=COLORS[i], marker='x' if arm in HISTORICAL else 'o')
            ax.annotate(LABELS[i].replace(' (historical)', '*'), (row['forward_macs']/1000, row['validation_oracle_mse']), fontsize=6, xytext=(3, 3), textcoords='offset points')
        ax.set(title=title, xlabel='Dense MACs / 1000', ylabel='Oracle MSE'); ax.grid(alpha=.15)
        for metric, label, role, color in (('extra_output_rms_over_input', 'Lift extra output', 'lift', '#bb4242'),
                                          ('extra_input_rms_over_input', 'Project extra input', 'project', '#255f88')):
            trace = [r for r in flows if r['dataset'] == dataset and r['label'] == 'lowrank_augmented' and r['role'] == role]
            steps = sorted({r['step'] for r in trace})
            axes[1, col].plot(steps, [np.mean([r[metric] for r in trace if r['step'] == step]) for step in steps], label=label, color=color)
        trace = [r for r in matrices if r['dataset'] == dataset and r['label'] == 'lowrank_augmented' and r['role'] == 'project']
        for metric, label, color in (('extra_right_columns_norm', 'V2 extra columns', '#bb4242'), ('extra_controller_input_columns_norm', 'Gate2 extra input columns', '#255f88')):
            steps = sorted({r['step'] for r in trace})
            axes[2, col].plot(steps, [np.mean([r[metric] for r in trace if r['step'] == step]) for step in steps], label=label, color=color)
        axes[1, col].set(ylabel='Extra-feature RMS / native-input RMS', xlabel='Training update')
        axes[2, col].set(ylabel='Receiving weight Frobenius norm', xlabel='Training update')
    for row in (1, 2):axes[row, -1].legend(fontsize=7, frameon=False)
    fig.suptitle('Extra paths: means across three blocks, not uncertainty or causal importance. Dense MACs are not latency.', fontsize=9)
    save(fig, 'augmented-state-cost-and-extra-paths')


def analyze(current, output):
    current, output = current.resolve(), output.resolve(); source = current/'source'
    assert output != current and output not in current.parents
    for protected in (source, current/'output', current/'preflight'):
        assert output != protected and protected not in output.parents, 'Output would overwrite training evidence'
    hashes = {}
    def read(path):hashes[str(path)] = sha(path); return json.loads(path.read_text())
    def load(path):hashes[str(path)] = sha(path); return torch.load(path, map_location='cpu', weights_only=False)
    def verify(root):
        manifest = read(root/'source-manifest.json')
        for name, digest in manifest['all_files'].items():assert sha(root/'source'/name) == digest, name
        return manifest
    def folder(study, cfg, dataset, kind):return study/'runs'/cfg['experiment_id']/dataset/kind/'seed000'
    manifest = verify(current)
    assert len(manifest['all_files']) == 604
    assert read(current/'output/augmented_state_queue_20261025_m1/status.json')['state'] == 'COMPLETE', 'Partial study refused'
    for name in ('summary.json', 'full-width-initialization.json', 'all-arm-initialization.json', 'lowrank-trainer-resume.json',
                 'historical-baseline-trainer.json', 'executed-call-counts.json', 'historical-model-trajectories.json',
                 'historical-frontier-bindings.json', 'dependency-check.json'):
        assert read(current/'preflight'/name)['state'] == 'PASS', name
    oldroot = current.parent/'affine_forcing_20261025_m1'; oldmanifest = verify(oldroot)
    assert len(oldmanifest['all_files']) == 595 and Path(manifest['parent_source']).resolve() == oldroot/'source'
    assert manifest['parent_manifest_sha256'] == sha(oldroot/'source-manifest.json')
    siluroot = current.parent/'diagonal_flow_20261024_m1'; verify(siluroot)
    studies = [(suffix, current/'output'/f'augmented_state_{suffix}_20261025_m1') for suffix in ('v1', 'product16')]
    for _, study in studies:assert read(study/'status.json')['state'] == 'COMPLETE'
    output.mkdir(parents=True, exist_ok=True); archived = output/'source'; archived.mkdir(exist_ok=True)
    replay_source = Path(__file__).resolve().with_name('audit_architecture_endpoints.py'); assert sha(replay_source) == REPLAY_SHA
    for original in (Path(__file__).resolve(), replay_source):
        destination = archived/original.name
        if original != destination:shutil.copy2(original, destination)
        hashes[str(destination)] = sha(destination)
    sys.path.insert(0, str(source)); torch.set_num_threads(1)
    from synthetic.models import VelocityNet
    from synthetic.diagnostics import spectrum
    from synthetic.train import restore_model
    from scripts.run_architecture_study import macs
    for module in ('synthetic.models', 'synthetic.diagnostics', 'synthetic.train', 'synthetic.augmented', 'scripts.run_architecture_study'):
        assert Path(sys.modules[module].__file__).resolve().is_relative_to(source), module
    rows, curves, initials, flows, raw_flows, matrices, channels, baselines, monitors, audits, replays, historical = ([] for _ in range(12))
    rng_checks = 0
    for suffix, study in studies:
        data = Path('/Users/mr.z/ode-workspace-prep/data/synthetic')/('v1' if suffix == 'v1' else 'product16_v1')
        data_manifest = read(data/'manifest.json')
        for label, args in (
            ('native', [source/'scripts/analyze_architecture_study.py', '--study', study, '--source', source, '--data-root', data, '--output', output/f'{suffix}-native']),
            ('replay', [archived/replay_source.name, '--study', study, '--source', source, '--data-root', data, '--output', output/f'{suffix}-endpoint-replay.json'])):
            result = subprocess.run([sys.executable, *map(str, args)], cwd=source, capture_output=True, text=True)
            (output/f'{suffix}-{label}.log').write_text(result.stdout+result.stderr)
            assert result.returncode == 0, f'{suffix} {label}: {result.stderr}'
        audit, replay = read(output/f'{suffix}-native/audit.json'), read(output/f'{suffix}-endpoint-replay.json')
        assert audit['state'] == replay['state'] == 'PASS'; audits.append(audit); replays.append(replay)
        rows.extend(dict(**r, study=str(study), evidence_role='fresh_phase_A') for r in audit['rows'])
        provenance = read(study/'provenance.json'); plan = provenance['plan']
        assert [a['label'] for a in plan['arms']] == list(ARMS) and plan['seed'] == 0 and plan['steps'] == 5000 and plan['device'] == 'cpu'
        hashes[str(source/plan['protocol'])] = sha(source/plan['protocol'])
        telemetry = study/'telemetry.jsonl'; hashes[str(telemetry)] = sha(telemetry)
        history = [json.loads(line) for line in telemetry.read_text().splitlines()]; assert history and all(r['memory_available_gib'] > 0 for r in history)
        monitors.append(dict(study=study.name, rows=len(history), first=history[0], last_actual_sample=history[-1],
            minimum_available_gib=min(r['memory_available_gib'] for r in history), terminal=read(study/'status.json')))
        oldstudy = oldroot/'output'/f'affine_forcing_{suffix}_20261025_m1'
        silustudy = siluroot/'output'/f'diagonal_flow_{suffix}_20261024_m1'
        for historic_study, historic_source in ((oldstudy, oldroot/'source'), (silustudy, siluroot/'source')):
            assert read(historic_study/'status.json')['state'] == 'COMPLETE'
            historic_prov = read(historic_study/'provenance.json')
            assert historic_prov['dataset_manifest_sha256'] == provenance['dataset_manifest_sha256']
            for name, digest in historic_prov['source_sha256'].items():assert sha(historic_source/name) == digest
        for dataset in plan['datasets']:
            dim = 16 if dataset == DATASETS[2] else 2
            for filename in ('train.npz', 'validation.npz', 'test.npz', 'probe.npz'):
                path = data/dataset/filename; hashes[str(path)] = sha(path)
                assert sha(path) == data_manifest['files'][dataset+'/'+filename]['sha256']
            with np.load(data/dataset/'probe.npz', allow_pickle=False) as probe:
                probe_x = torch.tensor(probe['xt'][:256], dtype=torch.float32); probe_t = torch.tensor(probe['t'][:256], dtype=torch.float32)
            for arm in HISTORICAL:
                historical_row, receipt = historical_guard(oldstudy, oldroot/'source', data, dataset, arm, study, read, load, hashes)
                rows.append(historical_row); historical.append(receipt)
            reference_rng = {}; narrow_initial_outputs = None
            for arm in plan['arms']:
                label, kind = arm['label'], arm['model']; cfg = read(study/'configs'/f'{label}.json')
                run = folder(study, cfg, dataset, kind); reference = cfg['backbone_initialization'][dataset]
                assert sha(reference['checkpoint']) == reference['sha256']; ref = load(Path(reference['checkpoint']))
                paths = sorted((run/'checkpoints').glob('step_*.pt')); assert [p.name for p in paths] == [f'step_{s:07d}.pt' for s in STEPS]
                saved = {step: load(path) for step, path in zip(STEPS, paths)}; initial = saved[0]
                torch.manual_seed(0); model = VelocityNet(dim, kind, **cfg['model_options']).eval()
                native_rng, native = torch.get_rng_state().clone(), model.state_dict(); common, ffn = [], []
                for name, value in initial['model'].items():
                    if '.ffn.' in name:equal(value, native[name]); ffn.append(name)
                    else:equal(value, ref['model'][name]); common.append(name)
                equal(initial['torch_rng'], native_rng); equal(initial['runtime_buffers'], ref['runtime_buffers'])
                receipt = read(run/'provenance.json')['backbone_initialization']
                assert receipt['source_sha256'] == reference['sha256'] and receipt['copied_state_keys'] == sorted(common)
                assert receipt['copied_runtime_buffer_keys'] == sorted(initial['runtime_buffers'])
                row = next(r for r in rows if r['dataset'] == dataset and r['label'] == label)
                assert model.parameter_count() == row['parameters'] and macs(model, dim) == row['forward_macs']
                restore_model(model, initial)
                h = torch.linspace(-2, 2, 9*64).reshape(9, 64)
                if label == 'lowrank_narrow':
                    with torch.no_grad():narrow_initial_outputs = [block.ffn(h).clone() for block in model.blocks]
                pair_difference = None
                if label == 'lowrank_augmented':
                    # Finite native FFN map check independent of zero-head masking; preflight binds the full derivative checks.
                    with torch.no_grad():
                        differences = [(block.ffn(h)-expected).abs().max().item() for block, expected in zip(model.blocks, narrow_initial_outputs)]
                    pair_difference = max(differences); assert pair_difference < 3e-6
                initials.append(dict(dataset=dataset, label=label, reference=reference, native_ffn_entries=len(ffn),
                    non_ffn_tensors=len(common), native_ffn_rng_unchanged=True, expanded_narrow_FFN_max_difference=pair_difference))
                oldcfg = None
                if label in ('swiglu', 'silu128'):
                    baseline_study, oldlabel = (oldstudy, 'swiglu') if label == 'swiglu' else (silustudy, 'ffn_silu')
                    oldcfg = read(baseline_study/'configs'/f'{oldlabel}.json')
                    for key in CONFIG_KEYS:assert cfg[key] == oldcfg[key], (label, key)
                layers = list(model.ode_layers())+list(model.lowrank_layers())
                assert {name for name, _ in layers} == (set(LAYERS) if label in ARMS[:4] else set())
                initial_layers = {}
                for name, layer in layers:
                    initial_layers[name] = {key: value.detach().clone() for key, value in layer.state_dict().items()}
                    initial_layers[name]['P'] = (layer.proj.weight if label == 'vector_parent' else layer.full_operator()).detach().clone()
                raw_by_checkpoint, probe_by_checkpoint = {}, {}
                for step, cp in saved.items():
                    if label == 'vector_parent':reference_rng[step] = cp['data_rng']
                    else:equal(cp['data_rng'], reference_rng[step]); rng_checks += 1
                    if oldcfg:
                        oldpath = folder(baseline_study, oldcfg, dataset, kind)/'checkpoints'/f'step_{step:07d}.pt'; oldcp = load(oldpath)
                        for key in STATE_KEYS:equal(cp[key], oldcp[key])
                        baselines.append(dict(dataset=dataset, label=label, step=step, historical_checkpoint=str(oldpath), state='BIT_IDENTICAL', compared_keys=list(STATE_KEYS)))
                    if not layers:continue
                    restore_model(model, cp); captured = {}; handles = []
                    for name, layer in layers:
                        handles.append(layer.register_forward_pre_hook(lambda module, args, name=name: captured.__setitem__(name, args[0].detach().clone())))
                    try:
                        with torch.no_grad():model(probe_x, probe_t)
                    finally:
                        for handle in handles:handle.remove()
                    assert set(captured) == set(LAYERS)
                    for name, layer in layers:
                        with torch.no_grad():
                            duration = layer.step_size(captured[name]).mean(0)
                            activation = layer.activation_stats(captured[name])
                            compressed = spectrum(layer.compressed_operator(duration)) if label != 'vector_parent' else None
                        matrix, spectral, raw = matrix_record(layer, initial_layers[name], dataset, label, step, name, duration, spectrum)
                        matrices.append(matrix); channels.extend(spectral); raw_by_checkpoint[step, name] = raw
                        probe_by_checkpoint[step, name] = (activation, compressed)
                metricpath = run/'metrics.jsonl'; hashes[str(metricpath)] = sha(metricpath)
                trace = [json.loads(line) for line in metricpath.read_text().splitlines()]
                assert [m['step'] for m in trace] == list(range(0, 5001, 500))
                for metric in trace:
                    step = metric['step']; curves.append(dict(dataset=dataset, label=label, step=step, train_pair_mse=metric['train_pair_mse'],
                        validation_pair_mse=metric['validation_pair_mse'], validation_oracle_mse=metric['validation_oracle_mse'],
                        wall_seconds=metric['wall_seconds'], grad_norm_before_clip=metric['grad_norm_before_clip'], validation_sw2=metric.get('sample', {}).get('sliced_w2')))
                    if not layers:assert not metric['ode_layers']; continue
                    assert {e['layer'] for e in metric['ode_layers']} == set(LAYERS)
                    for entry in metric['ode_layers']:
                        name, activation, raw = entry['layer'], entry['activation'], entry['raw']
                        if step in saved:
                            for key in raw:np.testing.assert_allclose(raw[key], raw_by_checkpoint[step, name][key], atol=1e-12, rtol=1e-11)
                            measured_activation, measured_compressed = probe_by_checkpoint[step, name]
                            for key, value in measured_activation.items():
                                if isinstance(value, (float, int)):np.testing.assert_allclose(activation[key], value, atol=1e-7, rtol=2e-5)
                                else:assert activation[key] == value
                            if measured_compressed:
                                for key, value in measured_compressed.items():
                                    np.testing.assert_allclose(entry['compressed_at_probe_mean_duration'][key], value, atol=1e-10, rtol=1e-9)
                        if label != 'vector_parent':
                            assert entry['operator_type'] == 'pruned_lowrank_K1'
                            assert activation['role'] == ('lift' if name.endswith('ode1') else 'project')
                            assert activation['state_width'] == cfg['model_options']['lowrank_state_width']
                            for key, value in activation.items():
                                if isinstance(value, (float, int)):assert np.isfinite(value)
                            if label != 'lowrank_augmented':assert activation['extra_output_rms_over_input'] == activation['extra_input_rms_over_input'] == 0
                        raw_flows.append(dict(dataset=dataset, label=label, step=step, **entry))
                        record = dict(dataset=dataset, label=label, step=step, layer=name)
                        record.update({'P_'+k: v for k, v in raw.items() if k != 'sigma_quantiles'}); record.update(activation); flows.append(record)
    counts = (sum(a['checkpoints_checked'] for a in audits), sum(a['arrays_checked'] for a in audits), sum(a['final_checkpoints_reloaded'] for a in replays))
    assert counts == (108, 90, 18) and len(rows) == 24 and len(initials) == 18
    assert len(curves) == 198 and len(baselines) == 36 and rng_checks == 90 and len(historical) == 6
    assert sum(len(r['checkpoints']) for r in historical) == 36 and sum(r['arrays_checked'] for r in historical) == 30
    assert len(flows) == len(raw_flows) == 792 and len(matrices) == 432
    assert len(channels) == sum(row['state_width'] for row in matrices) == 34560
    rows.sort(key=lambda r: (DATASETS.index(r['dataset']), (ARMS+HISTORICAL).index(r['label'])))
    for name, records in (('comparison', rows), ('learning-curves', curves), ('clock-factor-trajectories', flows), ('P-factor-checkpoints', matrices), ('P-spectral-channels', channels)):
        csv_write(output/(name+'.csv'), records)
    write(output/'raw-clock-factor-trajectories.json', raw_flows)
    write(output/'historical-generic-audit.json', dict(state='PASS', rows=historical, checkpoints_checked=36, arrays_checked=30))
    write(output/'initialization-audit.json', dict(state='PASS', rows=initials, snapshot_data_rng_comparisons=rng_checks,
        scope='Native FFN draws and common non-FFN states checked. Narrow/expanded FFN functions match numerically; zero-head kernel masking is not nonzero-head tangent equality. Full responses are bound by the preflight receipts.'))
    write(output/'baseline-trajectory-replay.json', dict(state='PASS', rows=baselines,
        scope='Swi128 reproduces Stage23 and SiLU128 reproduces actual Stage19 ffn_silu six saved states per task. Deterministic compatibility reruns, not independent repeats.'))
    write(output/'monitoring-evidence.json', monitors)
    decision = registered_screen(rows); write(output/'screen-decision.json', decision)
    make_figures(output, rows, curves, flows, matrices)
    for root in (current, oldroot, siluroot):verify(root)
    scope = ('Exposed validation only. Raw P is the full zero-padded generator; probe-mean VDU omits state-dependent clock derivatives. '
             'Factor norms are gauge dependent and independently sorted spectra do not track modes. Extra activation/weight movement is not causal importance. '
             'Legacy anchored Jacobians equal native for low-rank layers and are not clock ablations. No new frozen intervention, holdout, latency or goal-completion claim.')
    write(output/'provenance.json', dict(state='PASS', analysis_sha256=sha(__file__), input_sha256=hashes,
        source_files_verified=len(manifest['all_files']), checkpoints_native_audited=counts[0], arrays_native_audited=counts[1], endpoint_replays=counts[2],
        maximum_forward_replay_difference=max(a['maximum_forward_difference'] for a in replays),
        maximum_jacobian_replay_difference=max(a['maximum_derivative_difference'] for a in replays), scope=scope))
    summary = dict(state='PASS', comparison_rows=len(rows), fresh_rows=18, historical_rows=6, curve_rows=len(curves), clock_stat_rows=len(flows),
        P_factor_checkpoint_rows=len(matrices), P_spectral_channel_rows=len(channels), baseline_saved_states_equal=len(baselines), snapshot_rng_comparisons=rng_checks,
        checkpoints_native_audited=counts[0], arrays_native_audited=counts[1], endpoint_replays=counts[2], historical_checkpoints_checked=36, historical_arrays_checked=30,
        practical_leads=decision['practical_leads'], augmentation_allocation_evidence=decision['augmentation_allocation_evidence'], architecture_goal_complete=False)
    write(output/'summary.json', summary); return summary


def self_test():
    rows = [dict(dataset=d, label=a, **{m: 1. for m in METRICS}, parameters=100, forward_macs=100) for d in DATASETS for a in ARMS+HISTORICAL]
    assert registered_screen(rows)['state'] == 'STOP_FIXED_STATE_ALLOCATION'
    for row in rows:
        if row['label'] == 'lowrank_augmented':row[METRICS[0]] = .8
    decision = registered_screen(rows); assert decision['practical_leads'] == ['lowrank_augmented'] and decision['augmentation_allocation_evidence']
    # A stronger historical generic control must block the proposed lead.
    for row in rows:
        if row['label'] == 'swiglu_budget':row[METRICS[0]] = .85
    assert not registered_screen(rows)['practical_leads']
    for row in rows:
        row[METRICS[0]] = 1.
        if row['label'] == 'lowrank_narrow':row.update(parameters=80, forward_macs=80)
    assert registered_screen(rows)['practical_leads'] == ['lowrank_narrow']
    for row in rows:
        if row['label'] == 'lowrank_narrow' and row['dataset'] == DATASETS[2]:row['jacobian_error'] = 1.100001
    assert not registered_screen(rows)['practical_leads']
    curves = [dict(dataset=d, label=a, step=s, validation_oracle_mse=1/(1+s)) for d in DATASETS for a in ARMS for s in (0, 5000)]
    flows = [dict(dataset=d, label='lowrank_augmented', step=s, role=role, extra_output_rms_over_input=.1, extra_input_rms_over_input=.1)
             for d in DATASETS for s in (0, 5000) for role in ('lift', 'project')]
    matrices = [dict(dataset=d, label='lowrank_augmented', step=s, role='project', extra_right_columns_norm=s/5000, extra_controller_input_columns_norm=s/5000)
                for d in DATASETS for s in (0, 5000)]
    with tempfile.TemporaryDirectory(prefix='augmented-analyzer-contract-') as temp:
        make_figures(Path(temp), rows, curves, flows, matrices)
        assert len(list((Path(temp)/'figures').iterdir())) == 6
    return dict(state='PASS', scope='Synthetic screen and plot contracts only; no formal outcomes accessed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current', type=Path); parser.add_argument('--output', type=Path); parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:print(json.dumps(self_test(), indent=2))
    else:
        assert args.current and args.output
        print(json.dumps(analyze(args.current, args.output), indent=2))
