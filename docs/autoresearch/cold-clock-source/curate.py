"""Complete endpoint, P-spectrum and clock summaries after actual terminal exit."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import torch

LABELS = {
    'ode_k1': 'K1 native', 'fixed06': 'Clock fixed at 0.6',
    'matched_silu64': 'Matched-init SiLU64', 'matched_swiglu86': 'Matched-init Swi86',
    'feature_clock_zero': 'Feature clock (paired init)',
    'time_clock_zero': 'Time clock (paired init)',
    'swiglu86': 'Default Swi86', 'swiglu129': 'Swi129',
    'silu64': 'Default SiLU64', 'silu131': 'SiLU131', 'silu193': 'SiLU193',
    'ode_k2': 'K2 native', 'native_posthoc06': 'Native posthoc 0.6',
    'ode_k1_time': 'Frozen K1 time', 'ode_k1_pooled': 'Frozen K1 pooled',
    'ode_k1_constant': 'Frozen K1 constant', 'ode_k2_constant': 'Frozen K2 constant'}
FOCUS = ('ode_k1','fixed06','matched_silu64','matched_swiglu86','swiglu129',
         'swiglu86','feature_clock_zero','time_clock_zero')
P_ARMS = ('ode_k1','fixed06','feature_clock_zero','time_clock_zero')


def program(root):
    path = root/'source/study.py'
    if 'clock_source_study' not in sys.modules:
        spec = importlib.util.spec_from_file_location('clock_source_study', path)
        value = importlib.util.module_from_spec(spec)
        sys.modules['clock_source_study'] = value
        spec.loader.exec_module(value)
    value = sys.modules['clock_source_study']
    assert Path(value.__file__).resolve() == path.resolve()
    return value


def jsonlines(path):
    return [json.loads(v) for v in path.read_text().splitlines()]


def table(path, rows):
    with path.open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def typed_history(rows):
    strings = {'arm','provenance'}
    integers = {'step','stored_learned_parameters','active_learned_parameters',
                'training_active_learned_parameters','FFN_forward_MACs',
                'generated','reference','Heun_steps'}
    return [{k: v if k in strings else int(v) if k in integers else float(v)
             for k, v in row.items()} for row in rows]


def sample_path(study, root, arm, step=20000):
    support, _, _, _ = study.parents()
    if arm in study.ARMS:
        return root/'sampling'/f'{arm}-{step:07d}.npy'
    if arm == 'matched_swiglu86':
        return study.PARENT/'sampling'/f'{step:07d}.npy'
    if arm in ('fixed06','matched_silu64'):
        return support.COLD/'sampling'/f'{arm}-{step:07d}.npy'
    assert step == 20000
    return support.BUDGET/'output/runs'/arm/f'generated-{step:07d}.npy'


def training_rows(study, root, arm):
    support, c, _, _ = study.parents()
    if arm in study.ARMS:
        return jsonlines(root/'output/runs'/arm/'metrics.jsonl')
    if arm == 'matched_swiglu86':
        return jsonlines(study.PARENT/'output/runs'/arm/'metrics.jsonl')
    if arm in ('fixed06','matched_silu64'):
        return jsonlines(support.COLD/'output/runs'/arm/'metrics.jsonl')
    early = c.PARENTS['ordinary'] if arm.startswith('silu') else support.NATIVE
    values = jsonlines(early/'output/runs'/arm/'metrics.jsonl')
    values += jsonlines(support.BUDGET/'output/runs'/arm/'metrics.jsonl')
    return list({row['step']: row for row in values}.values())


def p_arrays(study, root, arm, step):
    support, _, _, _ = study.parents()
    if arm == 'ode_k1':
        state = study.native_state(step)
    else:
        parent = root if arm in study.ARMS else support.COLD
        state = torch.load(parent/'output/runs'/arm/f'step_{step:07d}.pt',
                           weights_only=False, map_location='cpu')
    return {f'block{j}.P{k}': state['model'][f'blocks.{j}.mlp.ode{k}.proj.weight'].numpy().copy()
            for j in range(3) for k in (1, 2)}


def main(root):
    study = program(root)
    study.environment(root, 'training')
    study.verify(root)
    gate = study.module('clock_source_analysis_gate', root/'source/controller.py')
    gate.gate(root)
    registration = study.load(root/'curation-source/registration.json')
    assert registration['registered_before_new_FID']
    for name, digest in registration['files'].items():
        assert study.sha(root/'curation-source'/name) == digest
    terminal = study.load(root/'controller-tool-exit.json')
    assert terminal['exit_code'] == terminal['terminal_tool_result']['exit_code'] == 0
    assert study.load(root/'orchestrator/status.json')['state'] == 'COMPLETE'
    launches = study.load(root/'orchestrator/launches.json')
    assert len(launches) == 7 and all(v['exit_code'] == 0 for v in launches)
    assert {v['arm'] for v in launches if v['phase'] == 'training'} == set(study.ARMS)
    for phase in ('training','sampling','metrics'):
        assert study.load(root/f'audit-{phase}/result.json')['status'] == 'PASS'
    formal = study.load(root/'metric-output/result.json')
    assert formal['status'] == 'PASS' and formal['historical_rows'] == study.historical_rows()
    assert len(formal['historical_rows']) == 35
    out = root/'curation'
    out.mkdir()
    comparison = typed_history(formal['historical_rows'])
    support, _, _, _ = study.parents()
    reference = np.load(support.NATIVE/'metric-inputs/reference-features.npy').astype(np.float64)
    assert [(v['arm'], v['step']) for v in formal['rows']] == list(study.SAMPLES)
    for row in formal['rows']:
        arm, step = row['arm'], row['step']
        raw = np.load(sample_path(study, root, arm, step))
        assert raw.shape == (1024, 3, 32, 32) and np.isfinite(raw).all()
        features = np.load(root/'metric-output'/f'{arm}-{step:07d}-features.npy').astype(np.float64)
        mean = float(np.sum((features.mean(0)-reference.mean(0))**2))
        comparison.append(dict(arm=arm, step=step, weighted_mse=row['weighted_mse'],
            clean_mse=row['clean_mse'], fid=row['fid'], FID_mean_term=mean,
            FID_covariance_term=row['fid']-mean, raw_tail_fraction=float(np.mean(np.abs(raw)>1)),
            stored_learned_parameters=211094, active_learned_parameters=211094,
            training_active_learned_parameters=211094, FFN_forward_MACs=row['FFN_forward_MACs'],
            generated=1024, reference=2048, Heun_steps=128,
            provenance='new cold paired clock-source trajectory'))
    assert len(comparison) == len({(v['arm'], v['step']) for v in comparison}) == 41
    comparison.sort(key=lambda v: (v['step'], list(LABELS).index(v['arm'])))
    primary = [v for v in comparison if v['step'] == 20000]
    assert len(primary) == 17
    table(out/'comparison.csv', comparison)
    table(out/'primary-20k.csv', primary)
    controls = [dict(arm=v['arm'], fid=float(v['fid']), weighted_mse=float(v['weighted_mse']),
                     clean_mse=float(v['clean_mse'])) for v in formal['historical_rows']
                if int(v['step']) == 20000]
    controls.append(next(v for v in formal['rows'] if v['arm'] == study.ARMS[0] and v['step'] == 20000))
    best = min(controls, key=lambda v: v['fid'])
    candidate = next(v for v in formal['rows'] if v['arm'] == study.ARMS[1] and v['step'] == 20000)
    checks = dict(FID=candidate['fid'] <= .95*best['fid'],
                  weighted_mse=candidate['weighted_mse'] <= 1.05*best['weighted_mse'],
                  clean_mse=candidate['clean_mse'] <= 1.05*best['clean_mse'])
    decision = study.load(root/'metric-output/decision.json')
    assert decision['best_control'] == best and decision['candidate'] == candidate
    assert decision['checks'] == checks and decision['quality_pass'] == all(checks.values())
    assert not decision['new_ODE_method_established']
    study.write(out/'decision.json', decision)
    contrasts = []
    for arm in study.ARMS:
        a = next(v for v in primary if v['arm'] == arm)
        for b in primary:
            if b['arm'] != arm:
                contrasts.append(dict(arm=arm, reference=b['arm'], step=20000,
                    FID_difference=a['fid']-b['fid'],
                    **{key+'_ratio': a[key]/b[key] for key in ('fid','weighted_mse','clean_mse')}))
    assert len(contrasts) == 32
    table(out/'primary-contrasts.csv', contrasts)
    curves = []
    for arm in FOCUS:
        for row in sorted(training_rows(study, root, arm), key=lambda v: v['step']):
            curves.append(dict(arm=arm, step=row['step'], **{key: row.get(key) for key in
                ('weighted_mse','clean_mse','train_loss','grad_norm','clip_fraction')}))
    table(out/'training-curves.csv', curves)
    matrices, eigenvalues, initial = [], [], None
    for arm in P_ARMS:
        for step in study.STEPS:
            arrays = p_arrays(study, root, arm, step)
            if step == 0:
                if initial is None:
                    initial = arrays
                else:
                    assert all(np.array_equal(v, initial[k]) for k, v in arrays.items())
            for name, value in arrays.items():
                a = value.astype(np.float64)
                singular, eigen = np.linalg.svd(a, compute_uv=False), np.linalg.eigvals(a)
                matrices.append(dict(arm=arm, step=step, layer=name,
                    sigma_max=float(singular[0]), sigma_min=float(singular[-1]),
                    singular_CV=float(singular.std()/singular.mean()), frobenius=float(np.linalg.norm(a)),
                    eigen_max_real=float(eigen.real.max()), eigen_max_abs=float(np.abs(eigen).max())))
                eigenvalues += [dict(arm=arm, step=step, layer=name, index=j,
                                     real=float(v.real), imag=float(v.imag)) for j, v in enumerate(eigen)]
    assert len(matrices) == 168 and len(eigenvalues) == 10752
    table(out/'P-history.csv', matrices)
    table(out/'P-eigenvalues.csv', eigenvalues)
    profiles, heads, panels, conditional = [], [], [], []
    for arm in study.ARMS:
        for step in study.STEPS:
            folder = root/'output/runs'/arm
            values = study.load(folder/f'clock-profile-{step:07d}.json')
            assert len(values) == 30
            profiles += [dict(arm=arm, step=step, **v) for v in values]
            state = torch.load(folder/f'step_{step:07d}.pt', weights_only=False, map_location='cpu')
            for j in range(3):
                for k in (1, 2):
                    name = f'blocks.{j}.mlp.ode{k}.t_func.2'
                    heads.append(dict(arm=arm, step=step, layer=f'block{j}.clock{k}',
                        weight_norm=float(state['model'][name+'.weight'].double().norm()),
                        bias=float(state['model'][name+'.bias'][0])))
            for t in (.1, .3, .5, .7, .9):
                with np.load(folder/f'panel_{step:07d}_{t:.1f}.npz') as saved:
                    bank = study.bank(t)
                    error = saved['prediction'].astype(np.float64)-bank['target'].astype(np.float64)
                delta = bank['delta'].astype(np.float64)[:, None, None, None]
                panels.append(dict(arm=arm, step=step, time=t, weighted_mse=float(np.mean((error/delta)**2)),
                                   clean_mse=float(np.mean(error**2)), images=64))
            if arm == study.ARMS[1]:
                arrays = p_arrays(study, root, arm, step)
                for v in values:
                    assert v['minimum'] == v['maximum']
                    name = v['layer'].replace('.clock', '.P')
                    a = np.eye(64)+v['minimum']*arrays[name].astype(np.float64)
                    s, e = np.linalg.svd(a, compute_uv=False), np.linalg.eigvals(a)
                    conditional.append(dict(arm=arm, step=step, layer=name, time=v['time'],
                        clock=v['minimum'], sigma_max=float(s[0]), sigma_min=float(s[-1]),
                        singular_CV=float(s.std()/s.mean()), eigen_max_real=float(e.real.max()),
                        eigen_max_abs=float(np.abs(e).max())))
    assert (len(profiles), len(heads), len(panels), len(conditional)) == (420, 84, 70, 210)
    table(out/'clock-profiles.csv', profiles)
    table(out/'clock-head-history.csv', heads)
    table(out/'time-panels.csv', panels)
    table(out/'conditional-weight-history.csv', conditional)
    spans = []
    for arm in study.ARMS:
        for step in study.STEPS:
            for layer in sorted({v['layer'] for v in profiles}):
                selected = [v for v in profiles if (v['arm'], v['step'], v['layer']) == (arm, step, layer)]
                assert len(selected) == 5
                spans.append(dict(arm=arm, step=step, layer=layer,
                    panel_mean_min=min(v['mean'] for v in selected), panel_mean_max=max(v['mean'] for v in selected),
                    panel_mean_span=max(v['mean'] for v in selected)-min(v['mean'] for v in selected),
                    mean_within_panel_std=float(np.mean([v['std'] for v in selected]))))
    table(out/'clock-panel-spans.csv', spans)
    resources = jsonlines(root/'orchestrator/resources.jsonl')
    use = []
    for child in launches:
        values = [v for row in resources if row['phase'] == child['phase']
                  for v in row['resources'] if v['pid'] == child['pid']]
        assert values
        use.append(dict(phase=child['phase'], arm=child['arm'], pid=child['pid'],
            cohort_phase_seconds=child['phase_seconds'],
            training_cell_seconds=study.load(root/'output/runs'/child['arm']/'result.json')['seconds']
                if child['phase'] == 'training' else None,
            peak_RSS=max(v['RSS'] for v in values), mean_CPU_percent=float(np.mean([v['CPU_percent'] for v in values]))))
    table(out/'resources.csv', use)
    ledger = []
    for arm in study.ARMS:
        row = next(v for v in primary if v['arm'] == arm)
        ledger.append(dict(commit=study.load(root/'settings.json')['source_commit'][:7], arm=arm,
            fid=row['fid'], peak_child_RSS_GiB=max(v['peak_RSS'] for v in use if v['arm'] == arm)/1024**3,
            status='keep' if arm == study.ARMS[1] and all(checks.values()) else 'control' if arm == study.ARMS[0] else 'discard',
            description='Paired cold clock-source comparison; primary20k; no method established by this ledger'))
    assert not (root/'results.tsv').exists()
    for path in (root/'results.tsv', out/'results.tsv'):
        with path.open('w') as file:
            writer = csv.DictWriter(file, fieldnames=list(ledger[0]), delimiter='\t')
            writer.writeheader()
            writer.writerows(ledger)
    np.savez(out/'case-images.npz', **{arm: np.load(sample_path(study, root, arm))[:8] for arm in FOCUS})
    study.write(out/'plot-inputs.json', dict(labels=LABELS, focus=FOCUS, P_arms=P_ARMS, cases=FOCUS))
    study.write(out/'summary.json', dict(status='PASS', endpoints=41, primary20k=17,
        new_trajectories=2, contrasts=32, P_rows=168, P_eigenvalues=10752,
        clock_profiles=420, clock_heads=84, clock_spans=84, conditional_linear_maps=210,
        denoising_panels=70, fixed_cases=64, quality_gate_pass=all(checks.values()),
        new_ODE_method_established=False))
    study.verify(root)
    print('COLD_CLOCK_SOURCE_CURATION_COMPLETE', json.dumps(decision), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root.resolve())
