"""Complete fixed-budget comparison after the controller's actual terminal exit."""
from __future__ import annotations
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import numpy as np

ARM = 'matched_swiglu86'
LABELS = {
    'ode_k1': 'K1 native', 'fixed06': 'Clock fixed at 0.6',
    'matched_silu64': 'Matched-init SiLU64', ARM: 'Matched-init Swi86',
    'swiglu86': 'Default Swi86', 'swiglu129': 'Swi129',
    'silu64': 'Default SiLU64', 'silu131': 'SiLU131', 'silu193': 'SiLU193',
    'ode_k2': 'K2 native', 'native_posthoc06': 'Native posthoc 0.6',
    'ode_k1_time': 'Frozen K1 time', 'ode_k1_pooled': 'Frozen K1 pooled',
    'ode_k1_constant': 'Frozen K1 constant', 'ode_k2_constant': 'Frozen K2 constant'}
FOCUS = ('ode_k1', 'fixed06', 'matched_silu64', 'swiglu86', ARM, 'swiglu129', 'silu131', 'silu64')
CASE_ARMS = ('ode_k1', 'fixed06', 'matched_silu64', 'swiglu86', ARM, 'swiglu129')


def rows(path):
    return [json.loads(v) for v in path.read_text().splitlines()]


def table(path, values):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def historical_metrics(study, arm):
    if arm in ('fixed06', 'matched_silu64'):
        return rows(study.COLD/'output/runs'/arm/'metrics.jsonl')
    c, _, _ = study.parents()
    early = c.PARENTS['ordinary'] if arm.startswith('silu') else study.NATIVE
    values = rows(early/'output/runs'/arm/'metrics.jsonl')
    values += rows(study.BUDGET/'output/runs'/arm/'metrics.jsonl')
    return list({v['step']: v for v in values}.values())


def case_path(study, root, arm, step=20000):
    if arm == ARM:
        return root/'sampling'/f'{step:07d}.npy'
    if arm in ('fixed06', 'matched_silu64'):
        return study.COLD/'sampling'/f'{arm}-{step:07d}.npy'
    return study.BUDGET/'output/runs'/arm/f'generated-{step:07d}.npy'


def main(root):
    sys.path.insert(0, str(root/'source'))
    import init_study as study
    study.parents()
    study.verify(root)
    source = root/'curation-source'
    registration = study.load(source/'registration.json')
    assert registration['registered_before_new_FID']
    for name, digest in registration['files'].items():
        assert study.sha(source/name) == digest
    assert study.load(root/'orchestrator/status.json')['state'] == 'COMPLETE'
    exit_record = study.load(root/'controller-tool-exit.json')
    assert exit_record['terminal_tool_result']['exit_code'] == exit_record['exit_code'] == 0
    phases = study.load(root/'orchestrator/launches.json')
    assert len(phases) == 6 and all(v['exit_code'] == 0 for v in phases)
    for phase in ('training', 'sampling', 'metrics'):
        assert study.load(root/f'audit-{phase}/result.json')['status'] == 'PASS'
    formal = study.load(root/'metric-output/result.json')
    historical = study.load(study.COLD/'metric-output/result.json')['rows']
    assert formal['historical_rows'] == historical and len(historical) == 32
    out = root/'curation'
    out.mkdir()
    old = list(csv.DictReader((study.COLD/'curation/comparison.csv').open()))
    assert len(old) == 32
    by = {(r['arm'], r['step']): r for r in historical}
    integers = {'step', 'stored_learned_parameters', 'active_learned_parameters',
                'training_active_learned_parameters', 'FFN_forward_MACs', 'generated',
                'reference', 'Heun_steps'}
    strings = {'arm', 'provenance'}
    comparison = [{k: v if k in strings else int(v) if k in integers else float(v)
                   for k, v in r.items()} for r in old]
    for r in comparison:
        original = by[r['arm'], r['step']]
        for key in ('fid', 'weighted_mse', 'clean_mse'):
            assert r[key] == original[key]
    reference = np.load(study.NATIVE/'metric-inputs/reference-features.npy').astype(np.float64)
    for r in formal['rows']:
        step = r['step']
        features = np.load(root/'metric-output'/f'{step:07d}-features.npy').astype(np.float64)
        mean = float(np.sum((features.mean(0)-reference.mean(0))**2))
        raw = np.load(root/'sampling'/f'{step:07d}.npy')
        assert raw.shape == (1024, 3, 32, 32) and np.isfinite(raw).all()
        comparison.append(dict(arm=ARM, step=step, weighted_mse=r['weighted_mse'], clean_mse=r['clean_mse'],
            fid=r['fid'], FID_mean_term=mean, FID_covariance_term=r['fid']-mean,
            raw_tail_fraction=float(np.mean(np.abs(raw)>1)), stored_learned_parameters=211412,
            active_learned_parameters=211412, training_active_learned_parameters=211412,
            FFN_forward_MACs=16512, generated=1024, reference=2048, Heun_steps=128,
            provenance='new cold matched-function Swi86 control'))
    assert len(comparison) == 35 and len({(r['arm'], r['step']) for r in comparison}) == 35
    comparison.sort(key=lambda v: (v['step'], list(LABELS).index(v['arm'])))
    primary = [v for v in comparison if v['step'] == 20000]
    assert len(primary) == 15
    table(out/'comparison.csv', comparison)
    table(out/'primary-20k.csv', primary)
    candidate = next(v for v in primary if v['arm'] == ARM)
    baseline = next(v for v in primary if v['arm'] == 'swiglu86')
    contrasts = []
    for control in primary:
        if control['arm'] == ARM:
            continue
        contrasts.append(dict(reference=control['arm'], step=20000,
            FID_difference=candidate['fid']-control['fid'],
            **{k+'_ratio': candidate[k]/control[k] for k in ('fid', 'weighted_mse', 'clean_mse')}))
    table(out/'primary-contrasts.csv', contrasts)
    checks = dict(FID_five_percent=candidate['fid'] <= .95*baseline['fid'],
                  weighted_mse_guard=candidate['weighted_mse'] <= 1.05*baseline['weighted_mse'],
                  clean_mse_guard=candidate['clean_mse'] <= 1.05*baseline['clean_mse'])
    study.write(out/'decision.json', dict(status='PASS', quality_gate=checks,
        initialization_control_supported=all(checks.values()), new_ODE_method=False,
        primary_step=20000, new_trajectories=1, historical_rows_retained=32,
        default_FID=baseline['fid'], matched_FID=candidate['fid'],
        FID_improvement_percent=100*(1-candidate['fid']/baseline['fid']),
        rescue_if_failed=False))
    curves = []
    for arm in FOCUS:
        values = rows(root/'output/runs'/ARM/'metrics.jsonl') if arm == ARM else historical_metrics(study, arm)
        for value in sorted(values, key=lambda v: v['step']):
            curves.append(dict(arm=arm, step=value['step'], **{k: value.get(k) for k in
                ('weighted_mse', 'clean_mse', 'train_loss', 'grad_norm', 'clip_fraction')}))
    table(out/'training-curves.csv', curves)
    matrix = []
    branches = []
    for step in study.STEPS:
        folder = root/'output/runs'/ARM
        with np.load(folder/f'matrices-{step:07d}.npz') as values:
            for name in sorted(values.files):
                singular = np.linalg.svd(values[name].astype(np.float64), compute_uv=False)
                matrix.append(dict(step=step, layer=name, sigma_max=float(singular[0]),
                    frobenius=float(np.linalg.norm(values[name].astype(np.float64))),
                    singular_CV=float(singular.std()/singular.mean()) if singular.mean() else None))
        branches += [dict(step=step, **v) for v in study.load(folder/f'branch-history-{step:07d}.json')]
    assert len(matrix) == 63 and len(branches) == 21
    table(out/'matrix-history.csv', matrix)
    table(out/'branch-history.csv', branches)
    panels = []
    c, n, _ = study.parents()
    for step in study.STEPS:
        for t in (.1, .3, .5, .7, .9):
            bank = c.bank(t)
            with np.load(root/'output/runs'/ARM/f'panel_{step:07d}_{t:.1f}.npz') as values:
                error = values['prediction'].astype(np.float64)-bank['target'].astype(np.float64)
            panels.append(dict(step=step, time=t, clean_mse=float(np.mean(error**2)),
                weighted_mse=float(np.mean((error/bank['delta'].astype(np.float64)[:, None, None, None])**2)), images=64))
    table(out/'time-panels.csv', panels)
    resources = rows(root/'orchestrator/resources.jsonl')
    table(out/'resources.csv', [dict(action=p['action'], seconds=p['seconds'],
        peak_RSS=max(v.get('RSS', 0) for v in resources if v['action'] == p['action']),
        mean_CPU_percent=float(np.mean([v['CPU_percent'] for v in resources
                               if v['action'] == p['action'] and 'CPU_percent' in v]))) for p in phases])
    ledger = dict(commit=study.load(root/'settings.json')['source_commit'][:7],
        fid=candidate['fid'], peak_child_RSS_GiB=max(v.get('RSS', 0) for v in resources)/1024**3,
        status='keep' if all(checks.values()) else 'discard',
        description='One cold matched-initial-function Swi86 baseline control; fixed20k; not a new ODE method')
    assert not (root/'results.tsv').exists()
    for path in (root/'results.tsv', out/'results.tsv'):
        with path.open('w') as file:
            writer = csv.DictWriter(file, fieldnames=list(ledger), delimiter='\t')
            writer.writeheader()
            writer.writerow(ledger)
    # The first eight original starts are fixed before any new quality scores.
    images = {arm: np.load(case_path(study, root, arm))[:8] for arm in CASE_ARMS}
    np.savez(out/'case-images.npz', **images)
    study.write(out/'plot-inputs.json', dict(labels=LABELS, focus=FOCUS, cases=CASE_ARMS))
    study.verify(root)
    print('SWIGLU_INIT_CURATION_COMPLETE', json.dumps(study.load(out/'decision.json')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root.resolve())
