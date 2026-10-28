"""Verify every curated comparison and preserve a completed clock-source cohort."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import csv
import importlib.util
import re
import shutil
import subprocess
from pathlib import Path
import numpy as np
import torch


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(root, doc):
    curator = load_module('clock_source_registered_curator', root/'curation-source/curate.py')
    study = curator.program(root)
    support, _, _, _ = study.parents()
    study.verify(root)
    assert not (root/'sealed-runtime.json').exists()
    repo = doc.parents[2]
    registered = study.load(root/'curation-source/registration.json')
    assert registered['registered_before_new_FID']
    for name, digest in registered['files'].items():
        assert study.sha(doc/name) == study.sha(root/'curation-source'/name) == digest
        expected = subprocess.check_output(['git','-C',str(repo),'show',
            registered['source_commit']+':'+str((doc/name).relative_to(repo))])
        assert expected == (doc/name).read_bytes()
    for name in ('study.py','audit_family.py','controller.py'):
        assert study.sha(doc/name) == study.sha(root/'source'/name)
    for name in ('controller','curation','plot'):
        receipt = study.load(root/f'{name}-tool-exit.json')
        assert receipt['exit_code'] == receipt['terminal_tool_result']['exit_code'] == 0
    launches = study.load(root/'orchestrator/launches.json')
    assert len(launches) == 7 and all(v['exit_code'] == 0 for v in launches)
    assert study.load(root/'orchestrator/status.json')['state'] == 'COMPLETE'
    audits = {name: study.load(root/f'audit-{name}/result.json')
              for name in ('training','sampling','metrics')}
    assert all(v['status'] == 'PASS' for v in audits.values())
    assert audits['training']['stream_boundaries_verified'] == 42
    assert audits['training']['physical_last_updates'] == 12
    assert not audits['training']['full_optimizer_trajectory_replayed']
    assert audits['sampling']['generated_images_bitwise'] == 6144
    assert audits['metrics']['generated_features_bitwise'] == 6144
    assert audits['metrics']['reference_features_bitwise'] == 2048
    results = doc/'results'
    for source in (root/'curation').rglob('*'):
        if source.is_file() and source.suffix != '.npz':
            assert study.sha(source) == study.sha(results/source.relative_to(root/'curation'))
    read = lambda name: list(csv.DictReader((results/name).open()))
    formal = study.load(root/'metric-output/result.json')
    expected_rows = {(v['arm'], int(v['step'])): v for v in formal['historical_rows']}
    expected_rows.update({(v['arm'], v['step']): v for v in formal['rows']})
    comparison = read('comparison.csv')
    assert len(comparison) == len(expected_rows) == 41
    for row in comparison:
        expected = expected_rows[row['arm'], int(row['step'])]
        for key in ('weighted_mse','clean_mse','fid'):
            assert float(row[key]) == float(expected[key])
    primary = read('primary-20k.csv')
    assert len(primary) == 17 and primary == [v for v in comparison if int(v['step']) == 20000]
    decision = study.load(results/'decision.json')
    assert decision == study.load(root/'metric-output/decision.json')
    best = next(v for v in primary if v['arm'] == decision['best_control']['arm'])
    candidate = next(v for v in primary if v['arm'] == study.ARMS[1])
    assert float(best['fid']) == min(float(v['fid']) for v in primary if v['arm'] != study.ARMS[1])
    checks = dict(FID=float(candidate['fid']) <= .95*float(best['fid']),
        weighted_mse=float(candidate['weighted_mse']) <= 1.05*float(best['weighted_mse']),
        clean_mse=float(candidate['clean_mse']) <= 1.05*float(best['clean_mse']))
    assert decision['checks'] == checks and decision['quality_pass'] == all(checks.values())
    assert not decision['new_ODE_method_established']
    contrasts = read('primary-contrasts.csv')
    assert len(contrasts) == 32
    for row in contrasts:
        a = next(v for v in primary if v['arm'] == row['arm'])
        b = next(v for v in primary if v['arm'] == row['reference'])
        assert row['arm'] in study.ARMS and row['reference'] != row['arm']
        assert float(row['FID_difference']) == float(a['fid'])-float(b['fid'])
        for key in ('fid','weighted_mse','clean_mse'):
            assert float(row[key+'_ratio']) == float(a[key])/float(b[key])

    cache = {}
    def state(arm, step):
        key = arm, step
        if key not in cache:
            if arm == 'ode_k1':
                parent = support.NATIVE if step <= 5000 else support.BUDGET
            elif arm == 'fixed06':
                parent = support.COLD
            else:
                assert arm in study.ARMS
                parent = root
            cache[key] = torch.load(parent/'output/runs'/arm/f'step_{step:07d}.pt',
                weights_only=False, map_location='cpu')['model']
        return cache[key]
    def array(arm, step, layer):
        j, k = re.fullmatch(r'block([0-2])\.P([12])', layer).groups()
        return state(arm, step)[f'blocks.{j}.mlp.ode{k}.proj.weight'].numpy().astype(np.float64)
    spectra = read('P-history.csv')
    assert len(spectra) == 168
    spectrum_cache = {}
    for row in spectra:
        a = array(row['arm'], int(row['step']), row['layer'])
        singular, eigen = np.linalg.svd(a, compute_uv=False), np.linalg.eigvals(a)
        expected = [singular[0], singular[-1], singular.std()/singular.mean(),
                    np.linalg.norm(a), eigen.real.max(), np.abs(eigen).max()]
        observed = [float(row[k]) for k in ('sigma_max','sigma_min','singular_CV',
                    'frobenius','eigen_max_real','eigen_max_abs')]
        np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)
        spectrum_cache[row['arm'], int(row['step']), row['layer']] = eigen
    eigen_rows = read('P-eigenvalues.csv')
    assert len(eigen_rows) == 10752
    for row in eigen_rows:
        value = spectrum_cache[row['arm'], int(row['step']), row['layer']][int(row['index'])]
        np.testing.assert_allclose([float(row['real']), float(row['imag'])],
                                   [value.real, value.imag], rtol=1e-12, atol=1e-12)
    profiles = read('clock-profiles.csv')
    assert len(profiles) == 420
    actual_profiles = {}
    for arm in study.ARMS:
        for step in study.STEPS:
            for row in study.load(root/'output/runs'/arm/f'clock-profile-{step:07d}.json'):
                actual_profiles[arm, step, row['layer'], row['time']] = row
    for row in profiles:
        expected = actual_profiles[row['arm'], int(row['step']), row['layer'], float(row['time'])]
        for key in ('mean','std','minimum','maximum','values'):
            assert float(row[key]) == expected[key]
    conditional = read('conditional-weight-history.csv')
    assert len(conditional) == 210
    for row in conditional:
        p = array(row['arm'], int(row['step']), row['layer'])
        profile = actual_profiles[row['arm'], int(row['step']), row['layer'].replace('.P','.clock'), float(row['time'])]
        assert profile['minimum'] == profile['maximum'] == float(row['clock'])
        w = np.eye(64)+float(row['clock'])*p
        s, e = np.linalg.svd(w, compute_uv=False), np.linalg.eigvals(w)
        np.testing.assert_allclose([float(row[k]) for k in ('sigma_max','sigma_min','singular_CV','eigen_max_real','eigen_max_abs')],
            [s[0], s[-1], s.std()/s.mean(), e.real.max(), np.abs(e).max()], rtol=1e-12, atol=1e-12)
    assert len(read('clock-head-history.csv')) == len(read('clock-panel-spans.csv')) == 84
    panels = read('time-panels.csv')
    assert len(panels) == 70
    for row in panels:
        arm, step, t = row['arm'], int(row['step']), float(row['time'])
        with np.load(root/'output/runs'/arm/f'panel_{step:07d}_{t:.1f}.npz') as saved:
            bank = study.bank(t)
            error = saved['prediction'].astype(np.float64)-bank['target'].astype(np.float64)
        delta = bank['delta'].astype(np.float64)[:, None, None, None]
        assert float(row['weighted_mse']) == float(np.mean((error/delta)**2))
        assert float(row['clean_mse']) == float(np.mean(error**2))
    curves = read('training-curves.csv')
    assert {v['arm'] for v in curves} == set(curator.FOCUS)
    for arm in curator.FOCUS:
        expected = {v['step']: v for v in curator.training_rows(study, root, arm)}
        selected = [v for v in curves if v['arm'] == arm]
        assert len(selected) == len(expected)
        for row in selected:
            value = expected[int(row['step'])]
            for key in ('weighted_mse','clean_mse','train_loss','grad_norm','clip_fraction'):
                assert (float(row[key]) if row[key] != '' else None) == value.get(key)
    visual = study.load(root/'visual-review.json')
    pngs = {p.name for p in (results/'figures').glob('*.png')}
    assert visual['status'] == 'PASS' and len(pngs) == 7 and set(visual['files']) == pngs
    for name, digest in visual['files'].items():
        assert study.sha(results/'figures'/name) == digest
        pdf = (results/'figures'/name).with_suffix('.pdf')
        info = subprocess.check_output(['/opt/homebrew/bin/pdfinfo', str(pdf)], text=True)
        assert re.search(r'^Pages:\s+1$', info, re.M)
    reported = []
    for line in (results/'analysis-report.md').read_text().splitlines():
        cells = [v.strip().replace('**','') for v in line.strip().strip('|').split('|')]
        if len(cells) == 4 and cells[0] in curator.LABELS.values():
            reported.append(cells)
    assert len(reported) == 17
    for label, *values in reported:
        arm = next(k for k, value in curator.LABELS.items() if value == label)
        row = next(v for v in primary if v['arm'] == arm)
        assert values == [format(float(row[k]), fmt) for k, fmt in
                          [('weighted_mse','.6f'),('clean_mse','.6f'),('fid','.2f')]]
    assert (root/'results.tsv').read_bytes() == (results/'results.tsv').read_bytes()
    assert not subprocess.check_output(['git','-C',str(repo),'diff','--name-only','98730e0',
                '--','src','synthetic','configs','scripts','reports/report1'], text=True)
    study.verify(root)
    study.write(results/'artifact-verification.json', dict(status='PASS', endpoints=41,
        primary20k=17, contrasts=32, P_summaries=168, P_eigenvalues=10752,
        clock_profiles=420, conditional_weight_summaries=210, panels=70,
        reported_primary_rows_verified=17, figures_visually_reviewed=7,
        standalone_one_page_PDFs=7, original_source_and_Report1_unchanged=True,
        new_ODE_method_established=False))
    shutil.copytree(results, root/'curated-results')
    closure = root/'closure-source'
    closure.mkdir()
    for p in doc.iterdir():
        if p.is_file():
            shutil.copyfile(p, closure/p.name)
    files = sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                   and p.name != 'sealed-runtime.json')
    study.write(root/'sealed-runtime.json', {str(p.relative_to(root)): study.sha(p) for p in files})
    for name, digest in study.load(root/'sealed-runtime.json').items():
        assert study.sha(root/name) == digest
    print(study.json.dumps(dict(status='SEALED_AND_VERIFIED', files=len(files),
        private_bytes=sum(p.stat().st_size for p in files),
        seal_sha256=study.sha(root/'sealed-runtime.json'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--doc', type=Path, required=True)
    args = parser.parse_args()
    main(args.root.resolve(), args.doc.resolve())
