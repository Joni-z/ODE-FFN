"""Check published numerical artifacts and preserve the complete private runtime."""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np


def main(root, doc):
    sys.path.insert(0, str(root/'source'))
    import init_study as study
    study.parents()
    study.verify(root)
    assert not (root/'sealed-runtime.json').exists()
    repo = doc.parents[2]
    registration = study.load(root/'curation-source/registration.json')
    for name, digest in registration['files'].items():
        assert study.sha(root/'curation-source'/name) == study.sha(doc/name) == digest
        committed = subprocess.check_output(['git', '-C', str(repo), 'show',
                     registration['source_commit']+':'+str((doc/name).relative_to(repo))])
        assert committed == (doc/name).read_bytes()
    for name in ('init_study.py', 'init_audit.py', 'init_controller.py'):
        assert study.sha(root/'source'/name) == study.sha(doc/name)
    for name in ('controller', 'curation', 'plot'):
        receipt = study.load(root/f'{name}-tool-exit.json')
        assert receipt['exit_code'] == receipt['terminal_tool_result']['exit_code'] == 0
    assert study.load(root/'orchestrator/status.json')['state'] == 'COMPLETE'
    phases = study.load(root/'orchestrator/launches.json')
    assert len(phases) == 6 and all(v['exit_code'] == 0 for v in phases)
    for phase in ('training', 'sampling', 'metrics'):
        assert study.load(root/f'audit-{phase}/result.json')['status'] == 'PASS'
    spec = importlib.util.spec_from_file_location('registered_init_curator', root/'curation-source/curate_init.py')
    curator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(curator)
    out = root/'curation'
    results = doc/'results'
    for source in out.rglob('*'):
        if source.is_file() and source.suffix != '.npz':
            assert study.sha(source) == study.sha(results/source.relative_to(out))
    read = lambda name: list(csv.DictReader((results/name).open()))
    comparison = read('comparison.csv')
    formal = study.load(root/'metric-output/result.json')
    source_rows = {**{(v['arm'], v['step']): v for v in formal['historical_rows']},
                   **{(study.ARM, v['step']): v for v in formal['rows']}}
    assert len(comparison) == len(source_rows) == 35
    for row in comparison:
        value = source_rows[row['arm'], int(row['step'])]
        for key in ('fid', 'weighted_mse', 'clean_mse'):
            assert float(row[key]) == value[key]
    primary = read('primary-20k.csv')
    assert len(primary) == 15 and primary == [r for r in comparison if int(r['step']) == 20000]
    candidate = next(v for v in primary if v['arm'] == study.ARM)
    default = next(v for v in primary if v['arm'] == 'swiglu86')
    checks = dict(FID_five_percent=float(candidate['fid']) <= .95*float(default['fid']),
        weighted_mse_guard=float(candidate['weighted_mse']) <= 1.05*float(default['weighted_mse']),
        clean_mse_guard=float(candidate['clean_mse']) <= 1.05*float(default['clean_mse']))
    decision = study.load(results/'decision.json')
    assert decision['quality_gate'] == checks
    assert decision['initialization_control_supported'] == all(checks.values())
    assert not decision['new_ODE_method'] and not decision['rescue_if_failed']
    assert (results/'results.tsv').read_bytes() == (root/'results.tsv').read_bytes()
    ledger = list(csv.DictReader((results/'results.tsv').open(), delimiter='\t'))
    assert len(ledger) == 1 and float(ledger[0]['fid']) == float(candidate['fid'])
    assert ledger[0]['status'] == ('keep' if all(checks.values()) else 'discard')
    contrasts = read('primary-contrasts.csv')
    assert len(contrasts) == 14
    for row in contrasts:
        control = next(v for v in primary if v['arm'] == row['reference'])
        assert float(row['FID_difference']) == float(candidate['fid'])-float(control['fid'])
        for key in ('fid', 'weighted_mse', 'clean_mse'):
            assert float(row[key+'_ratio']) == float(candidate[key])/float(control[key])
    assert len(read('matrix-history.csv')) == 63 and len(read('branch-history.csv')) == 21
    assert len(read('time-panels.csv')) == 35
    visual = study.load(root/'visual-review.json')
    pngs = {v.name for v in (results/'figures').glob('*.png')}
    assert visual['status'] == 'PASS' and len(pngs) == 5 and set(visual['files']) == pngs
    for name, digest in visual['files'].items():
        assert study.sha(results/'figures'/name) == digest
        pdf = (results/'figures'/name).with_suffix('.pdf')
        assert pdf.read_bytes().startswith(b'%PDF-')
    # The human-readable primary table must reproduce every endpoint, not a subset.
    reported = []
    for line in (results/'analysis-report.md').read_text().splitlines():
        cells = [v.strip().replace('**', '') for v in line.strip().strip('|').split('|')]
        if len(cells) == 4 and cells[0] in curator.LABELS.values():
            reported.append(cells)
    assert len(reported) == 15
    for label, *values in reported:
        arm = next(k for k, v in curator.LABELS.items() if v == label)
        row = next(v for v in primary if v['arm'] == arm)
        assert values == [format(float(row[k]), precision) for k, precision in
                          [('weighted_mse', '.6f'), ('clean_mse', '.6f'), ('fid', '.2f')]]
    assert not subprocess.check_output(['git', '-C', str(repo), 'diff', '--name-only',
        '98730e0', '--', 'src', 'synthetic', 'configs', 'scripts', 'reports/report1'], text=True)
    study.verify(root)
    study.write(results/'artifact-verification.json', dict(status='PASS',
        endpoint_rows=35, primary_rows=15, contrasts=14, matrix_rows=63,
        branch_rows=21, time_panels=35, figures_visually_reviewed=5,
        protected_scopes_unchanged=True, new_ODE_method=False))
    shutil.copytree(results, root/'curated-results')
    closure = root/'closure-source'
    closure.mkdir()
    for p in doc.iterdir():
        if p.is_file():
            shutil.copyfile(p, closure/p.name)
    files = sorted(v for v in root.rglob('*') if v.is_file() and '__pycache__' not in v.parts
                   and v.name != 'sealed-runtime.json')
    study.write(root/'sealed-runtime.json', {str(p.relative_to(root)): study.sha(p) for p in files})
    for name, digest in study.load(root/'sealed-runtime.json').items():
        assert study.sha(root/name) == digest
    print(json.dumps(dict(status='SEALED_AND_VERIFIED', files=len(files),
        private_bytes=sum(p.stat().st_size for p in files),
        seal_sha256=study.sha(root/'sealed-runtime.json'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--doc', type=Path, required=True)
    args = parser.parse_args()
    main(args.root.resolve(), args.doc.resolve())
