"""Verify numerical tables and preserve the read-only coordinate check."""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main(root, doc):
    spec = importlib.util.spec_from_file_location('pcoordinate_closure_source', root/'source/study.py')
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    support, _, _, _ = study.parents()
    study.verify(root)
    assert support.sha(doc/'study.py') == support.sha(root/'source/study.py')
    actual = support.load(root/'doc-validation/records.json')
    assert actual['invocation_count'] == 1 and actual['mismatch_list'] == []
    assert actual['documented_command'] == actual['executed_command']
    assert actual['exit_code'] == actual['terminal_tool_result']['exit_code'] == 0
    assert actual['observed_sentinel'] is True
    assert actual['expected_sentinel'] == 'P_COORDINATE_REAL_STATES_PASS'
    assert 'P_COORDINATE_REAL_STATES_PASS' in actual['terminal_tool_result']['output']
    result = support.load(root/'output/result.json')
    assert result['status'] == 'PASS' and not result['new_training'] and not result['new_FID']
    assert [v['step'] for v in result['rows']] == list(study.STEPS)
    assert len(result['matrix_rows']) == 36 and not result['new_ODE_method']
    for row in result['rows']:
        assert row['native_prefix_rows_bitwise'] == 64 and row['literal32_prefix_bitwise']
        assert row['active_gradients_checked'] == 49
        for name in ('gradient_pullback_max_error', 'mapped_parameter_max_error',
                     'mapped_moment_max_error', 'post_prediction_max_error'):
            assert 0 <= row[name] < 1e-10
    out = doc/'results'
    out.mkdir()
    for name, rows in (('state-checks.csv', result['rows']), ('matrix-increments.csv', result['matrix_rows'])):
        with (out/name).open('w') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        values = list(csv.DictReader((out/name).open()))
        assert len(values) == len(rows)
        for expected, observed in zip(rows, values):
            for key, value in expected.items():
                if isinstance(value, bool):
                    assert observed[key] == str(value)
                elif value is None:
                    assert observed[key] == ''
                elif isinstance(value, (int, float)):
                    assert float(observed[key]) == value
                else:
                    assert observed[key] == value
    support.write(out/'result.json', result)
    support.write(out/'actual-tool-receipt.json', actual)
    support.write(out/'artifact-verification.json', dict(status='PASS', native_prefix_rows=384,
        active_gradient_tensors=294, state_rows=6, matrix_increment_rows=36,
        all_rows_preserved=True, native_float32_optimization_equivalence=False,
        new_training=False, new_FID=False, new_ODE_method=False))
    protected = ['src', 'synthetic', 'configs', 'scripts', 'reports/report1']
    repo = doc.parents[2]
    assert not subprocess.check_output(['git', '-C', str(repo), 'diff', '--name-only',
                                       '98730e0', '--', *protected], text=True)
    study.verify(root)
    assert not (root/'sealed-runtime.json').exists()
    shutil.copytree(out, root/'curated-results')
    closure = root/'closure-source'
    closure.mkdir()
    for file in doc.iterdir():
        if file.is_file():
            shutil.copyfile(file, closure/file.name)
    files = sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                   and p.name != 'sealed-runtime.json')
    support.write(root/'sealed-runtime.json', {str(p.relative_to(root)): support.sha(p) for p in files})
    for name, digest in support.load(root/'sealed-runtime.json').items():
        assert support.sha(root/name) == digest
    print(json.dumps(dict(status='SEALED_AND_VERIFIED', files=len(files),
        private_bytes=sum(p.stat().st_size for p in files),
        seal_sha256=support.sha(root/'sealed-runtime.json'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--doc', type=Path, required=True)
    args = parser.parse_args()
    main(args.root.resolve(), args.doc.resolve())
