"""Verify Report2 source bindings, catalogue, PDF and frozen Report1.

This verifies reporting artifacts; it does not rerun training or claim fresh
independent reproduction of every archived experiment.
"""
from __future__ import annotations
import argparse
import csv
import datetime
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(*command):
    return subprocess.check_output(command, text=True)


def rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def selected(path, selector):
    found = [r for r in rows(path) if all(r[k] == v for k, v in selector.items())]
    assert len(found) == 1, (path, selector, len(found))
    return found[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-visual-review', action='store_true')
    args = parser.parse_args()
    pdf = ROOT/'report2.pdf'
    tex = (ROOT/'report2.tex').read_text()
    info = run('pdfinfo', str(pdf))
    pages = int(re.search(r'^Pages:\s+(\d+)', info, re.M)[1])
    assert pages == 3, pages
    assert re.search(r'^Author:\s+Zhizhe Zhang\s*$', info, re.M)
    text = run('pdftotext', '-layout', str(pdf), '-')
    assert len([p for p in text.split('\f') if p.strip()]) == 3
    assert '3 November 2026' in text and 'Draft' in text
    assert all(heading in text for heading in ['Local dynamics: clocks, order and optimization','Changing the complete field and its directions','Input-dependent features and current evidence'])
    assert 'professor' not in text.lower() and 'suggestion' not in text.lower()
    assert all(topic in tex for topic in ['Clocks and order.','Full-gradient readout.','Input-dependent directions.','Nonlinear internal flow.','local-teacher','Muon','PD(h)'])
    assert 'PD(h)' in text.replace(' ', '')
    assert 'Hypotheses for the next two weeks' in text.split('\f')[2]
    assert not any(marker in text for marker in ('??', '[?]', '[VERIFY]', '[CLAIM NEEDS EVIDENCE]'))
    assert not re.search(r'\\(?:cite|bibliography|bibliographystyle)|thebibliography', tex)
    assert not re.search(r'^(References|Bibliography)\s*$', text, re.M)

    # Every extracted chart/table value must identify one exact source row.
    source_hashes = json.loads((ROOT/'data/ode-mainline-sources.json').read_text())
    for relative, digest in source_hashes.items():
        assert sha(REPO/relative) == digest, relative
    values = rows(ROOT/'data/ode-mainline.csv')
    for r in values:
        original = selected(REPO/r['source'], json.loads(r['selector']))
        assert float(r['value']) == float(original[r['metric']]), r
    checked_numbers = []

    def present(value, digits):
        printed = f'{float(value):.{digits}f}'
        assert printed in text, printed
        checked_numbers.append(printed)

    # Retained quantitative positive cases, now compressed for the new focus.
    for study,dataset,arm,metric,digits in [
        ('readout','rotated_product_bimodal_16d','ode_k2_direct','oracle_mse',6),
        ('readout','rotated_product_bimodal_16d','ode_k2_gradient','oracle_mse',6),
        ('readout','rotated_product_bimodal_16d','swiglu192_gradient','oracle_mse',6),
        ('frame','thin','frame_k2','oracle_mse',6),
        ('frame','thin','frame_k1','oracle_mse',6),
        ('frame','thin','swiglu192','oracle_mse',6),
        ('frame','thick','frame_k2','tangent_sw2',4),
        ('frame','thick','swiglu192','tangent_sw2',4),
        ('cubic','rotated_product_bimodal_16d','cubic_split','oracle_mse',6),
        ('cubic','rotated_product_bimodal_16d','free_isru','oracle_mse',6),
        ('cubic','rotated_product_bimodal_16d','swiglu160','oracle_mse',6),
    ]:
        matching=[r for r in values if (r['study'],r['dataset'],r['arm'],r['metric'])==(study,dataset,arm,metric)]
        assert len(matching)==1
        present(matching[0]['value'],digits)
    case=json.loads((REPO/'docs/research-2026-10-21-gates/results/ring8-case-study.json').read_text())
    for row in case:
        if row['model'] in ['ode_fixed','ode_original','swiglu']:present(100*row['off_mode_fraction'],1)
    professor=json.loads((ROOT/'data/professor-evidence.json').read_text())
    for file,digest in professor['source_hashes'].items():assert sha(REPO/file)==digest,file
    for record in professor['records']:
        original=selected(REPO/record['source'],record['selector'])
        assert float(original[record['metric']])==record['value']
        if record['printed_digits'] is not None:
            value=record['value']*record.get('printed_scale',1.)
            printed=f"{value:.{record['printed_digits']}f}"
            assert printed in text or (printed.startswith('0.') and printed[1:] in text),printed
            checked_numbers.append(printed)
    native=json.loads((REPO/'docs/autoresearch/professor-20261103/readiness/cycle13/cpu.json').read_text())
    assert native['passed']
    for family in ['pd_isru_k1','pd_constant_k1','free_isru','swiglu64']:
        row=next(r for r in native['rows'] if r['dim']==16 and r['family']==family)
        assert f"{row['parameters']:,}" in text
    new_results=json.loads((REPO/'docs/autoresearch/professor-20261103/cycle13-results/verification.json').read_text())
    assert new_results['endpoints']==18 and new_results['exact_native_anchors']==10
    assert new_results['max_generated_error']==0 and new_results['continuation_eligible']==0

    inventory = json.loads((ROOT/'attempt-inventory.json').read_text())
    ledger = json.loads((ROOT/'attempt-ledger.json').read_text())
    records = ledger['records']
    areas = {r['area'] for r in inventory['areas']}
    assert areas <= {r['area'] for r in records}
    assert len(areas) == ledger['coverage_areas']
    assert len({r['id'] for r in records}) == len(records)
    catalogue = (ROOT/'attempts.md').read_text()
    for r in records:
        assert catalogue.count('| '+r['id']+' ·') == 1
        if r['source'].startswith('https://'):
            assert r['kind'] == 'legacy configuration inventory'
            legacy = run('git','-C',str(REPO),'ls-tree','-r','--name-only',
                         'research-archive-20261101','--','archive/imagenet/configs').splitlines()
            assert sorted(legacy) == sorted(r['legacy_files'])
        else:
            assert sha(REPO/r['source']) == r['source_sha256'], r['id']
        assert all((REPO/p).is_file() for p in r['tables']), r['id']
    csv_ledger = rows(ROOT/'attempt-ledger.csv')
    assert len(csv_ledger) == len(records)
    for original, rendered in zip(records, csv_ledger):
        assert all(original[k] == v for k, v in rendered.items())

    for path in [ROOT/'README.md',ROOT/'EVIDENCE.md',ROOT/'evidence-index.md',ROOT/'attempts.md',
                 ROOT/'analysis/analysis-report.md',ROOT/'analysis/figure-catalog.md']:
        for link in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if '://' not in link and not link.startswith('#'):
                assert (path.parent/link.split('#')[0]).exists(), (path,link)
    fonts = run('pdffonts', str(pdf))
    embedded = []
    for line in fonts.splitlines():
        match = re.search(r'\s(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$', line)
        if match:
            embedded.append(match[1] == 'yes')
    assert embedded and all(embedded)
    content = (ROOT/'report2.log').read_text(errors='replace')
    assert not re.search(r'Overfull \\[hv]box|undefined|! LaTeX Error',content,re.I)
    assets = [ROOT/'report2.tex']
    assets.extend(ROOT/path for path in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',tex))
    for path in assets:
        assert path.is_file() and path.stat().st_mtime <= pdf.stat().st_mtime, path
    for relative,digest in json.loads((ROOT/'data/file-hashes.json').read_text()).items():
        assert sha(ROOT/relative) == digest, relative
    old = json.loads((ROOT/'scope.json').read_text())['report1']
    assert run('git','-C',str(REPO),'rev-parse','HEAD:reports/report1').strip() == old['git_tree']
    assert sha(REPO/'reports/report1/report1.pdf') == old['pdf_sha256']
    assert run('git','-C',str(REPO),'diff',old['last_requested_revision_commit'],'--','reports/report1') == ''
    revision=json.loads((ROOT/'data/report-revision.json').read_text())
    assert run('git','-C',str(REPO),'diff',revision['baseline_commit'],'--',*revision['protected_paths']) == ''
    visual_path = ROOT/'visual-review.json'
    visual = json.loads(visual_path.read_text()) if visual_path.exists() else {}
    visual_ok = visual.get('pdf_sha256') == sha(pdf) and visual.get('pages_reviewed') == [1,2,3]
    if args.require_visual_review:
        assert visual_ok, 'Render and inspect the current PDF before renewing the receipt.'
    result = dict(status='PASS' if visual_ok else 'LAYOUT_PASS_VISUAL_REVIEW_PENDING',
        observed_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        pages=pages,total_pages_including_figures=pages,author='Zhizhe Zhang',language='English',
        bibliography=False,all_fonts_embedded=True,undefined_references=False,overfull_boxes=False,
        source_bound_values=len(values)+len(professor['records']),printed_numeric_values_checked=checked_numbers,
        research_coverage=['clock/P/order controls','Muon/spectrum','full-gradient readout','conditional frames','nonlinear cubic flow','diagonal features','ordinary controls and diagnostics'],
        attribution_style_headings_absent=True,
        current_report_date='3 November 2026',report_revision_baseline=revision['baseline_commit'],
        documentation_areas_covered=len(areas),ledger_records=len(records),
        ledger_is_independent_run_count=False,ledger_source_hashes_verified=True,
        old_evidence_snapshot_unchanged=True,primary_experiment_docs_unchanged=True,
        Report1_unchanged=True,Report1_pdf_sha256=old['pdf_sha256'],
        no_new_training_or_sampling=True,all_historical_experiments_rerun=False,
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in assets+[ROOT/'data/professor-evidence.json',ROOT/'data/report-revision.json']},pdf_sha256=sha(pdf),
        visual_review_receipt_valid=visual_ok)
    (ROOT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
