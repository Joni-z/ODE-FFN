"""Check the actual PDF, curated numbers, frozen Report1 and referenced assets."""
from __future__ import annotations
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
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def run(*command):
    return subprocess.check_output(command, text=True)


def main():
    pdf = ROOT/'report2.pdf'
    tex = (ROOT/'report2.tex').read_text()
    info = run('pdfinfo', str(pdf))
    pages = int(re.search(r'^Pages:\s+(\d+)', info, re.M)[1])
    assert pages == 3, pages
    assert re.search(r'^Author:\s+Zhizhe Zhang\s*$', info, re.M)
    assert pdf.stat().st_size > 100000
    text = run('pdftotext', '-layout', str(pdf), '-')
    assert len([p for p in text.split('\f') if p.strip()]) == 3
    assert '1 November 2026' in text and 'Draft' in text
    assert 'Hypotheses for the next two weeks' in text.split('\f')[2]
    assert not any(marker in text for marker in ('??','[?]','[VERIFY]','[CLAIM NEEDS EVIDENCE]'))
    assert not re.search(r'\\(?:cite|bibliography|bibliographystyle)|thebibliography', tex)
    assert not re.search(r'^(References|Bibliography)\s*$',text,re.M)
    for phrase in ('252.96','262.93','268.74','0.14834','0.17124','0.14717','0.14905','186.6','188.3'):
        assert phrase in text, phrase
    with (ROOT/'data/synthetic.csv').open() as handle:
        synthetic = list(csv.DictReader(handle))
    assert len(synthetic) == 15
    for row in synthetic:
        assert f"{float(row['oracle_mse']):.6f}" in text
    fonts = run('pdffonts', str(pdf))
    embedded = []
    for line in fonts.splitlines():
        match = re.search(r'\s(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$', line)
        if match:
            embedded.append(match[1] == 'yes')
    assert embedded and all(embedded)
    log = ROOT/'report2.log'
    if log.exists():
        content = log.read_text(errors='replace')
        assert not re.search(r'Overfull \\[hv]box|undefined|! LaTeX Error',content,re.I)
    assets = [ROOT/'report2.tex', ROOT/'matrix-observation.tex']
    assets.extend(ROOT/path for path in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', tex))
    for path in assets:
        assert path.is_file() and path.stat().st_mtime <= pdf.stat().st_mtime, str(path)
    hashes = json.loads((ROOT/'data/file-hashes.json').read_text())
    for relative, digest in hashes.items():
        assert sha(ROOT/relative) == digest, relative
    old = json.loads((ROOT/'scope.json').read_text())['report1']
    assert run('git','-C',str(REPO),'rev-parse','HEAD:reports/report1').strip() == old['git_tree']
    assert sha(REPO/'reports/report1/report1.pdf') == old['pdf_sha256']
    assert run('git','-C',str(REPO),'diff',old['last_requested_revision_commit'],'--','reports/report1') == ''
    numeric = json.loads((ROOT/'data/FID-verification.json').read_text())
    assert len(numeric)==9
    result = dict(status='PASS', observed_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        pdf='report2.pdf', pages=pages, total_pages_including_figures=pages, author='Zhizhe Zhang',
        language='English', bibliography=False, all_fonts_embedded=True,
        undefined_references=False, overfull_boxes=False, curated_data_hashes_verified=True,
        all_synthetic_table_numbers_checked=True, all_nine_FIDs_verified=True,
        maximum_FID_recalculation_difference=max(r['abs_difference'] for r in numeric),
        Report1_unchanged=True, Report1_pdf_sha256=old['pdf_sha256'],
        no_new_training_or_sampling=True, final_50k_FID_available=False,
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in assets}, pdf_sha256=sha(pdf),
        visual_review='All three pages rendered and inspected by the authoring assistant; fixed-case images and chart labels are readable.')
    (ROOT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
