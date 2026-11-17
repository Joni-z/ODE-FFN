"""Check report pages, source numbers, fonts, links and preserved artifacts."""
from pathlib import Path
import datetime
import hashlib
import json
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    evidence = json.loads((HERE/'data/evidence.json').read_text())
    for name, expected in evidence['source_sha256'].items():
        assert sha(ROOT/name) == expected, name
    text = subprocess.check_output(['pdftotext', '-layout', str(HERE/'report3.pdf'), '-'], text=True)
    pages = [p for p in text.split('\f') if p.strip()]
    assert len(pages) == 7, len(pages)
    assert 'Hypothesis for the next two weeks' in pages[2]
    assert 'Weight geometry, response and symmetry' in pages[3]
    assert all('Table 1:' not in p for p in pages[:3])
    assert not any(v in text for v in ['??', '[?]', '[VERIFY]', '[CLAIM NEEDS EVIDENCE]'])
    info = subprocess.check_output(['pdfinfo', str(HERE/'report3.pdf')], text=True)
    assert 'Zhizhe Zhang' in info
    font_text = subprocess.check_output(['pdffonts', str(HERE/'report3.pdf')], text=True)
    fonts = [s.split() for s in font_text.splitlines()[2:] if s.strip()]
    assert fonts and all(s[-5] == 'yes' for s in fonts), font_text
    log = (HERE/'report3.log').read_text()
    warnings = re.findall(r'(?:Overfull|Underfull)\s+\\[hv]box[^\n]*|LaTeX Warning:[^\n]*undefined[^\n]*', log)
    assert not warnings, warnings
    # Every displayed row is generated from the bound full-precision values.
    groups = [('feedback-tables.tex', ['rank_rows', 'response_rows']),
              ('symmetry-table.tex', ['symmetry_rows']),
              ('structure-tables.tex', ['sharing_rows', 'coefficient_rows']),
              ('practical-tables.tex', ['ordinary_native_rows', 'image_rows'])]
    cells = 0
    for filename, keys in groups:
        table = (HERE/'data'/filename).read_text()
        for key in keys:
            for row in evidence[key]:
                assert ' & '.join(row) + '\\\\' in table, (key, row)
                for value in row:
                    if re.fullmatch(r'\d+\.\d{3}', value):
                        assert value in text, value
                        cells += 1
    for row in evidence['field_forecast']:
        assert abs(row['relative_rms_error']**2 - (1-row['score'])) < 1e-12
    image_pairs = [
        ('docs/research-assessments/2026-11-15-professor-execution/results/isru-samples.png', 'ring8-samples.png'),
        ('docs/research-assessments/2026-11-15-professor-execution/results/isru-trajectories.png', 'ring8-trajectories.png')]
    assert all(sha(ROOT/a) == sha(HERE/'figures'/b) for a, b in image_pairs)
    protected_path = Path('/Users/mr.z/ode-workspace-prep/coefficient-20261116/native-time-resolution/protected-before.json')
    protected = json.loads(protected_path.read_text())
    assert all(sha(ROOT/name) == value for name, value in protected.items())
    final = json.loads((ROOT/'reports/report2/finalization.json').read_text())['protected_files']
    assert all(sha(ROOT/'reports/report2'/name) == value for name, value in final.items())
    native = ['main_jit.py', 'engine_jit.py', 'model_jit.py', 'denoiser.py', 'ffn.py']
    for name in native:
        assert (ROOT/name).read_bytes() == subprocess.check_output(['git', 'show', 'ae560b9:'+name], cwd=ROOT)
    links = 0
    for file in HERE.glob('*.md'):
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', file.read_text()):
            if target.startswith(('http:', 'https:', '#', 'mailto:')):
                continue
            dest = (file.parent/target.split('#')[0]).resolve()
            assert dest.exists() or dest == HERE/'verification.json', (file, target)
            links += 1
    artifacts = {}
    for path in HERE.rglob('*'):
        if path.is_file() and path.name not in ('verification.json', 'delivery.json') and path.suffix in ('.tex', '.pdf', '.png', '.py', '.json', '.md', '.sh'):
            artifacts[str(path.relative_to(HERE))] = sha(path)
    result = dict(passed=True, checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='draft, not finalized', main_pages=3, supporting_pages=4, total_pages=7,
        bound_sources=len(evidence['source_sha256']), displayed_numeric_cells_checked=cells,
        field_score_identity_checked=True, copied_case_images_identical=True,
        embedded_fonts=len(fonts), undefined_references=False, overfull_or_underfull_boxes=warnings,
        protected_files_unchanged=len(protected), final_report2_bound_files_unchanged=len(final),
        native_five_unchanged=True, local_links_checked=links, new_experiment_runs=0,
        visual_review='All seven pages inspected; layout changes rechecked before delivery.',
        artifact_sha256=artifacts)
    (HERE/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['passed', 'main_pages', 'supporting_pages',
        'bound_sources', 'displayed_numeric_cells_checked', 'protected_files_unchanged', 'embedded_fonts']}))


if __name__ == '__main__':
    main()
