"""Render the curated JSON ledger as Markdown and CSV without changing its evidence."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INTRO = '''# ODEFFN attempt ledger

Zhizhe Zhang · 2 November 2026

This is the complete evidence map supporting the short Report2, not a leaderboard.
It covers every area in [attempt-inventory.json](attempt-inventory.json), additional
root-level architecture studies, and the legacy ImageNet configuration inventory.
Entries are questions/evidence families, not independent training runs; some reuse
identical controls or describe the same study at a different analytical level.

The mainline is the professor's two Taylor-flow maps around SiLU. Ordinary
SiLU/SwiGLU and free ISRU are controls or derivative leads. A successful ordinary
control is not credited as an ODE-specific result. Training, frozen interventions,
teacher regression, proof fixtures, proposals and engineering are labeled separately.

Earlier Report1 material is included only as context. Report1 remains final.
All small-image rankings retain the later baseline correction: no architecture
quality verdict follows before a competent ordinary baseline. Historical stop
decisions describe their exact screens, not impossibility of an entire direction.
The source links preserve exact settings, metrics, negative outcomes and limits.

Machine-readable versions: [JSON](attempt-ledger.json), [CSV](attempt-ledger.csv).
The [three-page report](report2.pdf) selects evidence that changes the scientific
argument. No training was launched to prepare this ledger.
'''
ENDING = '''
## What is not claimed

The ledger does not turn a configuration into a completed run, an audit PASS into
model effectiveness, or a fixed-budget failure into an approximation theorem.
The six early ImageNet configurations have no newly recovered completed-run
binding here. Existing protocols/results are preserved; primary-source hashes
are recorded. Selected main-report values are checked against their tables in
this reporting pass. The November1 synthetic checkpoint replay is earlier,
separate evidence. Other entries index the preserved analyses without claiming
that every experiment was independently rerun for this report.

## Decision

Keep the research question centered on whether a specific ODE constraint earns
its cost against an ordinary realization. Input clocks can help mode separation,
complete adjoints can improve the original predictor, and conditional operator
directions can help thin data. None alone establishes a general advantage of the
original Taylor coupling. Future hypotheses require a prospective controlled
comparison, not reinterpretation of an old failed criterion.
'''


def main():
    records = json.loads((ROOT/'attempt-ledger.json').read_text())['records']
    fields = ('id','group','kind','period','purpose','observation','claim_limit','source')
    with (ROOT/'attempt-ledger.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    sections = [INTRO]
    for group in sorted(set(r['group'] for r in records)):
        sections.append(f'\n## {group}\n\n| ID / evidence type | Purpose and change | Observation and limit | Primary evidence |\n|---|---|---|---|\n')
        for r in records:
            if r['group'] != group:
                continue
            source = r['source'] if r['source'].startswith('https://') else '../../' + r['source']
            clean = lambda s: s.replace('|', '\\|').replace('\n', ' ')
            sections.append(f"| {r['id']} · {clean(r['kind'])} | {clean(r['purpose'])} | {clean(r['observation'])} | [source]({source}) |\n")
    (ROOT/'attempts.md').write_text(''.join(sections) + ENDING)
    print(f'Rendered {len(records)} evidence records.')


if __name__ == '__main__':
    main()
