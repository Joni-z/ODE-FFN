# Report2 — ODEFFN mainline

**Zhizhe Zhang · 3 November 2026 · Draft for review**

[Three-page PDF](report2.pdf) · [LaTeX](report2.tex) ·
[Evidence map](EVIDENCE.md) · [Verification](verification.json) ·
[Current research index](evidence-index.md)

The professor's two suggestions organize the report:

- **Muon and the spectrum.** Show how update stable rank and learned P evolve,
  compare field fitting with ordinary controls, and explain what the direct
  spectral-constraint tests establish.
- **Alternative P D(h) designs.** Separate learned controllers, simple input
  diagonals, order and growth controls. Present the complete latest ISRU-D,
  ordinary free-ISRU and SwiGLU comparison, retaining both optimizer settings.

The main distinction is a useful feature response versus an ODE-specific gain.
The report retains the earlier gradient-readout and conditional-frame positive
cases, with their ordinary controls and transfer limits. New20k results are
kept separate from earlier5k and thin/thick protocols. No image-quality claim
follows from these vector-network studies.

The entire PDF is **three pages including both figures and tables**, English,
sole author Zhizhe Zhang, no bibliography. It opens with project/progress and
ends with two hypotheses for the next two weeks. The reporting revision starts
from945a380; the prior November2PDF remains in Git. No new training or sampling
was performed for this edit.

## Evidence and historical coverage

[Professor evidence](data/professor-evidence.json) binds each new figure/table
value to an exact row and file hash. [EVIDENCE.md](EVIDENCE.md) states the
claim boundaries; [evidence-index.md](evidence-index.md) links the completed
professor-guided studies. Curves, samples and all optimizer outcomes remain
in the primary study directories.

The historical [attempt ledger](attempts.md), also in
[JSON](attempt-ledger.json)/[CSV](attempt-ledger.csv), contains127linked
records from the November2inventory. These are evidence families, not
independent training runs or a new inventory of the later professor cycles.
The current index supplies that subsequent coverage. The Chinese
[meeting brief](meeting-brief.md) is the historical November2briefing.

Report1 remains final at994e660921d2885592704f5c471bd43a12b1c6ef. Its files,
primary experiment documents, trainers, data and checkpoints are unchanged.
The reporting scripts verify the final Report1 identity and the protected
source paths recorded in [revision metadata](data/report-revision.json).

## Rebuild

From the repository root:

```sh
python3 -B reports/report2/make_professor_figures.py
bash reports/report2/build.sh
python3 -B reports/report2/verify_report.py
```

The figure builder uses tracked CSVs, NumPy and Matplotlib. Compilation uses
latexmk, Tectonic or pdflatex; verification uses pdfinfo, pdftotext and pdffonts.
A fresh PDF must be rendered and inspected before its visual-review receipt
is renewed and `verify_report.py --require-visual-review` can pass.

`make_mainline_figures.py` preserves the earlier quantitative evidence and
November2figure. `collect_evidence.py`/`make_figures.py` reproduce the superseded
November1image-centered snapshot. They are not needed to build the current PDF.

## Report delivery to MBP

After completing a report revision, publish it to GitHub and fast-forward the
MBP checkout at `/Users/zzz/Documents/NYU/Researches/ODE FFN/ODE-FFN`.
Check for local changes before pulling; preserve them. Verify the delivered
PDF SHA-256 against the reviewed source copy. A GitHub push alone is not the
completed report delivery.
