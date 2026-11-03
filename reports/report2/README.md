# Report2 — ODEFFN mainline

**Zhizhe Zhang · 4 November 2026**

**Editable draft.** The user has not submitted Report2 and will explicitly
decide when it is final. A completed revision remains open to further changes.

[Three-page PDF](report2.pdf) · [LaTeX](report2.tex) ·
[Evidence map](EVIDENCE.md) · [Verification](verification.json) ·
[Current research index](evidence-index.md)

The report follows the project's research questions rather than the origin of
individual ideas:

- **Local dynamics:** the original clock, P, polynomial order, static/free maps,
  state-feedback variants and Muon/spectral controls.
- **Whole-field computation:** complete-gradient readout, learned orthogonal
  directions, thin/thick transfer, nonlinear cubic flow and error-location probes.
- **Input-dependent features:** controller/range/order controls, direct and
  radial diagonals, the ISRU-D lead and strong ordinary references.
- **Checkpoint diagnosis:** centered-residual distributions, mode-sector
  probabilities, common transport paths and boundary-flux response.

Each main experiment states its purpose, observation and interpretation.
Two figures show optimizer/P trajectories and the new Ring8 shape/mass
checkpoint comparison. The current table reports field fitting,
common-bank sampling and a separate pooled-sample precision check, including
same-AdamW and selected-Muon SwiGLU controls. Product16 now uses the completed
eight-bank Heun128 comparison. Readout/frame results are compressed into prose
to make room for the cubic-to-ISRU construction and transport diagnosis.

The entire PDF is **three pages including figures and tables**, English,
sole author Zhizhe Zhang, no bibliography. It opens with project/progress and
ends with two hypotheses for the next two weeks. The previous focused version
remains in Git at23e2469; the preceding full-project report is at3285bdc.
The shape/mass result motivates a first-order sector-flux derivation and two
testable hypotheses. This revision uses completed evidence; it runs no new
experiments. Report1 and primary artifacts remain unchanged. The previous
Nov4 PDF/source remain in Git at `5075b3b`.

## Evidence and historical coverage

[Optimizer/diagonal evidence](data/professor-evidence.json) binds each new figure/table
value to an exact row and file hash. [Ring8 bindings](data/validation-evidence.json)
cover the Ring8 table, archived checkpoint curves and matrix-work counts.
[Transport bindings](data/transport-evidence.json) cover the current checkpoint
figure, complete Product16 results and oracle-switch example.
[EVIDENCE.md](EVIDENCE.md) maps report statements to their sources;
[evidence-index.md](evidence-index.md) links the completed studies.

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
python3 -B reports/report2/make_validation_figure.py
python3 -B reports/report2/make_transport_figure.py
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
