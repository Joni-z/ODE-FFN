# Report 1 — ODE-FFN

[Read the PDF](report1.pdf) · [LaTeX source](report1.tex) · [Evidence map](EVIDENCE.md)

English academic progress report by **Zhizhe Zhang**, dated 20 October 2026.
The title block and single-column organization follow the user-supplied NeuroMotion
Report 1. The report is a draft for author review.

**The entire PDF is three pages, including its table and two figures.** It has
no bibliography or appendix. Body text remains 11 pt. The report ends with the
hypotheses for the next two weeks.

The text covers the original layer, synthetic setup, baseline and compute-budget
comparisons, frozen-order results, and the behavior of P. Detailed numerical
checks and additional plots remain separate files for readers who need them.


## Files and evidence

- `report1.tex`, `report1.pdf`: source and compiled report.
- `EVIDENCE.md`, `evidence_manifest.json`: claim locations, source hashes and limits.
- `make_figures.py`: regenerate earlier experiment figures, baseline table and manifest.
- `../../scripts/analyze_frozen_order.py`: regenerate follow-up figures and exact tables.
- `figures/09-*.pdf` through `13-*.pdf`: full order matrix, fixed-input accuracy,
  endpoint examples, operator geometry/training, and solver checks.
- `data/target_examples.csv`: first 1,000 actual test points per target.
- `data/ring8-frozen-examples.csv`: fixed-index Ring8 target/generated examples;
  original complete arrays are hashed in the frozen-order evidence.
- Earlier figures `01-*` through `08-*` remain available as detailed diagnostics.

The full [analysis bundle](../../docs/research-2026-10-20-followup/analysis-report.md)
contains all exact effects, claim boundaries, a statistical appendix and a figure
catalog. The [earlier stage report](../../docs/research-2026-10-20/2026-10-20--odelayer--r00--ablation-report.md)
retains the longer original analysis.

## Build

Compile the self-contained LaTeX folder with the included figures:

```bash
bash reports/report1/build.sh
```

Rebuild numerical figures from the repository's committed evidence with Python,
NumPy and Matplotlib, then compile:

```bash
python reports/report1/make_figures.py
python scripts/analyze_frozen_order.py
bash reports/report1/build.sh
```

The build script supports Tectonic, latexmk or pdflatex. Tectonic may download
missing TeX packages on first use. No GPU or checkpoints are needed to rebuild
the report; re-running the actual interventions requires the manifest-bound
checkpoints and data on torch/145. Raw datasets, checkpoint files and execution
archives stay outside Git. The PDF, reviewed figures and curated evidence are
versioned.
