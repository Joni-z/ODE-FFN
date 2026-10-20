# Report 1 — ODE-FFN

[Read the PDF](report1.pdf) · [LaTeX source](report1.tex) · [Evidence map](EVIDENCE.md)

An English academic progress report by **Zhizhe Zhang**, dated 20 October 2026. The title block, compact single-column layout, and experiment narrative follow the user-supplied NeuroMotion Report 1.

The compiled PDF has **three pages of main text, including the key figures and comparison table**, followed by one page of references. The main text ends with hypotheses, tests, and expected outcomes for the next two weeks. Evidence provenance is documented separately in `EVIDENCE.md`. No experiment was launched to prepare this report.

The report covers the original ODELayer, the synthetic protocol, spectral caps, SiLU/SwiGLU comparisons, compute budgets, input/time gating, Taylor order, and the training trajectories of P. Two figures contain six panels; a Ring8 case study connects the diagnostics to generation quality. The six detailed plots from the earlier layout remain available in `figures/`. Observations are separated from causal hypotheses and proposed work.

## Files

| File | Purpose |
|---|---|
| `report1.pdf` | Compiled report: main text and figures on pages 1–3; references on page 4 |
| `report1.tex` | Standalone LaTeX document with inline bibliography |
| `comparison_table.tex` | Data-generated comparison table |
| `figures/07-main-ablations.pdf`, `figures/08-main-trajectories.pdf` | Compact figures included in the document |
| `figures/01-*.pdf` through `figures/06-*.pdf` | Additional full-size diagnostic plots and dataset examples |
| `make_figures.py` | Rebuild figures and the table from the repository's committed experimental evidence |
| `data/report_values.json` | Numeric figure data, including gate differences and P trajectories |
| `data/target_examples.csv` | First 1,000 held-out target points per dataset, exported from the actual torch v1 test splits |
| `evidence_manifest.json` | Source paths, SHA256 hashes, and evidence revision |
| `EVIDENCE.md` | Claim-to-source map and limits of interpretation |
| `build.sh` | Compile with Tectonic, latexmk, or pdflatex |

## Build

To compile the document using the included figures:

```bash
cd reports/report1
bash build.sh
```

The folder is self-contained for LaTeX compilation. Tectonic may download missing TeX packages on its first run. A standard LaTeX installation can also run `pdflatex report1.tex` twice.

To regenerate the figures and table, retain the surrounding repository and use Python with NumPy and Matplotlib installed:

```bash
python reports/report1/make_figures.py
bash reports/report1/build.sh
```

The figure script reads existing results and performs no model training. LaTeX intermediates and preview images are ignored by Git; the compiled PDF and figures are versioned.

## Earlier internal report

The longer Chinese stage analysis remains at
[`docs/research-2026-10-20/2026-10-20--odelayer--r00--ablation-report.md`](../../docs/research-2026-10-20/2026-10-20--odelayer--r00--ablation-report.md).
Report 1 distills that stage together with the earlier spectral-cap evidence into the requested short academic format.
