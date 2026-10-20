# Report 1 — ODE-FFN

[Read the PDF](report1.pdf) · [LaTeX source](report1.tex) · [Evidence map](EVIDENCE.md)

An English academic progress report dated 20 October 2026, prepared as a draft for discussion.

The compiled PDF has **three pages of main text**, one page containing only references, and three pages of supplementary visualizations. The main text ends with hypotheses proposed for the next two weeks. Evidence provenance is documented separately in `EVIDENCE.md`. No experiment was launched to prepare this report.

The report covers the original ODELayer, the synthetic protocol, spectral caps, SiLU/SwiGLU comparisons, compute budgets, input/time gating, Taylor order, and the training trajectories of P. It includes six graphical displays and a Ring8 case study. Observations are separated from causal hypotheses and proposed work.

## Files

| File | Purpose |
|---|---|
| `report1.pdf` | Compiled report: main text on pages 1–3, references on page 4, visualizations on pages 5–7 |
| `report1.tex` | Standalone LaTeX document with inline bibliography |
| `comparison_table.tex` | Data-generated comparison table |
| `figures/*.pdf` | Six figures included in the document |
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
