# ODE-FFN: Progress Report 3

**Working draft, 17 November 2026. Author: Zhizhe Zhang.**

[Read the report](report3.pdf). The main report is **three pages**, followed
by **four supporting pages** with tables, plots and a Ring8 case study.
The report remains a draft until the user finalizes it.

The central question is which part of ODE-inspired feature evolution helps:
operator sharing, the prescribed coefficients, or numerical solution accuracy.
The report also addresses weight/update rank, Ring8 rotation symmetry, and
the necessity of the ISRU response. It includes the completed common-bank
ordinary controls and parameter-matched CelebA results.

The layout follows the professor's readability feedback: connected main text,
two short equations, and references to supporting material rather than tables
embedded between paragraphs. The main text ends with the hypothesis proposed
for the next two weeks. No experiment was launched to prepare this document.
Reports 1 and 2 are unchanged.

## Source and build

- [LaTeX](report3.tex)
- [Evidence map](EVIDENCE.md) and [bound numeric data](data/evidence.json)
- [Table and figure generation](make_artifacts.py)
- [Verification](verification.json)
- [Earlier planning outline](outline.md)

From this directory, run `bash build.sh`. It regenerates the tables and two
vector plots from completed repository results, copies the two original case
figures unchanged, then compiles with latexmk, Tectonic or pdflatex. The plotting
step uses the existing Python NumPy/Matplotlib installation; it does not load a
model or run training, sampling or metrics on new data.
Run `python3 verify_report.py` for the source, numeric, page-count, font and
preserved-report checks on the development machine.

The source bindings preserve different evaluation cohorts. Read comparisons
within each table; the historical coefficient and sharing cohorts differ from
the latest ordinary-control banks. Dense forward matrix work is labeled
separately from total training FLOPs.
