# ODE-FFN: Progress Report 3

**Finalized by the user on 18 November 2026. Author: Zhizhe Zhang.**

[Read the report](report3.pdf). The main report is **three pages**, followed
by **four supporting pages** with tables, plots and a Ring8 case study.
The [finalization record](finalization.json) binds the final files. Preserve the
PDF, TeX, figures and evidence; subsequent work belongs to Report4.

The central question is which part of ODE-inspired feature evolution helps:
operator sharing, the prescribed coefficients, or numerical solution accuracy.
The report also addresses weight/update rank, Ring8 rotation symmetry, and
the necessity of the ISRU response. It includes the completed common-bank
ordinary controls and parameter-matched CelebA results.

The main result favors shared nonlinear iteration as the leading architectural
hypothesis: sharing helps the tested stage comparisons, while prescribed
coefficients and increased internal solution accuracy give no consistent gain.
The final-stage proposal compares midpoint, a direct tied residual block,
Free ISRU and SwiGLU at matched training computation on both synthetic tasks.
The 20k midpoint models set the budget. Ordinary FFNs keep their current widths
and training settings and continue until total training FLOPs match; the tied
residual control trains to the same budget. Widening is a separate question.
The result will determine whether to retain midpoint, simplify to recurrence,
or conclude that the present gain requires more computation. It is a proposal,
not a launched study.

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
- [Finalization](finalization.json)
- [Earlier planning outline](outline.md)

For reproduction in a separate copy, `bash build.sh` regenerates the tables and two
vector plots from completed repository results, copies the two original case
figures unchanged, then compiles with latexmk, Tectonic or pdflatex. The plotting
step uses the existing Python NumPy/Matplotlib installation; it does not load a
model or run training, sampling or metrics on new data.
The final directory must not be rebuilt. The retained verification and delivery
JSON files document the preceding draft checks; `finalization.json` records the
subsequent user approval. `verify_report.py` writes a new verification receipt,
so run it only in a separate copy.

The source bindings preserve different evaluation cohorts. Read comparisons
within each table; the historical coefficient and sharing cohorts differ from
the latest ordinary-control banks. Dense forward matrix work is labeled
separately from total training FLOPs.
