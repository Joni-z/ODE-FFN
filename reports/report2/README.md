# Report2 — ODEFFN mainline

**Zhizhe Zhang · 2 November 2026 · Draft for review**

[Three-page PDF](report2.pdf) · [LaTeX](report2.tex) ·
[Complete attempt ledger](attempts.md) · [Evidence map](EVIDENCE.md) ·
[Verification](verification.json)

The report follows the professor's original ODELayer: two input-clocked
Taylor maps around SiLU. It asks what the clock, generator P and polynomial
coupling actually contribute. Free ISRU is retained as a useful control and
a conditional field-fitting result, rather than credited as proof that the
original ODE structure works.

The main evidence is:

- P-only constraints and more accurate internal integration do not reliably
  improve generated distributions. Gate derivatives matter to the full layer.
- A complete gradient readout substantially improves the original ODE's
  Product16 field fitting, but ordinary backbones also benefit.
- Input-dependent orthogonal frames improve thin-manifold fitting. K1 nearly
  matches K2, and the sampling advantage fails on the thick distribution.
- Local response, teacher, label-noise and head-refit studies narrow several
  explanations for the remaining gap without proving an expressivity limit.
- Existing image runs do not meet the project's baseline-quality requirement.
  Neither poor FID rankings nor implementation checks establish effectiveness.

The entire PDF is **three pages including its table and figure**, with no
bibliography. It starts with the project and progress, and ends with two
prospective hypotheses. These are proposals for author discussion, not new
training launches or an automatic reopening of failed protocols.

## Complete history and evidence boundary

[attempts.md](attempts.md) organizes all 116 documentation areas inventoried
at `5c48a97`, additional architecture subquestions, and the early ImageNet
configuration inventory into 127 linked records. These are evidence families,
not 127 independent experiments. Some reuse the same controls; others are
proofs, diagnostics, proposals or implementation checks. The six archived
ImageNet configurations are not counted as completed runs without recovered
result bindings. Earlier Report1 evidence is labeled as context.

The catalogue is also available as [JSON](attempt-ledger.json) and
[CSV](attempt-ledger.csv). [attempt-inventory.json](attempt-inventory.json)
records its coverage; the [evidence index](evidence-index.md) gives a shorter
reading route. Primary records remain intact. Selected report/chart values
are checked directly against their source tables; this reporting pass does
not rerun every archived model. The earlier
[synthetic replay audit](../../docs/synthetic-audit-2026-11-01/EXPERIMENT_AUDIT.md)
is separately identified.

Report1 remains final at `994e660921d2885592704f5c471bd43a12b1c6ef`.
[scope.json](scope.json) retains its tree/PDF identities. No Report1 file,
experiment source, raw dataset, checkpoint or trainer was changed for this
rewrite. Historical image data and figures remain here for traceability, but
are not the current report's main narrative or live-process status.

## Rebuild

From the repository root:

```sh
python3 -B reports/report2/render_attempts.py
python3 -B reports/report2/make_mainline_figures.py
bash reports/report2/build.sh
python3 -B reports/report2/verify_report.py
```

The figure builder needs NumPy and Matplotlib and reads only tracked CSVs.
Compilation uses latexmk, Tectonic or pdflatex. Verification uses `pdfinfo`,
`pdftotext` and `pdffonts`, and checks source-row bindings and Report1 identity.
A visual-review receipt is valid only for its recorded PDF hash; a fresh PDF
must be rendered and inspected again before that receipt is renewed.

`collect_evidence.py` and `make_figures.py` belong to the **superseded November 1
image-centered draft**. They are preserved for reproducing that snapshot;
they are not required to rebuild the current report. There is no need to
access private checkpoints or remote machines for the current build.
