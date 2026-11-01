# Report2 continuity notes

## Draft — 1 November 2026
The user requested effectiveness and a Report2 draft. The live CIFAR process
continues beyond40k/50k with GPU99%; no restart, recipe change, new fit,
sampling, or feature extraction was performed for this draft.
`collect_evidence.py` completed with exit0 (91228/67900d), checking frozen
bindings, generated/feature hashes, all nine completed FIDs via an independent
low-rank formula, and cold/25k matrix statistics from actual tensors.
The report snapshot fixes validation at40k and FID at25k.

The25k EMA FIDs252.96/262.93/268.74 for ISRU/matched/wide are still poor.
The corresponding raw grids are more textured than EMA; raw FID is not known.
ISRU raw/EMA loss0.14834/0.17124 at25k becomes0.14717/0.14905 at40k.
The long EMA retains8.2% cold parameters at25k, a useful diagnostic lead,
not proof that EMA explains all FID. Preserve the registered results.
Mean W1/W2 spectral maxima grow1.27→1.84/1.60 with high entropy ranks;
this does not demonstrate a spectral cause or justify a new penalty.

The English draft is exactly three pages including figures/table, sole author
Zhizhe Zhang. It retains all five synthetic tasks and both ordinary budgets,
the input-frame thin-data improvement and K1/thick limitations, and ends
with two prospective hypotheses. No references or seed-instability caveat.
Report1 is byte-identical. Final endpoint and extra raw/EMA evaluations remain
future work, not claimed results.

Build notes: initial cached-only Tectonic compilation lacked Latin Modern8/6
font resources; normal Tectonic fetched these and succeeded, without changing
training environments. The first layout spilled each planned page; content
was shortened to three pages at unchanged11pt/margins. A long inline/model
name caused an overfull line and was rephrased. Final compilation is clean.
One combined text patch failed on whitespace; targeted patches corrected it.
The broad compiler-path search hit protected Library directories; the existing
Homebrew Tectonic was sufficient and no protected path was modified.

## User instruction — October 29, 2026
The last requested Report1 revision finalized Report1. All work after that
point belongs to Report2. Do not interpret routine preservation of Report1
as an instruction to keep extending it. Maintain this separate report and
its Markdown work record as research progresses.

## Git boundary
The last commit touching `reports/report1/` is
`994e660921d2885592704f5c471bd43a12b1c6ef`, dated October 20, 2026,
23:54:44 Asia/Shanghai. The finalized directory tree is
`17e7ffd2f758c705379e1192e0dd3773da5ceb23`, equal to the current committed
Report1 tree when this scope was established. The exact PDF hash and evidence
inventory snapshot are in [scope.json](scope.json).

## Current research feeding Report2
The user wants to pursue the positive ISRU synthetic design. Its canonical
[research note](../../docs/autoresearch/isru-followup/README.md),
[tasks](../../docs/autoresearch/isru-followup/task_plan.md) and
[notes](../../docs/autoresearch/isru-followup/notes.md) carry operational detail.
Keep that positive result visible. Failed transfer narrows its scope, while
training adequacy and attribution still need assessment.

The image screen's ordinary baseline has not been shown to be a competent,
converged generator. Its numbers remain recorded, but they do not establish
the intended architecture claim. Include this correction in Report2 rather
than retrospectively changing Report1 or presenting the image screen as a win.

## Drafting discipline
Report2 should explain what was learned and what remains unresolved. Use
selected experiments and cases to support the argument. Large inventories, execution counts and reproducibility receipts belong in supporting records.
Do not present proposed ISRU follow-ups as completed work. No new training
or Report1 rebuild was launched while creating this reporting boundary.
