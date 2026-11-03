# Narrative correction,3November

User clarified that Report2 covers the full project exploration. The previous
revision overemphasized the two latest directions. Restore the substantive
readout, conditional-frame, nonlinear-flow and diagnostic work; integrate
Muon/diagonal tests as normal research sections. Remove professor-attribution
headings and wording from the formal report. Keep strong ordinary controls,
three-page limit and mandatory MBP delivery.

# Professor directions update, 3 November 2026

User requires both professor suggestions in the formal report: Muon's effect
on the spectrum and alternative diagonal parameterizations of P D(h). Use
completed primary tables, including the ordinary free-ISRU control, rather
than only adding a future-work sentence. Compress older detail to preserve
the three-page limit; no bibliography. No new model runs or external claims.

# Report2 continuity notes

## Professor briefing — 2 November 2026

Added `meeting-brief.md` in Chinese for the user's imminent professor meeting.
It organizes attempts by P/spectral control, clock versus polynomial tying,
internal integration accuracy, complete-gradient readout, conditional frames,
and cubic/free-ISRU response. Main distinctions checked: spectral cap is not
spectral flattening; fixed-gate P/B comparison has equal initial function and
parameter count but different map sets/coordinates; vector clocks have20k
follow-ups; gradient readout includes inference derivative cost; conditional
frames preserve a given P's spectrum, not spectra across trained models;
free ISRU also changes the clock, so it is not a pure untying ablation.
The main report, raw experiments and training remain unchanged. Anticipated
questions cover model identity, convergence, oracle labels, time variables,
metric disagreement and the scope of an ODE-specific contribution.
Validation passed for all briefing source links,19 selected CSV values and
five mode fractions; the current Report2 PDF hash remains unchanged.

## ODEFFN mainline rewrite — 2 November 2026

The user explicitly restored the original ODEFFN investigation as the mainline
and requested all attempts in Report2. Replaced the image/ISRU-centered draft
with the original P/clock/Taylor question, complete-gradient readout and
conditional-frame evidence, local response diagnoses and objective-level math.
Free ISRU remains a useful control, with independent matrices and target-law
reversals explicit. No image ranking is credited as effectiveness.

The full catalogue has127 records covering116 preserved documentation areas,
additional architecture subquestions and legacy config presence. These counts
are coverage bookkeeping, not independent experiments or a progress claim.
Each record labels evidence type and links a hashed primary source. The old
ImageNet configs do not have newly recovered completed-run bindings.

The report uses the existing11pt/.8in layout. Initial page1 taxonomy table
spilled onto a fourth page; replacing that table with concise prose restored
three pages without reducing the font. Current chart source values are
extracted from three tracked CSVs. The former draft remains in Git at5c48a97;
old data/figures/scripts are preserved and explicitly marked historical.

Report1, experiment sources, raw artifacts and trainers are untouched. The
report hypotheses are prospective, not training authorization. The preceding
November1 synthetic replay remains earlier evidence; this pass checks report
provenance, values and presentation rather than rerunning the experiments.

Verification completed:186 source-bound chart/table values,36 printed numeric values, all127 catalogue records/source links, embedded PDF fonts and three rendered pages. The report has no overfull boxes or unresolved references. The Report1 tree/PDF and preserved image evidence hashes still match.

All entries below are historical snapshots; their process-status statements
were not refreshed by this reporting task.


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

## Historical ISRU focus — superseded by the November 2 instruction
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

## Revision checks
Initial layout exceeded three pages (6then4); shortened prose rather than shrinking body text. Third compilation is exactly3pages at11pt. The first extended verification treated the after-build revision metadata as a LaTeX input and failed its mtime check. Metadata now remains hash-bound but is excluded from compile-input timestamps; actual tex/graphics retain the stale-input check. No source/result value changed.

Visual QA found the initial Muon figure fixed its upper y limit after the first series, clipping the second. Removed that per-series limit and applied only a zero lower bound after both series. Numeric source bindings were already correct. Table numeric columns now have explicit spacing and leading zeros. Regenerate/recompile/review before publication.

Final visual review: all three current pages inspected after the Muon-axis and table-spacing corrections. Both Muon curves are fully visible; numeric columns are distinct. Body remains11pt; main text ends with the two-week hypotheses on page3. No clipping, overlaps or bibliography. All printed values and protected source identities passed automated verification before renewing the current visual receipt.

MBP delivery,3November: clean main checkout fast-forwarded from800d09f to9854647 via GitHub. Report2 PDF SHA-256 e54eadb907133037e967895a0787efb0659d25ac72d9044363b10a613d36d9cc matches the reviewed Mac-mini copy. User requires future completed report revisions to be pulled onto MBP as part of delivery. No remote model/environment changes.

Balanced layout: first two builds were4pages because the local-dynamics page overflowed. Tightened repeated setup/conclusions while keeping11pt body; third build is3pages. Final small wording edit defines D explicitly and defines off-mode distance, followed by recompilation. The first verification expected literal PD(h) from the earlier heading; the definition now appears directly in the neutral mathematical setup. All new and restored numeric checks pass. No experiment or source values changed.

Final balanced-report review passed: all three current pages inspected, neutral headings verified, substantive readout/frame/cubic/diagnostic coverage restored, Muon and diagonal evidence retained naturally. All main numbers bind to primary tables; Report1, docs/studies and image trainer paths remain identical to23e2469. No new model execution. Publish and sync MBP as required.

4 November revision: previous PDF was complete but predated cycle14. User explicitly requests direct factual prose without repeated defensive caveats. Incorporate the completed Ring8 fixed-checkpoint/bank audit, keep broad clock/P/order/readout/frame/cubic/diagonal exploration, retain actual ordinary controls and negative outcomes as findings. Product16 enhanced sampling remains pending. Report-only change; no experiment launch.

4 November reporting checks: initial layout overflowed to page4; shortening prose restored three pages without changing11pt body. A long Product16 model label caused an overfull line; rewriting the sentence fixed it. New Ring8 figure/table bind directly to native CSVs and cost receipt. Repeated generic disclaimers and the old5% continuation argument are removed from report prose; concrete outcomes and settings remain.

Final4November PDF inspected on all three rendered pages:11pt body, three readable vector figures, complete tables/equations, next-two-week hypotheses on page3. Latest Ring8 bank/pool numbers, all historical printed cases and matrix counts match source artifacts. Report1 and primary docs/studies remain byte-identical to3285bdc. Final publication and MBP delivery follow.
