# Current task — ODEFFN mainline report, 2 November 2026

The user asks to organize every ODEFFN attempt and make that the center of Report2.
The previously proposed ISRU-centered continuation is not the current report narrative.

## Executable plan
- [x] Inventory all archived experimental families, controls and diagnostics, separating prior Report1 context from later work.
- [x] Read existing protocols/results; write a complete linked attempt ledger with purposes, observations, and limits.
- [x] Rebuild the three-page report around original P/clock/Taylor structure and the evidence that changed our understanding.
- [x] Verify selected values/figures, compile and inspect all PDF pages, and confirm Report1/source/runs remain untouched.
- [x] Update the evidence map and continuity notes; push verified changes to GitHub.

## Constraints
English, sole author Zhizhe Zhang, no bibliography, at most three PDF pages including selected figures. Full inventory is separate Markdown/data within Report2. Start with project/progress, finish with next-two-week hypotheses. No training or checkpoint mutation.

## Status
Complete: 127 linked evidence records, a three-page ODEFFN report, source-row/number checks, embedded-font/layout checks and visual inspection of all three pages. Report1 and primary experiment documents remain identical. Publish these verified files to canonical GitHub main and confirm its head matches the local commit. Previous draft remains in Git at 5c48a97.

---

The sections below are historical plans; current scope and progress are above.

# Report2 plan

## Goal
Produce an evidence-led Report2 covering progress after the finalized Report1,
while continuing the underlying ODE-FFN research objective.

## Scope decisions
- Report1 is final at `994e660921d2885592704f5c471bd43a12b1c6ef`.
- All subsequent research is Report2 material, not an update to Report1.
- The current ISRU follow-up and baseline-validity reassessment belong here.
- Keep old protocols, sealed raw data and submitted Report1 unchanged.

## Current state
November1,2026: the user requests a Report2 draft and current effectiveness.
The CIFAR run is live at about40.45k/50k updates. Completed5k/10k/25k FIDs
remain poor; the25k ISRU/matched/wide scores are252.96/262.93/268.74.
Draft from a fixed evidence snapshot; do not interrupt or change training.
Use the submitted Report1 layout read-only, English, Zhizhe Zhang only,
three pages maximum, no bibliography, and end with two-week hypotheses.

## Draft execution — November1
- [x] Freeze the completed25k CIFAR artifacts and validation through40k;
  check source/data bindings, feature hashes, checkpoint identities and FID.
- [x] Inspect raw/EMA fixed-start image cases and W1/W2 trajectories.
- [x] Select the post-Report1 synthetic evidence, preserving all five ISRU
  tasks and both SwiGLU budgets; distinguish field and generation metrics.
- [x] Generate reproducible figures, numeric tables and an evidence map.
- [x] Write report2.tex in the existing report format with explicit purposes,
  observations, scoped mathematics and two-week hypotheses.
- [x] Compile report2.pdf, verify at most three pages, read extracted text
  and render/inspect pages; keep Report1 byte-identical.
- [x] Update project notes/status and publish verified draft and evidence.

## Draft verification
The complete PDF is three pages with table and figures, English, sole author
Zhizhe Zhang, no bibliography. All completed FIDs were independently recomputed
from saved features; maximum difference0.0000233. Cold/25k FFN matrix summaries
were checked against actual checkpoint tensors. The final PDF has embedded
fonts, no undefined references, no overfull boxes and no placeholder markers.
Report1's original tree/PDF hashes match. The live trainer was not changed.
See [verification.json](verification.json), [EVIDENCE.md](EVIDENCE.md) and
[analysis notes](analysis/analysis-report.md). Drafting is complete; the
registered image run and final50k results are still outstanding.

## Actions
- [x] Identify the last Report1 revision and verify its current tree is identical.
- [x] Create the separate Report2 directory and document the reporting boundary.
- [x] Map the principal existing results, limitations and current ISRU work.
- [ ] Complete the ISRU evidence/training-adequacy follow-up recorded in the
  [research plan](../../docs/autoresearch/isru-followup/task_plan.md).
- [ ] Update [evidence-index.md](evidence-index.md) whenever a major result or
  interpretation changes, retaining both positive and negative observations.
- [ ] Select the main narrative from the strongest complete evidence rather
  than counting experiments or listing every attempted structure.
- [ ] Draft a separate English LaTeX report, sole author Zhizhe Zhang, with a
  maximum three-page main report and next-two-week hypotheses at the end.
- [ ] Verify every reported value/figure and the compiled page count before
  calling Report2 ready; do not modify Report1 during this process.

## Evidence boundaries
ISRU's finite-budget positive result is retained, with transfer failures and
unresolved convergence/attribution. Current small-model image FIDs do not
substantiate architecture superiority or rejection. Mathematical identities,
implementation checks and performance claims must retain distinct scopes.
