# ISRU follow-up plan

## Goal
Determine whether the existing ISRU gain is a useful, adequately trained
synthetic result and identify its operative mechanism before another design.
Retain the full ODE-FFN objective; this evidence review is not goal completion.

## Current status
Evidence inventory and code identification are complete. No new experiment,
training extension, architecture, or hyperparameter search has been launched.

## Actions in priority order
- [x] Locate `free_isru` and distinguish it from the tied `cubic_split` arm.
- [x] Re-read the initial result, dependent-sign transfer and half-mixture
  control; preserve their positive and negative findings together.
- [x] Check the actual 2.5k/5k CSV records and stored clipping windows.
- [x] Retain all three original tasks and both Swi64/Swi160 budgets, plus
  both transfer laws; record the partial Ring8 gain as well as its wider-control deficit.
- [ ] P0: recover exact frozen model/config/optimizer/data identities and
  locate the reusable ISRU and ordinary-baseline checkpoints.
- [ ] P0: assess baseline training adequacy and ISRU/oracle gaps using the
  existing curves, target scale, generated distributions and oracle sampler.
  Record what the current artifacts can and cannot establish.
- [ ] P1: write one focused follow-up hypothesis and controls that separate
  the positive response/parameterization lead from initialization and budget
  effects. Justify the training budget before a new result is observed.
- [ ] P1: only after that design is concrete, implement and verify the exact
  entrypoint, then run a registered, monitored comparison on Mac mini or Lenovo.
- [ ] Update the readout and parent plan/notes, and publish verified changes.

## Constraints and decisions
- Model seed0 only; preserve historical streams separately from that seed.
- No new image architecture screen until its ordinary baseline is competent.
- No automatic continuation of the old tied cubic candidate or a parameter
  sweep. A separately specified follow-up must state exactly what it tests.
- Preserve the original protocols, stopped-run outcomes, raw states and Report1.
- Do not label ordinary ISRU matrices W1/W2 as physical ODE generators P.
- Partial effectiveness, transfer, convergence and novelty are different claims.
- Read [notes.md](notes.md) and the [current evidence](README.md) before acting.
