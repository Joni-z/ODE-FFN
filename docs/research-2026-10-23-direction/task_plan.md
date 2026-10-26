# ODELayer mathematical and literature reassessment

## Goal
Use existing evidence, verified primary literature, and explicit mathematical arguments to select a small number of defensible next design and experimental directions. Do not launch experiments or alter the submitted report.

## Phases
- [x] Establish scope and persistent notes.
- [x] Audit existing results and implementation assumptions.
- [x] Review relevant literature and its limits.
- [x] Derive and check proposed mechanisms and counterexamples.
- [x] Produce a ranked research memo with minimal discriminating experiments and stopping criteria.

## Questions
1. What advantage belongs to input gating, polynomial parameterization, or continuous-flow structure?
2. Which constraints on P are justified by the actual target and measured failures?
3. What evidence would distinguish a useful mechanism from a decorative ODE interpretation?

## Decisions
- Keep seed 0 and existing baselines; no broad new sweep.
- Separate hidden feature dynamics, network depth, and external generation time.
- Use literature as evidence with checked scope, not as an authority for untested designs.

## Errors encountered
Two guessed arXiv version URLs returned 404; actual records were checked. The large Flow Matching Guide PDF exceeded web extraction size; no full-text review is claimed. A proceedings PDF section fetch timed out; accessible arXiv sections were used with version caveats. An early shell glob matched no model files; repeated with rg --files and explicit paths. Exact TiedFlow naming remains unresolved, so no claims depend on it.

## Status
Complete: decision memo, evidence audit, two literature notes, mathematical audit, objective bridge and numerical verification. Mathematical and evidence cross-reviews passed after scope clarifications. No new training launched.
