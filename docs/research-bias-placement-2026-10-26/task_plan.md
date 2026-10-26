# Task plan: Matched offset placement in the free SiLU core

## Goal
Test whether moving nonlinear thresholds improves the remaining local fitting gap more than an equally sized output offset, before proposing another change to the original ODE layer.

## Progress
- [x] Complete and seal the P-versus-W study; its known-representer result rejects the declared large P-learning-difficulty pattern.
- [x] Register two matched added-vector arms, unchanged targets/budget and twelve historical controls.
- [x] Verify initial-function pairing, exact representers, placement gradients, reuse and resume.
- [x] Freeze source/input bindings and launch twelve new monitored fits once.
- [x] Audit all new/reused outputs, streams and derivatives, plot every outcome and update the design direction.

## Constraints
Generic diagnostic only; no original ODE, width, activation, optimizer, target strength or budget rescue. Seed 0. Preserve every prior source/run/report, including submitted Report1 and the dirty worktree. No commit/push, subagents or GPU allocation.

## Starting evidence
The immediately preceding P/W seal is 724291d6c0a28250d623e3e9f33e4617fc5ffe63e692ecde8da838ea36967e1c. P fits its known-representer targets better than direct W; both retain a large SwiGLU-target deficit. The new question changes one vector's placement with matched initial functions and parameter counts. Lenovo SSH already timed out this turn; local two-worker CPU execution is feasible.

## Status
Complete. Twelve new fits finished in 30.2 seconds; the independent audit passed in 10.6 seconds. The substantial hidden-specific criterion fails at every position. All 24 outcomes and all saved vectors/spectra are exported, and three figures were visually inspected. See analysis-report.md and artifact-verification.json. This closes the generic placement question without claiming an ODE architecture gain.
