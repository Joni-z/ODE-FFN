# Complete-field diagnostic: numerical registration

Implements rk4-followup-protocol.md before complete-field outcomes are observed.
Keep all four 20k checkpoints and all 4,096 original selection-field inputs/task.
A witness uses the first 8 inputs, no checkpoint/source selection. Chunk128 is a
memory layout only; all formal rows remain. No parameters, buffers, input/time,
training, objective, sampler, ordinary control or finalized report is changed.

## Forward modes and capture

Reproduce original FP32 predictions at their native 1024 batch layout, then convert
to FP64 and use 128-input chunks. Original and identity-solver-patch outputs must
match bitwise at that same layout. Midpoint4 and RK4 two-step patches call their
existing class methods with fixed counts, preserving eight P applications each.
The method matching the trained model must reproduce original outputs exactly.

Two complete continuous forwards use the unchanged ISRU/P vector field on [0,1]
and original energy-work DOP853 reference, tolerance pairs (1e-10,1e-11) and
(1e-12,1e-13). Each pass computes every layer from its own actual upstream output.
Save local entrances, matrices, endpoints/work and complete predictions for all
six layers/modes; do not substitute references at original native entrances.

Complete predictions and corresponding layer entrances/endpoints from the two
continuous passes must agree per component within 1e-9+1e-9*abs(tight value).
Each reference's energy-work closure must satisfy the existing 1e-8 scaled gate.
An independent NumPy verifier reconstructs every discrete local solve, and
state-only RK45 integrates every saved continuous local solve at its own entrance
(rtol 1e-11, atol 1e-13), using 64-point chunks instead of 128. Independent endpoints
must agree within 2e-8+2e-8*abs(reference). All failures stay in the output and
prevent a qualified full-case claim. Numerical execution success and qualification
are separate fields. Do not reduce the cohort or choose a passing layer.

## Risk units and qualification

Risks use mean squared vector norms; divide by physical dimension only when
comparing to the historical per-coordinate field MSE. Reconstruct R,S,C and
R_cont-R_num=2C+S independently from retained arrays. Risk changes have two-
tolerance numerical floors reported from the full prediction difference; use
triangle bounds from actual field discrepancy instead of treating tolerance
parameters as output error bounds. State these as resolution diagnostics, not
rigorous error certificates for the unknown exact continuous solution.

For compensation, retain every paired loss difference L_MM-L_MR and L_RR-L_RM
on both tasks. Report point means, per-state spread and conditional bootstrap
intervals over original IID field inputs, keeping matched predictions together.
Use evaluation RNG 0 and 10,000 resamples, 95% and four-comparison-adjusted percentile
intervals. Banks are not fabricated for these field probes. The strong prediction
requires both diagonal advantages on both tasks, numerically resolved. A failed
sign rejects that strong version; unresolved precision is reported separately.
Exposure/alignment use the exact accounting and Cauchy-Schwarz bound already
registered. No arbitrary small-effect percentage is used to label equivalence.

Run on two one-thread Mac CPU workers in the unchanged 04f11f75 environment.145
physical 0-3 remain prohibited;4-7only idle and unused for this CPU diagnostic.
Retain actual process exits, source/asset/checkpoint hashes and monitor progress.
