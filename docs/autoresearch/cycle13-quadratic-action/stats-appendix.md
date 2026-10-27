# Cycle13 units and numerical analysis

## Cohort and comparison unit

There are 24 existing checkpoints: original/input-frame/global-frame K2, thin/thick laws, updates0/1000/2500/5000. All inherited model seeds are0; there are no new training trajectories. Each checkpoint has20,000native field-bank inputs and six solve nodes. The2,880,000layer-source rows are repeated computations on the same sources and fitted models, not independent model observations. Four time bins plus a pooled row give720local rows. The32-row clock probes replay old diagnostics; they are not the field bank.

The full cohort was fixed before outcomes, including all checkpoints, layers and bins. Source/checkpoint/bank hashes are checked against the sealed lineage. Native inference chunks are1,024; the old512 training minibatch is a different setting. Comparisons at a layer use the exact native captured input; local algebra promotes those inputs and frozen parameters to float64, recomputing live controls. Float64 local output is compared to native output with the registered tolerance, not claimed bitwise identical to float32 inference.

## Estimands and normalization

Local relative RMS is `sqrt(sum(residual^2)/sum(K2_increment^2))` pooled over source rows and64coordinates. It is not the average of per-source ratios. Raw quadratic magnitude has the same denominator. The second-action orthogonal fraction divides orthogonal P²z energy by total P²z energy, not by total increment energy. Best-clock projection is source-wise least squares, with an unconstrained coefficient or its closed-box clipping. Both coefficient arrays, extremes and clipped fractions are retained. Closed-box values can be unattainable finite-sigmoid endpoints; the optimum is an infimum for the open native interval.

Per-layer pooled summaries and four external-time bins use the same fixed rows. The maximum local residual in the comparison table is the maximum over all six pooled solves, a descriptive worst-layer summary, not an outcome-selected layer or a registered acceptance metric. “Below1%” is the registered descriptive flag, not an architecture acceptance or significance criterion. Only8of36final layer/checkpoint pairs satisfy it;77of144pooled pairs do over the full checkpoint history.

Complete displacement RMS divides squared prediction displacement by original prediction energy. Complete risk change divides the MSE difference by original oracle MSE. Full/tangent/normal MSEs are normalized by128/2/126 respectively. Cold clean-head fields can be nonzero despite a zero raw output head; null relative metrics are used only if the actual denominator is zero.

The original full-field frozen intervention and risk polarization were registered. The tangent/normal split in intervention-parts.csv is a postregistration reporting addition to those same saved predictions. No new intervention, hyperparameter, threshold, training decision or sampler was introduced. It prevents the known shared normal field from obscuring tangent effects. All72rows are retained. Full values reproduce the audited formal summary; every projected row satisfies the same signed risk identity.

## Descriptive effects

The main report gives every final family/law row. All non-final cases are in complete.csv and local.csv, without selecting a best checkpoint. Relative tangent risk changes at final endpoints are thin original+1.35385%, input frame−0.62456%, global frame−0.13663%; thick original+25.16620%, input frame+3.97620%, global frame+36.69074%. Negative changes are finite-bank observations, not model superiority claims. The intervention has no newly measured generation or SW2 outcome.

These are deterministic finite-bank descriptions and verified algebraic identities. No p-value, significance test, training-variation interval or population model-ranking claim is made. A source-wise error/clock distribution is available privately, but the study does not need inferential tests to verify exact projection or risk accounting. Reused banks and dependent layer/checkpoint measurements must not be treated as confirmatory independent replication. No multiplicity correction is applied because no family of significance claims is tested. The1%flags are not corrected tests or confidence bounds.

## Numerical evidence

- Pre-registration fixtures: exact least-squares optimum versus a scalar grid, zero directions, clipped boundaries, flat-singular counterexample and live-clock/live-frame value/gradient identity. Maximum AD discrepancy4.44e-15.
- Preflight: all24native20,000-row field predictions replay bitwise,7.71seconds.
- Formal run:24checkpoints/144layer pairs,78.04seconds,0.768GiBsampledRSS, exit0.
- Independent audit: all native inputs/fields and changed fields replay bitwise; explicit float64 controls and dense conditional frames reproduce every source envelope. Maximum sample discrepancy1.14e-12. All bins/scalars pass rtol1e-8/atol1e-10;56formal files unchanged.58.67seconds,0.848GiBsampledRSS, exit0.
- The native rotation primitive is shared for bitwise whole-model intervention replay. Its local float64 action is separately checked with explicit dense matrices over every frame source. Checkpoint/oracle semantics are immutable shared ground truth, not independently relearned models or a second mixture-oracle derivation.

## What the study does not establish

No retrained-K1 lower bound, finite-controller approximation guarantee, new model winner, sampler quality, global stability, image-generation transfer or beneficial spectral constraint follows. Static reparameterization and a relaxed duration are diagnostic freedoms. The data show where they do and do not preserve the current finite function. Earlier failed-screen decisions remain unchanged.
