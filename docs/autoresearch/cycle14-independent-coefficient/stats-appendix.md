# Cycle14 statistical units and verification

## Comparisons and units

Ten new trajectories cover five registered arms on two fixed laws. Original K2 and Swi192 contribute four reused trajectories, explicitly identified in comparison.csv. Each uses model seed 0, 5,000 updates and the same historical stream identifier 100000. These are architecture/task comparisons, not fourteen independent replicates of one treatment. The full comparison cohort and thresholds were fixed before outcomes at 19af9a8.

Field MSEs average all 20,000 states per law. The tangent projection has two coordinates; normal energy is divided by 126, and full energy by 128. Native Jacobian panels have 64 anchors at each of five times. Their full squared Frobenius errors are averaged equally across the five panels. Tangent-output Jacobian error projects only the output axis onto the saved basis, retaining all 128 input directions. It is not the 2-by-2 tangent-input block. All metrics are lower-is-better and retain the original capped-field definition.

Sample SW2 uses all 2,048 saved generated/target rows and the unchanged native directions. Tangent SW2 uses the already registered 128 uniformly spaced target-plane directions. It is a projected sample metric, not FID or an exact population Wasserstein distance. Point clouds are qualitative examples of these same complete sample banks, without selecting a favorable subset.

## Fixed engineering gates

Both proposed nominees face six metric checks on each law plus two cross-law control checks. All 28 ratios, thresholds and pass/fail values are retained in decision-checks.csv. Thin full/tangent errors require at least 10% improvement against both Swi192 and fresh Swi258; thick full/tangent errors must stay within 5% of the better reference. Full/tangent-output derivative errors and full/tangent SW2 must stay within 10% on both laws. Mechanism ratios are geometric means of the two law-specific tangent-risk ratios, with a 0.95 ceiling against both affine and free-gate controls.

Quadratic C's failed ratios are thick tangent MSE 1.162713, thick tangent SW2 1.225433, geometric-mean tangent MSE 0.990919 versus affine C and 0.997564 versus free-gate C. The thresholds are 1.05, 1.10, 0.95 and 0.95. Direct B also fails. No best update, new duration, hyperparameter or tolerance is selected after observing the failures. The score used only to break ties between eligible nominees does not override these per-law gates.

Historical Swi257 is included after the fixed decision as explicitly labeled context from sealed cycle11. It was not newly trained, inserted into the registered rule or used to select a result. Its stronger thick tangent reference reinforces the negative result. Primary decisions are reproducible from the registered Swi192/Swi258 rows alone.

These are finite-bank engineering continuation decisions, not significance tests. No p-values, multiplicity-adjusted significance statements or population ranking intervals are reported. Reused banks and correlated checkpoints/layers do not supply independent replication. Four checkpoint observations are a trajectory, not four repeated training trials. The published tables retain all arms and cases; no favorable contrast is filtered into an inferred method win.

## Costs, timing and initial pairing

The protocol gives actual dense parameter/inference/training-work counts. Pointwise operations, optimizer updates and matrix storage are not included in those MAC counts. A larger nominal Swi width need not produce a better fitted model, as the two retained references demonstrate. End-to-end cell wall times include data work, checkpointing, spectra, derivative and sample diagnostics. They are not isolated inference latency or matched-kernel speed measurements.

Quadratic/affine/free C start from zero and preserve the original complete predictions and inherited gradients bitwise under a nonzero receiving head. Extra C tangents do not match affine versus quadratic; no complete equal-kernel claim is made. The free gate is initialized as an exact copied head, so its common C tangent matches the quadratic arm and its added-head gradient starts at zero.

Direct B starts at the rounded native P@P. Preflight float32 active complete-output differences are 2.38e-7 on thin probes and 4.77e-7 on thick probes. Output-head gradient discrepancies are 3.73e-8 and 4.47e-8; its inherited P gradients intentionally differ. It shares the real-arithmetic class and a float64 initial equivalence, not exact float32 hidden functions or identical training kernels. Three-update/resume tests pass for every actual trainer arm, including data RNG and nonpersistent buffers.

## P, coefficients and telemetry

The wide spectrum table contains 192 new layer/checkpoint records and all 64 singular values of P, reference B and C. These reference matrices are formed in float64 from the stored native parameter tensors. For affine/free controls, reference B=P²+C is a bookkeeping matrix: their actual effective second coefficient depends on the input clock. Native activation telemetry separately records the actual added action, clocks and r−s. These quantities are not full-field Jacobian spectra or evidence of a stability guarantee.

The shaded bands in the matrix-history figure are minima/maxima over six different solves, not confidence intervals. Training-loss/gradient/clipping rows average fixed 500-update windows; field-risk entries exist at the saved checkpoints. The 154 telemetry rows are exact source-log copies, not smoothed independent observations.

## Audit evidence and interpretation limit

The independent audit checks 40 new and 16 reference checkpoints; 50 new and 20 reference full Jacobian panels; 10 new and four reference generated batches; 10,000 paired data-stream batches; and 192 P/coefficient snapshots. Native field, Jacobian and sample replays are bitwise. All optimizer states and checkpoint step counters are checked. Every fitted coefficient checkpoint also receives dense local value/input/parameter-derivative checks on all 32 native probes, for 6,144 input-Jacobian anchors in total. It shares immutable checkpoint/oracle semantics, not a separately derived mixture oracle.

Maximum fitted local discrepancies: values 3.55e-15, input Jacobians 9.99e-16, P gradients 1.07e-14 and coefficient gradients zero in the recorded checks. Full-bank scalar replay uses the inherited rtol 1e-7/atol 1e-9; local float64 algebra uses atol 1e-10/rtol 1e-8. All 293 formal files remain unchanged. No failed formal trajectory or audit was restarted.

The coordinate-metric addendum is post-result explanatory algebra, checked on a separate small deterministic fixture. It changes no experiment. Its Euclidean gradient-flow identity does not identify the cause of Adam's observed path or justify a coordinate/optimizer rescue. This screen establishes no useful new ODE architecture, holdout success or image-generation benefit.
