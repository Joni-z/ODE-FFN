# Statistical scope and verification

The unit is a common evaluation bank conditional on a fixed trained checkpoint.
There are eight banks for each model/checkpoint/solver combination. Means and
sample SD use all eight; paired deltas use the same source/target/direction
arrays. Positive deltas mean ISRU-D is worse. All outcomes are retained, with
no nonfinite rows. The 20k / Heun 128 primary endpoint predates these outcomes.
The10k / 15k and Heun 32 cases are sensitivity checks, not checkpoint selection.

No p-value or training-level confidence interval is computed. The design has
no independent training replication; banks, checkpoints and solver resolutions
cannot supply it. No multiplicity-adjusted significance claim is made. Effect
sizes are raw paired differences and ratios of corresponding model means. The
old 5% continuation rule was resource triage, not statistical significance, and
is not used to dismiss this sampling signal.

The 32,768-sample pool is an explicitly post-outcome precision diagnostic, using
only the primary saved arrays. Leave-one-bank-out pools overlap, so their range
and signs are descriptive sensitivity checks. The pooled result does not
replace the registered mean-of-eight 4,096-sample estimator. Neither target-
target floor is subtracted from a model score.

All240 generated arrays have recorded SHA256 identities. The field readout uses 15 checkpoint field arrays and five final native anchors. All five anchors are bitwise exact;
independent field mean-square errors and double-precision sorted projections
are recomputed from saved arrays. See replay.json, native_anchors.csv,
field_trajectories.csv, paired_differences.csv and precision/receipt.json.

145 Product16 is missing, not a failed transfer or a zero-valued result.
The failed CPU readiness output is not included as a scientific comparison.
The original submitted reports and 150-row training ledger remain unchanged.
