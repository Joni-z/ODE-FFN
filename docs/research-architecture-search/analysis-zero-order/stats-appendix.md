# Analysis scope

The registered eight-arm screen covers three populations and one fixed
training seed0. Endpoints, ratios and trajectories are descriptive. Times,
layers, evaluation anchors and repeated checkpoints are not independent
training replications; no p-values or across-run intervals are constructed.

The legacy identity factor preserves initial raw tensors but changes hidden
functions. The paired vector control starts with constant durations and an
explicit P translation. Its initial hidden functions agree in exact arithmetic;
actual float32 differences are recorded, not treated as bit equality. All
formal weights and training remain float32. Double checks validate identities,
not a double-precision training experiment.

Population-oracle validation and the finite-bank pair objective are distinct
and unchanged across these arms. Uniform-time oracle MSE, common fixed-time
Jacobian error and Heun32 validation SW2 use their recorded different laws.
The failed screen is not rescued by a test-set readout or a single SW2 value.

P norms are representation dependent: at fixed c the same complete map is
A=I+cP=cQ. Layer norms/averages and internal RMS ratios are diagnostics, not
complete-network stability certificates. Wall times come from the same
two-worker one-thread queue and are not isolated hardware benchmarks.

All initial/intermediate/final checkpoints remain on Lenovo. The selected
local replay archive includes final checkpoints and complete diagnostic
metadata; its exact scope/digests are recorded in collection.json. Native
full-checkpoint audits and Mac endpoint replay have different scopes.
