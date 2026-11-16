# Scope of the descriptive statistics

The comparison is a deterministic intervention at fixed trained weights,
not an independent-run sampling experiment. Each model has six local layers
observed at five physical clocks 0.1/0.3/0.5/0.7/0.9, using 16 common interpolation
pairs. These 30 records are dependent views of one function. Report medians,
IQR and all raw per-layer/clock values; do not treat them as 30 independent
replicates for a p-value or confidence interval.

The internal horizons 1,1/2,1/4,1/8,1/16are fixed before the trained probes.
Relative prediction error is RMS(actual mean-map difference minus predicted
bracket term)/RMS(actual difference), with a 1e-14 numerical floor. Cosines
are absent for exact nulls. Local changes are normalized by the original
local feature increment. Small-horizon orders use the 1/8 and 1/16 remainders;
values below 1e-13 are not used to estimate an order. No horizon is selected
for its outcome. Full-precision medians, IQR, maxima and all180records are
in summary.json and independent.json.

The full-field bank has 4,096 selection examples per task. Native float32
fields and oracle targets are retained; MSE and prediction shifts are
reconstructed in NumPy float64. Local inputs/operators are converted to
float64 only after native local-formula equality is checked. The inference
engine, training weights and future density inputs remain unchanged.

The two new-coordinate models and original anchored models share capacity
within each task. Differences between their learned computations follow
changes in coordinates under the native optimizer, including any consequent
clipping differences. The frozen controls do not replay historical AdamW
momenta and do not attribute a percentage of the sampling gap to this effect.
