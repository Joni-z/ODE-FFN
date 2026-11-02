# Capacity and computation controls

For Product16, exact parameter counts and dense forward linear MACs per input:

| Family | Parameters | Dense linear MACs |
|---|---:|---:|
|MLP PD K2|83,664|106,560|
|Full-affine PD K2|83,472|106,560|
|Factorized-linear PD K2|83,664|106,560|
|Direct-input PD K2, raw or bounded|58,512|81,984|
|Ordinary SwiGLU nominal64, actual42|58,572|57,024|
|Ordinary SwiGLU nominal128, actual85|83,598|81,792|
|Ordinary SwiGLU nominal192, actual128|108,624|106,560|

Counts include the common backbone. Dense MACs sum each Linear weight's entries,
adding one extra use of each P atK2. Fourier, elementwise activation, gate products,
normalization, optimizer and diagnostics are excluded; this is not measured latency.
Use actual wall time and memory separately. Model dimension changes common counts,
so these Product16 totals must not be copied to Ring8 without recomputation.

Direct gates save about30% of parameters and23% of dense MACs against learned
PD,K2. Their proper ordinary references include both nominal64 for parameter
matching and nominal128 for dense-MAC matching. A win against a larger network
is informative, but a loss to it does not settle a smaller-parameter frontier.

The current strong ordinary nominal128/192 results remain in every comparison.
A prospective six-to-eight-cell nominal64 ordinary control should be run before
claiming a parameter-budget benefit for direct gates: retain AdamW.0003/.001 and
Muon.006/.02 under the same20k protocol if selected. This is a baseline completion,
not a new architecture trick or an outcome-driven rescue of a candidate. Native
SwiGLU source is reused with only its nominal width argument changed.
