# Figure interpretation and caption requirements

## field-changes.png / .pdf
Purpose: test an immediate outer-transform repair over the entire cohort.
Data: paired-changes.csv, all 24native/opposite-transform checkpoint pairs.
Axes: four trained controller/output corners by three optimizer settings, separate
panels for Ring8/Product16. Colors are log10(swapped/native validation MSE),
with the same symmetric scale across tasks; labels are ratios. No smoothing or
error bars. Values below1 improve the fixed checkpoint.
Observation: all 12Product16pairs worsen; only two Ring8 pairs improve slightly.
Implication: reject a general immediate-output repair. Do not infer the cause of
training or universal superiority of one transfer function. Exact values are in CSV.

## product-cases.png / .pdf
Purpose: examine the four predeclared Product16/Adam.0003 interaction cases.
Data: hash-bound native/swapped arrays, all4096 samples; axes 0, 7, 15 after the
recorded rotation. Population curves use the unchanged known mixture parameters.
Caption must identify native/swap/population, fixed bin/frame, sample count and
per-axis fractions outside the plot. No finite subset is substituted for a failed
rollout. No error bars or smoothing beyond the displayed analytic density.
Observation: swaps alter separation, mode locations and tails; native models
also retain defects. Implication: scalar distance comparisons need their scope
and examples, not visual selection or a new metric-driven winner.

## ring-cases.png / .pdf
Purpose: inspect every corner at the predeclared Ring8/Muon.02 setting.
Data: same indices0,4,...,4092 from each 4096-sample array; target overlay and fixed
[-4,4]^2frame, with outside fractions. Caption must state that scores use all4096.
Observation: swaps can displace or broaden clusters despite unchanged P.
Implication: this visualizes the measured fixed-function sensitivity; it cannot
certify mode weights or establish a trained-method advantage.

All three figures were viewed after generation. Scientific interpretation remains
bounded by the fixed cohort and the unchanged native evaluator.

In every case figure, the row name describes the originally trained corner;
"swap" always means the opposite outer transfer, not a different controller.
