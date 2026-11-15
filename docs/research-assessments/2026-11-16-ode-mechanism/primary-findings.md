# What the completed controls establish

**Updating the hidden state is the positive architectural intervention.
Prescribing exponential or midpoint coefficients has not improved the fixed-
budget sampling scores consistently. Fixed midpoint does retain its expected
numerical convergence order after training. Operator sharing remains open.**

The six new20k trajectories and all28selection/14confirmation metric rows
are complete and independently recomputed. Training remains seed0. The
pending numerical/symmetry queue is a separate fixed-weight investigation;
its partial results are not used as completed evidence here.
These are width64/depth3 synthetic velocity MLPs with replaceable FFNs,
not JiT image-prediction models; no image-transfer improvement is claimed.

## Main comparison

Confirmation pooled SW2, lower is better:

| Model | Ring8 | Product16 |
|---|---:|---:|
| Fixed exponential K8 |0.088|0.056|
| Free Taylor coefficients |0.089|0.055|
| State flow / midpoint4 |0.071|0.052|
| Free midpoint stages |0.071|0.052|
| Free ISRU |0.082|0.051|
| SwiGLU |0.091|0.034|

The two coefficient-release controls match their reference initial
functions and common gradients. They retain the same P applications and
response function. Free Taylor coefficients add42scalars; free midpoint
stages add24. State feedback retains the fixed-K8 parameter count and eight
P applications per solve, but reevaluates the nonlinearity along the path.
It does not have the fixed-K8 initial hidden function.

State flow beats fixed K8 in all eight Ring8 banks and six of eight Product16
banks. On Product16, ordinary Free ISRU still wins at the same parameter
count, and the wider native SwiGLU remains clearly better. Ring8 SwiGLU64
uses Muon0.02; Product16 SwiGLU192 and all focal variants use AdamW0.0003.
The ordinary comparisons retain these configuration labels.

[Complete field/sampling tables and figure](completed-comparison.md),
[exact paired effects](results/paired-effects.json),
[bound metric artifacts](results-completion.json).

## The released parameters are active

Taylor coefficients depart from their exponential initialization and affect
features: resetting them to1 at frozen weights raises field MSE from0.011
to0.034 on Ring8 and from0.018 to0.195 on Product16. Yet the separately
trained fixed-coefficient reference has similar sampling quality. A model
using a supplied mechanism is not proof that the mechanism is essential.

Free midpoint gamma values reach1.520 on Ring8 and1.972 on Product16.
Resetting them at frozen weights shifts predicted velocities by0.059/0.064
RMS, respectively. Product16 field MSE rises from0.019 to0.024; Ring8 field
MSE is almost unchanged despite a nonzero vector-field change. Their similar
trained sampling scores are therefore not explained by coefficients simply
remaining inert at their initial values.

[Taylor reset diagnostic](coefficient-effects.json),
[midpoint reset diagnostic](stage-effects.json). These are fixed-weight
interventions, not substitute retraining comparisons.

## A concrete mathematical benefit of the solver constraint

At learned layer entrances, compare each local solve with an adaptive
DOP853 solution of the actual autonomous field `y'=P ISRU(y)` in float64.
The adaptive reference is checked again with tighter tolerances. Over six
layers and eight states per layer, median observed convergence rates at the
32->64 refinement are:

| Model | Ring8 rate | Product16 rate |
|---|---:|---:|
| Fixed midpoint |1.992|1.992|
| Learned midpoint stages |0.973|0.971|

Rate2 means halving the internal step reduces local-map error about fourfold;
rate1 means about twofold. The prescribed midpoint stages provide a real
second-order approximation to a well-defined continuous hidden field. Free
stages relinquish that property while attaining similar scores at their
trained four-step budget. Thus approximation order and generation quality
are distinct observations. Classical solver order is not itself a new result;
the full-sample effect of this distinction is being checked.

![Trained hidden-flow convergence](figures/trained-convergence.png)

Lines show medians and shading the interquartile range over six learned
layers. The reference solves the same local field, not the physical data
velocity. [Exact layer results](trained-convergence.json).

## What is unresolved

The current positive comparison identifies state feedback/repeated nonlinear
computation. It does not isolate the benefit of using one shared P, establish
an objective-specific relationship between hidden time and physical FM time,
or demonstrate a general replacement for ordinary FFNs. The additional
fixed-weight checks ask whether the observed gain survives internal/outer
resolution and whether Ring8 mode allocation explains it. The next conditional
training control releases stage matrices at the same matrix-work budget.

[Numerical/symmetry protocol](numerical-followup.md),
[single-generator control](operator-sharing-control.md),
[mathematical derivation](mathematical-basis.md).
