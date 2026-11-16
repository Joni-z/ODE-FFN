# What the completed controls establish

Latest continuation, 17 November: matched RK4 training, complete-field substitutions
and crossed sampling are now complete. The solver bias affects the output;
field-MSE rankings do not reliably predict its generative benefit. See the
[current causal comparison](cross-solver-results/analysis-report.md) and
[next missing mathematical link](target-aware-bias-transport.md). The earlier
cohort/protocol below keeps its original numbers and scope.

**The strongest remaining structural signal is a task-dependent sampling benefit
from sharing P. It is not a joint-density benefit in the completed comparison.
Prescribed exponential coefficients show no consistent sampling advantage.
State feedback helps the within-family Ring8 comparison; ordinary Product16
controls remain stronger.**

The six new 20k trajectories and all28selection/14confirmation metric rows
are complete and independently recomputed. Training remains seed0. All18
resolution cases and seven float64 symmetry cases are complete and independently
recomputed. Two additional stage-operator trajectories and their fresh common-bank
evaluations are also complete. Each cohort retains its own bound arrays.
These are width64/depth3 synthetic velocity MLPs with replaceable FFNs,
not JiT image-prediction models; no image-transfer improvement is claimed.

## Matched composition control

The two P/Bschedules now have completed20k native training and common-bank
evaluation, equal parameters/work, identical initial functions and data RNG.
Palindrome(+--+) versus alternating(+-+-) gives pooled SW2Ring8.086/.108,
Product16.072/.066. Leading cancellation holds exactly in the trained
palindrome, but the sampling direction reverses across tasks. Ring8's small
palindrome-versus-shared point lead has an unresolved paired interval.
[Complete comparison and feature dependence](schedule-order/analysis-report.md).
The full registered sweeps now preserve the task reversal: internal16
Ring8.084/.107, Product16.087/.075(palindrome/alternating). Independent
recomputation of all24native rows agrees within2.78e-17.
[Complete numerical analysis](schedule-order/numerical-results/analysis-report.md).
No image promotion follows.

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

## Does the signal survive finer internal integration?

Full32768-sample checks reuse the same checkpoint, source/target arrays and
Heun128 outer sampler. Only the hidden integration changes from4to16steps.

| Hidden model | Ring8,4steps | Ring8,16steps | Product16,4steps | Product16,16steps |
|---|---:|---:|---:|---:|
| Fixed midpoint |0.071|0.074|0.052|0.055|
| Free midpoint stages |0.071|0.071|0.052|0.053|

Ring8 retains its gain over fixed exponential K8(0.088). The fixed-midpoint
Product16 advantage shrinks from0.052versus0.056to0.055versus0.056. Thus the
Product16 result should not be described as a substantial continuous-flow
advantage. More accurate internal integration does not improve sampling at
these frozen weights. This is a resolution intervention, not retraining
at each resolution or a comparison at equal inference cost.

All14outer-resolution and four full internal-refinement cases independently
recompute within2.78e-17. [Bound verification](numerical-resolution/COMPLETE.json),
[all64scores](numerical-resolution/scores.csv). The failed float32 C8 shortcut
is preserved. All seven symmetry cases subsequently completed in float64,
with independent verification of all25cases and 78 scores. No mixed-precision
symmetry ranking is reported.

## Single-generator sharing and the optimizer control

The completed sharing control now tests a single P against a larger stage-dependent
class at equal matrix work and identical initial function. On its separate fresh
confirmation cohort, shared/released pooled SW2 is0.091/0.111 on Ring8 and
0.064/0.073 on Product16, with shared better in all eight banks per task.
Released stages use their additional matrices, and Product16 field fitting improves
while generation worsens. Sharing therefore has measured value in this comparison.
[Full sharing analysis and figures](sharing-results/analysis-report.md).

Full-bank CUDA refinement then reproduces CPUwithin3.44e-8and independently
recomputes all16pooled/128bank rows within2.78e-17. Heun128->512preserves
the gap. Internal4->16retains the pooled sharing lead: Ring8 shared/released
0.091/0.103, Product16 0.069/0.082. Ring8bank support falls to5/8with an
approximate mean-gap interval crossing zero; Product16stays8/8. Thus the
fixed-budget architectural result is stronger than a two-task claim about
the continuous hidden-flow limit.
[Complete refinement](sharing-results/full-resolution/analysis.md).

This supports autonomy/parameter reuse at this budget, not a benefit unique to ODEs
or a coupling of hidden time with physical FM time. The P+Delta parameterization
changes optimization geometry, and Free ISRU/SwiGLU still beat the ODE family on
Product16. Their CPU cost is also much lower. No image promotion follows.

The mean-preserving release is now trained and evaluated. It represents the
same four-stage function class as P+Delta while keeping the mean effective
matrix equal to P. On the same sharing banks, its pooled SW2 is 0.104 on
Ring8 and 0.072 on Product16. Shared P has 0.091/0.064. Sharing wins five of
eight Ring8 banks, with an unresolved interval, and all eight Product16 banks.
New-versus-anchored point improvements have unresolved intervals on both tasks.
[Completed coordinate control](coordinate-control/analysis-report.md).

At the identical frozen functions and effective gradients, fresh native AdamW
probes give mean-update amplification 1.636--1.648 in anchored coordinates
versus 1.000 in zero-mean coordinates. Thus the original release changed
optimization as well as stationarity. The probe establishes that effect;
it does not reconstruct the historical optimizer state or assign a causal
percentage of the final gap.

The new full-bank numerical sweep also matters: inner4->16 changes Ring8
anchored/new SW2 from 0.111/0.104 to 0.103/0.105. The small new-versus-anchored
point improvement therefore reverses. Shared remains 0.091. Product16 shared,
anchored and new change 0.064/0.073/0.072 to 0.069/0.082/0.080. Outer128->512
leaves the coarse ordering. [Independent sweep](coordinate-control/numerical-resolution/analysis.md).

The completed Ring8 C8 intervention removes the state-flow lead over ordinary
controls on its paired stratified bank: state flow0.067->0.023, Free ISRU
0.088->0.019 and SwiGLU0.094->0.022. This points to symmetry/mode allocation
as an error component to test, not an equivariance guarantee of hidden ODEs.
Released stage operators were absent from that earlier cohort. The new complete
eight-case sharing intervention includes them: Ring8 shared/stage SW2 changes
from 0.072/0.096 to 0.024/0.021 under C8 projection; Product16 changes
from 0.059/0.068 to 0.055/0.063 under selected reflections. Ring8 reverses,
Product16 retains the gap. The selected-symmetry explanation is insufficient
across tasks. [New analysis and examples](sharing-symmetry/analysis-report.md).
[Completed numerical findings](numerical-findings.md),
[registered symmetry test](sharing-symmetry-followup.md),
[next exact density-error diagnostic](post-symmetry-decision.md).

[Numerical/symmetry protocol](numerical-followup.md),
[single-generator control](operator-sharing-control.md),
[mathematical derivation](mathematical-basis.md).

## Joint density rejects one proposed explanation

The exact observer preserves native sampling and passes oracle derivatives,
complete-map determinant checks and all sixteen original/projected pilot cells.
Six qualify at512steps; ten require1024under unchanged tolerances. On the
new field banks, instantaneous density-source RMSdoes not reproduce the
known Product16sampling order. [Measured diagnostics and figures](density-transport/analysis-report.md).
The full original eight-model population comparison has completed, including
all-source refinement: all sixteen model/field cells qualify independently.
After the selected Product16 reflection projection, estimated joint KL is
0.668 for shared P, 0.464 for anchored stages, 0.289 for Free ISRU and 0.894
for SwiGLU. The paired anchored-minus-shared difference is -0.204, with
adjusted interval [-0.320,-0.092]. Ring8's paired intervals include zero.

This rejects lower joint density error as the explanation of the Product16
sharing SW2 lead. Shared components are broader than anchored ones, with
projected conditional variance ratios 1.190 versus 1.146 (target 1). Pairwise
dependence does not separate the models clearly. Width is a measured
discrepancy, not a complete causal account. The 1,024-source SW2 comparison
has a substantial finite-sample floor and does not replace the large-bank
primary score. [Full density analysis and calibration](density-transport/population-analysis.md).

## What stage variation actually computes

Frozen-weight mean collapse and order reversal show that the mean-preserving
models actively use stage-dependent directions. Collapsing to the mean
changes selection-field MSE from 0.011 to 0.186 on Ring8, and from 0.020 to
0.449 on Product16. This is a within-checkpoint computation test, not a
retrained generation comparison.

At small internal horizon tau, scheduled-minus-mean flow equals tau squared
times a sum of pairwise Lie brackets, divided by 32, plus a cubic remainder.
The coefficient, sign and cubic limit independently verify. At trained
horizon 1, relative approximation errors remain 0.408/0.441; higher orders
are substantial. Stage ordering is active, while the leading term alone is
not a sufficient generation explanation.
[Derivation and native measurements](coordinate-control/schedule-decomposition/analysis-report.md).

## Completed coordinate-control density result

All new source paths now qualify, with independent reconstruction within
3.82e-14. The combined analysis retains every source in all twenty fields;
count-weight bootstrap and fsum moments reproduce the eight registered paired
contrasts within6.11e-16. The two CPUrefinement workers finished successfully;
unused GPUqueues were cancelled without starting population children.

On Product16, mean-preserving/anchored KL is0.773/0.516 originally and
0.758/0.464 after projection. The new-minus-anchored differences0.256/0.294
have adjusted intervals[0.102,0.421]/[0.145,0.455]. The new-minus-shared
intervals include zero in both fields. All four Ring8 intervals also include
zero. Removing the mean-update amplification does not produce a resolved
shared-P joint-density advantage or improve the anchored result.

The new projected Product16components are broader: relative variance1.227,
versus anchored1.146and shared1.190. Pairwise dependence remains near the
same target-sample floor. Coordinate choice affects the learned distribution;
width alone is not a full explanation. [Completed analysis, three figures
and all paired statistics](coordinate-control/density-results/analysis-report.md).

## Next falsification test

Keep two matricesP/B, the response, optimizer coordinates, initial function
and eight matrix applications fixed. Compare stage orders(+,-,-,+)and
(+,-,+,-), with W+=P+B,W-=P-B. The first cancels the leading stage-bracket
term without removing B; the second retains it. This isolates a composition
constraint from the previous coordinate intervention. It is proposed, not
trained, and its small-time cancellation is not a performance guarantee.
[Exact hypothesis, predicted coefficient and rejection rules](schedule-order-control.md).
