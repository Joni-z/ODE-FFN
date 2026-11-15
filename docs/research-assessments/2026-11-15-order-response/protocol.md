# K1/K8 by response: registered comparison

## Question
Does the ISRU response gain an advantage specifically when composed with the
higher-order frozen-generator Taylor map? Existing K1 results are reused.

For each layer, calculate D(h) once at its entrance and retain it throughout
the series: term0=h; term_j=P[D(h)term_(j-1)]/j; output=sum(term0,...,termK).
ISRU D_i=(1+1.2h_i^2)^(-1/2). Tanh D_i=tanh(sqrt(1.2)h_i)/(sqrt(1.2)h_i),
with its continuous limit at zero. Two solves with middle SiLU are unchanged.
Only K changes from1 to8. This remains a frozen linear generator conditioned
on h; it is not a numerical solve of a continuously refreshed nonlinear field.

## Four new runs
| GPU | Dataset | Response | Order |
|---|---|---|---:|
|4|Ring8|ISRU|8|
|5|Ring8|scaled tanh|8|
|6|Product16|ISRU|8|
|7|Product16|scaled tanh|8|

Seed0,20k updates,batch512,AdamW3e-4,weight decay0,gradient clipping1.
Reuse native paired initialization, data streams, width64/depth3 and trainer.
No optimizer changes or new scaling/normalization/augmentation. Parameter
counts are identical within each dataset (56706/58512). K8 uses8x the FFN
P-matrix multiplications, so equal-update performance is not an equal-compute
claim. Record actual training and evaluator times; compare ordinary baselines.

## Fixed evaluations and diagnostics
At15k/20k, reuse eight4096sample IID banks,256directions,Heun128 and pooledSW2.
Ring8 also receives the same C8 projection, mode counts and rotated trajectory
checks as the K1 study. Verify native32step checkpoint sampling exactly.

At20k, switch each trained K8 model to K=1,2,4 without changing weights;
evaluate all eight common banks, field MSE and Ring8 equivariance. These are
frozen-weight counterfactuals, not independently trained low-order models.
D is recomputed at each layer entrance after upstream changes, then frozen
within that solve, exactly as in the native implementation.

Capture all six layer entrances on the first512 fixed validation states.
Measure RMS of individual Taylor terms relative to the input, combined
higher-order increment relative to term1, cancellation, and output change.
For eight entrances per layer, compare the K8 map with torch.matrix_exp(PD)h
infloat64. Approximation accuracy and generation quality are separate outcomes.
Use unchanged K1 anchors to compare changes in complete field geometry.

## Verification and stopping
Verify identical parameters, runtime buffers and RNG draws acrossK; compare
native outputs/full Jacobians with an explicit matrix-power implementation,
including derivatives through D(h). Verify8->16 resume equals16continuous.
A separate witness runs the documented commands, including actual evaluator
restoration. Stop a failed or nonfinite arm; do not rescue it with a new trick.

## Decision
Compare K8-K1 changes separately for each response/dataset, then their
interaction. If ISRU benefits specifically atK8, inspect which terms and
field-error components account for it. If both responses behave alike, the
ISRU closed-form motivation does not explain an order-specific gain here.
If high terms are negligible or harmful and ordinary controls remain better,
prioritize generator structure over a further Taylor-order sweep. Keep the
question open if fitting and sampling disagree; report both checkpoints.
