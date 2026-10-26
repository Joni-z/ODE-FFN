# Read-only projection-geometry diagnostic

Registered before running the archived-checkpoint analysis. This is a diagnostic,
not a new trained architecture or a rescue of the stopped clock-coordinate arm.
The full useful-architecture goal and its baseline/holdout requirements stand.

## Question and exact identities

For each ordinary linear projection z=Mh in the complete network, define
W_i=dv_i/dz_i. For P in y=h+D(h)Ph, use W_i=(dv_i/dy_i)D_i.
Then the mean-square-loss weight gradient is exactly

```text
grad_M = (2/(N*d)) sum_i (W_i^T (v_i-target_i)) h_i^T.
```

All response probes test this identity against full-network autograd. Independent
finite-difference tests perturb each projection through the whole backbone.
For P the controller is independent of that layer's P, and K1 is required.
These identities do not assume a frozen controller during actual training.

Measure the uncentered input moment G=E[hh^T], the response moment
B=E[W^TW], response-weighted input moments, and the fraction of coherent
and individual gradient energy in the lowest positive quarter of G's
eigenspace. Input moment and response factors alone are not a full
Gauss--Newton factorization: their correlations matter. A large condition
number is not automatically an optimization diagnosis.

## Frozen comparison and inputs

Use the Stage11 frozen source, configurations, actual runtime buffers and
checkpoints at steps0,1000,5000 for four arms: original vectorK1, paired clock
coordinates, small SwiGLU and full SwiGLU. Analyze Ring8 and product16.
The unpaired temperature arm is outside this focused paired comparison.

All arms share4096 bank-derived Gaussian-source/time-uniform calibration
inputs generated with diagnostic stream seed250000. This stream changes no
training seed. Reuse all512 inputs from each existing fixed-time common panel
at t=.1,.3,.5,.7,.9. Compute input moments on these panels and responses on
the first128 uniform-time inputs. Record source, configuration, manifest,
training-bank, checkpoint and output-array hashes. Analysis is float64 on
archived float32 weights and recorded buffers.

Use both the actual sampled pair labels and the analytic population velocity
on those same128 inputs. The latter is a counterfactual population-fitting
diagnostic, **not** the conditional mean of empirical-bank pair targets.
Gradient norms/fractions depend on parameter coordinates; no cross-arm norm
comparison is an invariant measure of model quality. These probes are not
training-risk estimates or repeated experimental runs.

## Structural rank and decision rules

At a fixed time the first FFN input has moment rank at most input_dim+2,
for both ODE and SwiGLU. The analysis tests that bound. Structural zeros are
excluded from positive-spectrum conditioning and must not be called a
training failure. Uniform-time pooling can exceed that rank; deeper nonlinear
features can have full linear span. The exactly zero initial output head
makes all backbone output responses zero at step0; those zeros carry no
evidence of a failed learned P.

Only consider feature-coordinate preconditioning if ill-conditioned positive
directions materially overlap the loss-relevant responses, and identify
whether this concerns P, controllers, or the generic backbone. A generic
backbone effect requires a SwiGLU control. Discard a P-only interpretation
if its predicted gradient signature is absent. No whitening, compression,
temperature sweep or combined trick is authorized by this diagnostic alone;
register any resulting single-axis training test before seeing its outcomes.

The analysis is descriptive. No p-values or across-seed intervals are
constructed from layers, anchors or eigendirections. Neither a local gradient
signature nor a condition number proves convergence, global expressivity or
a useful final architecture.
