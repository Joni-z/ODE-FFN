# Is the potential split special after matching dimension?

Registered after the exact dimension audit and before new output interventions.
The existing result identifies a large tangential output response, but the
normal and tangent spaces have dimensions 1 and 63. This experiment tests
whether that split differs from a generic split of the same dimensions.

## Fixed cohort and frames

Keep all six shared/PAL/ALT checkpoints, all six local layers and all 80
registered inputs per model. Reuse the exact native entrances, native4 outputs,
qualified continuous references and full downstream predictions from the
completed hidden-energy/readout studies. No new training or sampling inputs.

Before observing outputs, bind one complete orthonormal frame per case/layer.
Generate Gaussian 64-by-64 matrices with NumPy evaluation RNG 0, cases in
lexicographic id order and layers in chronological order. QR factorization
with the R-diagonal sign correction fixes the frame convention. Save all 36
frames and hashes outside Git; publish the manifest. All 64 unit directions
are retained, shared across that layer's 80 inputs. Directions are numerical
controls, not independent training replicates or statistical sampling units.

For delta=h4-h_ref and each frame normal r, compare:

```text
one-dimensional correction: h4 - r(r^T delta)
63-dimensional correction:  h_ref + r(r^T delta).
```

The first removes a 1D component; the second removes its 63D complement.
The actual potential-normal/tangential outputs are the prior exact interventions.
Every call changes only one layer, sees the exact original entrance and native
output, then leaves all downstream computation native. Identity predictions,
weights, buffers and hook removal must round-trip exactly.

## Primary comparisons and predictions

Use the same reproduction fraction F as the previous one-layer study, measured
against the full continuous-reference replacement. For the null, average
residual energies over all 64 directions before dividing by the common full
response energy. Pool over every layer and input; also retain each layer,
direction and input bank. Do not select a favorable frame direction.

Primary contrasts for each of six models:

```text
Delta_N = F_potential_normal - mean_r F_random_1D
Delta_T = F_potential_tangent - mean_r F_random_63D.
```

The strong tangential-specificity hypothesis requires Delta_T>0 in every
model. Any nonpositive point difference rejects that six-case claim. Separately,
normal specificity requires Delta_N>0 in every model. Neither prediction is
substituted for the other. Positive point estimates with intervals crossing
zero remain conditional evidence rather than resolved general support.

The eight original input banks are conditional uncertainty units. Enumerate
all 8^8 ordered bank resamples via 6435 weighted multisets, jointly resampling
the common response and both actual/null residual sums. Report ordinary 95%
and nominal Bonferroni-adjusted intervals for 12 contrasts (six models by two
components). This describes input variation at fixed weights and fixed frames.
No inference over training seeds or over all possible random frames is made.

## Supporting geometry and sensitivity

Verify for every state that the complete frame's mean removed 1D feature error
is exactly ||delta||^2/64; complement energy is 63||delta||^2/64, within
1e-11 absolute/relative rounding. Independently reconstruct every projected
candidate within 1e-11 absolute/relative. Retain exact zero-feature or
zero-response denominators as unresolved, without removing them.

Also report squared downstream response per squared removed feature length
for actual normal/tangent and their frame means. This is a finite-intervention
gain, not a Jacobian or proof that nonlinear effects scale linearly.

## Execution and research decision

Two one-thread Mac CPU workers, one dataset each, warm 04f11f75 environment.
First-layer witnesses use all 80 inputs and all 64 directions in each of the
three models per task. A fresh documented pass and independent reconstruction
precede the complete all-layer run. Preserve source freezes, all previous
precision failures, final Reports1/2 and native image code.

If a generic 63D correction reproduces at least as much output change, the
large tangential F alone is not evidence of unusual ODE sensitivity. If the
potential-normal component exceeds its 1D null, it carries structured signal
despite its smaller absolute contribution. Neither result proves a generation
advantage or selects a new FFN. A larger sampling experiment must follow the
resolved mechanism, not merely repeat the best small-pilot score.
