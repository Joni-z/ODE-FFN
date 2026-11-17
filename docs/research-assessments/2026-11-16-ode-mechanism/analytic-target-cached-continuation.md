# Cached continuation diagnosis of the analytic target forecast

The complete primary Heun128 groups have positive analytic forecast scores,
while finite endpoint tangents are inaccurate. This post-observation diagnosis
separates two sources of scalar forecast error without a new model call.
The registered Heun2048 prediction and all-source numerical gates remain
unchanged. Small-intervention success cannot replace that primary test.

The earlier target-transport cohort already saved all eight banks at
lambda = 0.125, 0.250, 0.500, 1.000 for the field

    v_lambda = v_native + lambda*(v_crossed-v_native).

For every complete Heun128 group now available, bind the earlier native,
crossed and intermediate endpoint files, targets and directions by hash.
Require all-source native/crossed endpoint replay within
1e-10+1e-9*abs(reference), exact target/direction identity, and independent
reconstruction of all original pooled loss changes. Retain all four lambda
values and all available groups; the other registered groups remain pending.
These old outcomes are known retrospective controls, not a prospective test.

Use the new analytic and simultaneous measured target derivatives La and Lm,
without fitting or rescaling. For each lambda, predict lambda*La and record
the zero-change-relative score. The exact error decomposition is

    lambda*La - actual_deltaL(lambda)
      = lambda*(La-Lm) + (lambda*Lm-actual_deltaL(lambda)).

The first term measures the analytic-field approximation's effect on this
linear functional. The second measures finite-intervention error of the
measured tangent. Neither term is a fitted explanation or a new numerical
qualification. Preserve every original finite-difference flag from both
cohorts and verify the identity from complete cached arrays.

If the analytic forecast improves at smaller lambda, this supports a local
response account and identifies finite-scale effects to inspect. If it does
not, the analytic approximation may miss target-sensitive directions even
locally. Neither outcome establishes an ordinary-FFN performance advantage.
