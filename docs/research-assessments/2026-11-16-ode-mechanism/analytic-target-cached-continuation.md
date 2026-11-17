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

## Actual endpoint geometry and source contributions

Registered after the complete Heun128 diagnostic, before this additional
accounting. The earlier remainder uses the potentially distant linearized
endpoint x+u. Instead let d=x_crossed-x_native be the actual displacement
and g the gradient of the same empirical squared SW2 at x_native. With no
exact projected ties, the following is an exact finite-change identity:

    actual_deltaL = dot(g,d) + Q(d) + C_sort(d).

Here Q(d) is the mean squared projected displacement. C_sort is the loss
with newly sorted crossed projections minus the loss with the original
native rank assignment; it is nonpositive because sorting gives the optimal
one-dimensional matching. For either predicted tangent u,

    dot(g,u)-actual_deltaL = dot(g,u-d) - (Q(d)+C_sort(d)).

Compute this on all eight banks and the 32,768-source pool of each completed
primary group. Reconstruct both stored tangent derivatives independently.
Check g against reverse-mode AD of the actual sorted metric, and reject exact
ties rather than choose a derivative silently. Keep both analytic/measured
tangents and all original numerical flags. Check native replay separately;
the identity uses the actual native endpoint, with no replay approximation.

Rank sources by squared tangent residual ||u-d||^2, breaking ties by original
index. For the prescribed top1, top10 and top1% report residual-energy,
target-gradient-energy and absolute gradient-contribution shares, plus the
signed contribution. Report the full signed/absolute cancellation ratio too.
These are descriptive decompositions; a small gradient projection alone does
not establish special orthogonality or a new architecture advantage. There
are no fitted scales, selected banks/directions, new model calls or samples.
The original Heun2048 and all-source precision gates remain unchanged.
