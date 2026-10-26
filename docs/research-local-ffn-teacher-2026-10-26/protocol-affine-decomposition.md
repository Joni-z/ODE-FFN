# Exploratory local error decomposition

Written after the eighteen aggregate risks and independent audit were viewed, but before the decomposition was computed. This is a post-result localization analysis, not a preregistered confirmatory endpoint and not a new training intervention.

At each position, use all 20000 fixed common validation inputs Z. Form A = [1, Z] in float64 and its thin SVD. Retain singular directions with singular value greater than 1e-12 times the largest. Let Q be the resulting orthonormal basis and P_A = QQ^T.

For every one of the eighteen student-minus-teacher errors E, the empirical orthogonal decomposition is

```text
E = P_A E + (I - P_A) E
mean(E^2) = mean((P_A E)^2) + mean(((I-P_A) E)^2)
```

Compute the same decomposition of all six teacher outputs. Report actual ranks, singular values, conditioning, orthogonality errors, component risks, and both ODE-versus-generic contrasts at each position and teacher. For positive total ODE excess, show the fraction accounted for by each component without clipping fractions to [0,1]. Negative component differences can offset positive ones.

The affine component describes a component of the observed error in these input coordinates. The remaining component is orthogonal to affine functions on this empirical bank. Neither is an architecture approximation bound. Projection is fitted on the evaluated bank for this algebraic description; it is not an out-of-sample predictor, a learned repair, a confidence interval, or evidence that a new module works. All six local functions and all three students remain visible. No intervention or architecture is selected from this calculation alone.
