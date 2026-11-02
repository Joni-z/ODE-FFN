# A feature metric for interpreting P

This is a derivation for a possible diagnostic, not a selected optimizer or a
new training result. Complete cycle8 before choosing the next experiment.

Historical scope: the earlier [projection-geometry analysis](../../research-architecture-search/analysis-projection-geometry/analysis-report.md)
already measured uncentered input moments and downstream responses, and did not
select P-only whitening. The present calculation specializes that question to
the PD first-term features and recent constrained-training outcomes. It is not
a discovery that anisotropic features exist, or a reason to reopen whitening.

## The object that is invariant under a diagonal rescaling

At one frozen K1 layer write q(h)=D(h)h, increment r=Pq, and
M=E[q q^T] under a specified hidden-state distribution. M is an uncentered
second moment: the mean of q matters. Then

```
E[r r^T] = P M P^T
E||r||^2 = tr(P M P^T).
```

For the earlier constant-diagonal change of coordinates q'=Cq and P'=PC^-1,
we have M'=CMC^T and exactly

```
P' M' P'^T = P M P^T.
```

This family symmetry requires an admissible transformed controller. Fixed
sigmoid bounds and the direct d(h)=h design do not generally admit an arbitrary
C within their original parameterization; the algebra is not a claim that all
current models have the same rescaling freedom.

Thus the singular values of P M^(1/2), including their nonzero multiplicities,
are invariant under this reparameterization even though those of P need not
be. They are square roots of the eigenvalues of P M P^T. No inverse is needed
to measure this quantity, even if M is singular. For a finite fixed feature
bank, use its empirical second moment and the identity remains algebraically
exact when the bank itself is transformed by C.

This is an increment-energy diagnostic in the chosen feature distribution,
not a full input-Jacobian spectrum, a global Lipschitz bound, or a measure of
prediction error. Changing the probe bank changes M. Endogenous hidden features
can also change during training. For K2, P occurs twice in a term and this one
increment identity does not characterize the complete polynomial.

## Why even isotropic hidden inputs need not produce isotropic features

For an illustrative h with independent standard-normal coordinates, q=h^2
has E[q_i^2]=3 and E[q_i q_j]=1 for i!=j. Therefore

```
M = 2I + 11^T.
```

The eigenvalue in direction1 is d+2; all d-1 orthogonal directions have value2.
At width64 their ratio is33. This anisotropy comes from the feature mean,
despite isotropic h. Replacing M by the centered covariance2I would miss it.
For q=h abs(h), M=3I instead. The actual trained hidden distribution is not
assumed Gaussian, and the absolute-diagonal training test already failed.
These examples explain what should be measured, not what should be changed.

## What a local functional step would optimize

Suppose q is held fixed and M is positive definite. For a matrix gradient G,
consider constraining the expected squared change of the increment:

```
min <G, DeltaP>F
subject to tr(DeltaP M DeltaP^T) <= epsilon^2.
```

Put B=DeltaP M^(1/2). The objective becomes <G M^(-1/2),B>F. Cauchy-Schwarz
then gives, for nonzero G,

```
DeltaP* = -epsilon G M^(-1)
          / sqrt(tr(G M^(-1) G^T)).
```

If the constraint instead uses ||DeltaP M^(1/2)||op <= eta, spectral/nuclear
duality gives one minimizer

```
DeltaP* = -eta polar(G M^(-1/2)) M^(-1/2),
minimum = -eta ||G M^(-1/2)||nuclear.
```

This specializes the earlier fixed-D [local-step argument](muon-metric.md).
It is not ordinary Muon: a feature metric enters before and after the polar
operation. A rank-deficient M creates unconstrained null directions, while
small eigenvalues can produce large parameter changes. Damping or truncation
would introduce additional choices and can break the exact coordinate identity.
The derivation also freezes q and ignores propagation through the rest of the
network. It supplies no global convergence or quality guarantee.

The practical first step, if selected after cycle8, is to measure M, P M P^T
and the existing first-term amplitudes on fixed native probes. It is premature
to implement whitening, a new loss or a new optimizer from this calculation.

[The seeded algebra witness](spectrum/feature-metric-witness.json) verifies the
second-moment/singular-value identities and both constrained-step formulas in
float64, including coordinate-transformed gradients. Its largest residual is
1.43e-14. This fixture is a numerical proof check, not a model-quality result.
