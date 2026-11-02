# A possible next contrast: identical conditional singular values, different features

This branch was subsequently selected in cycle5 and failed all6 matched field
comparisons; see cycle5-results/findings.md. The pointwise identity below remains
valid, but the feature intervention did not establish a useful model.

Fix the same P and hidden input h. Write H=diag(h), B=diag(abs(h)), and
S=diag(sign(h)), choosing sign0=1 at zero coordinates. Then H=B S and S is
orthogonal. Hence

```
A_raw=P H=(P B)S=A_abs S,
A_raw A_raw^T=A_abs A_abs^T.
```

Every singular value is identical, including zero values. Frobenius norm,
operator norm and singular-value stable rank therefore match exactly at these
matched states. Eigenvalues and orientations need not match. This statement
does not compare the different states/weights reached by two trained models.

Nevertheless their K1 branches compute different functions:

```
raw:  F(h)=h+P(h*h),       J_F-I=2Pdiag(h),
abs:  G(h)=h+P(h*abs(h)),  J_G-I=2Pdiag(abs(h)).
```

The raw increment is even, and the abs increment is odd. Their increment-
Jacobian singular values also match at the same P,h by the same right-sign
argument. The complete Jacobians include I, so those singular values generally
do not match: orientation relative to the identity matters. For scalar p=.4,
h=-.7, both generators have singular norm.28, both increment Jacobians have
singular norm.56, but complete derivatives are.44 and1.56 respectively.

This is a useful test of the sufficiency of a singular-spectrum explanation.
A future difference could not be explained by an intrinsic per-state singular-
value penalty alone, but could reflect sign/orientation, feature parity,
regularity, optimization or resulting hidden-state distributions. The abs
feature is C1, with a second-derivative change at zero. The existing SiLU and
backbone biases mean an odd hidden feature does not force the full predictor
to be odd. Nor does target symmetry by itself establish that this arm will help.

If selected later, retain K1, no new learned parameters, exact same inherited
P/common tensors, existing optimizer/rate grid and all ordinary controls. Check
both the algebraic singular-value match and full derivatives. This is a clean
feature counterfactual, not a claimed new optimizer or FM-specific theorem.
