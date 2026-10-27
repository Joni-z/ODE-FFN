# Derivatives of the conditional field, without assuming zero curl

Use the fixed Gaussian-mixture law and frozen input projection from cycle09. Write a=1-t, delta=max(a,.05), U=B X, Pi=B^T B, C=BQ and Sigma=qI+(.04-q)QQ^T. Times here are strictly less than1. With D=a²I+t²B Sigma B^T and K=t Sigma B^T D^-1, set M=Q-tKC. The posterior component mean and covariance in the two latent coordinates are m_bar and S_m. Exact Gaussian conditioning gives

```text
m(U) = E[Y|U,t] = K U + M m_bar
J_m = K B + M S_m (t C^T D^-1 B).
```

The posterior component probabilities have derivative `grad_U p_j = t*p_j*D^-1*C*(m_j-m_bar)`. Differentiating their mean yields the stated J_m, with no finite-difference approximation.

Another derivation uses the Gaussian observation U=t BY+a BZ:

```text
J_m = (t/a²) Cov(Y|U,t) Pi
Cov(Y|U,t) = Sigma-t² Sigma B^T D^-1 B Sigma + M S_m M^T.
```

This identity follows by differentiating the Gaussian likelihood inside the finite-moment posterior integral. The score is `(t/a²)B(Y-E[Y|U])`. The covariance formula follows by conditioning within each mixture component and adding the covariance of component posterior means. Both identities include the mixture responsibility derivatives.

The conditional expectation of Z is `B^T*(U-t B m(U))/a`; its unobserved orthogonal coordinates retain mean zero. Hence the optimal direct field from cycle09 has the alternative form

```text
f_U_direct = [(aI+tPi) m(U)-Pi X]/delta
J_U_direct = [(aI+tPi) J_m-Pi]/delta.

f_U_skip = [m(U)-X]/delta
J_U_skip = [J_m-I]/delta.
```

Clean and shifted heads share the conditional decoded field at fixed B, while their fitted B and raw scaling can differ. These are derivatives of the optimal **field-loss** predictor. They are not asserted to minimize a Jacobian-loss objective. The cycle09 population orthogonality of field errors does not imply orthogonality of their derivatives.

In particular, the full-input clean posterior derivative is symmetric positive semidefinite, but the conditional derivative need not be symmetric:

```text
skew(J_m) = t/(2a²) * [Cov(Y|U,t) Pi - Pi Cov(Y|U,t)].
```

A covariance and the input projector need not commute. Penalizing every skew component of a fitted conditional predictor would then penalize some of the field-optimal reference itself. Full observation Pi=I removes the commutator; this is a necessary verification control, not a new architecture suggestion.

For observed derivative errors define I_J=J_U-J_full and R_J=J_model-J_U. Always keep `||J_model-J_full||²=||I_J||²+||R_J||²+2<I_J,R_J>`. Its cross term is not discarded even in population. Also `||R_J||²=||sym(R_J)||²+||skew(R_J)||²`, because symmetric and skew matrices are Frobenius-orthogonal. Applying Q^T(.)Q before this split isolates the2D tangent-to-tangent block. These are algebraic diagnostic identities, not transport-error bounds or causal P effects.
