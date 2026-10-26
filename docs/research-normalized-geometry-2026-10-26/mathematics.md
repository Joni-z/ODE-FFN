# What normalization makes observable

For a pre-LayerNorm residual backbone, let u=1/sqrt(d) times the all-ones vector and C=I-u u^T. Its raw hidden states admit a uniform-shift gauge: if h is replaced by h+alpha*u, every subsequent LayerNorm sees the same state. Subsequent conditioned branches therefore agree, the difference remains a uniform shift through residual additions, and the final normalized readout agrees. Finite LayerNorm epsilon does not break this shift identity.

Radial scaling is different. For c=C h and sigma=sqrt(||c||²/d+epsilon),

```text
J_LN = diag(gamma)/sigma * (C-c c^T/(d sigma²))
J_LN u = 0
J_LN c = gamma*c*epsilon/sigma³.
```

The radial derivative is small only relative to an appropriate nonzero variance and finite epsilon. A radial branch also changes the relative size of the next angular displacement. Removing radial drift from a trained network is not a consequence of shift invariance.

For the second solve of the professor's scalar K2 layer, let z be its input after the first solve and SiLU, and a=a(z) its once-computed duration:

```text
L_P(z) = z + a Pz + a²/2 P²z.
```

The constructed choice P=lambda*u u^T gives L_P(z)=z plus a uniform shift. Thus the full normalized network is independent of lambda although P has eigenvalue lambda and largest singular value |lambda|. This is an exact identifiability counterexample within the actual family, not a claim that trained matrices have this form or that floating-point overflow is harmless.

An important boundary follows. For P'=P+u v^T:

```text
C(P'-P) = 0
C(P'²-P²) = (C P u) v^T.
```

At K1, changing the uniform output row of a second solve is invisible to the whole normalized backbone. At K2 it is generally observable: a uniform intermediate signal can return through C P u. One cannot project P to C P inside a K2 recurrence and claim the preceding full-output shift identity still applies.

In particular, taking P'=C P gives the exact second-solve identity

```text
C L_P(z) - C L_(CP)(z) = a(z)²/2 * (C P u) * (u^T P z).
```

The scalar gate a(z) is identical in these two local evaluations. The right side is a centered rank-one return path. It includes both initially centered and initially uniform inputs. Its derivatives include derivatives of the gate; detaching the gate would change the identity's Jacobian. Once a block is modified, all later states and clocks must be recomputed.

The identity applies specifically to the second solve, whose output is followed by normalized residual computation. Applying the same argument to the first solve is invalid because SiLU immediately follows it and does not discard uniform shifts. The next question can therefore isolate this K2 return path without changing the first solve, middle nonlinearity, first-order centered action, residual scale or normalization epsilon.

These identities are algebraic reasoning about the current architecture. They do not prove that a large return path is harmful, that its removal trains better, or that raw matrix spectra explain task error.
