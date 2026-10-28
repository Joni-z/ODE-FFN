# Frozen proof units and obligations

Input: derivation.md, source units U0–U8 (all nine sections). One shard covers
all units. Native reviewer restrictions are prompt-only: read-only, recursion
false, maximum eight turns. A coverage receipt reports execution coverage,
not semantic completeness. No result is accepted merely because a reviewer
or numerical check agrees.

| Unit | Main obligation | Dependencies |
|---|---|---|
| U0 | Finite-GMM regularity, finite moments, posterior positivity and derivative domination on interior compact sets | Fixed prior class |
| U1 | Conditional-expectation orthogonality and positive time-only loss weight | U0 |
| U2 | Heat kernel identity, Tweedie mean, covariance Jacobian and PSD | U0 |
| U3 | Score heat equation and derivative commutations imply denoiser PDE | U0,U2 |
| U4 | Correct coordinate transform and uncapped material derivative | U0,U3 |
| U5 | Gaussian non-invariance and distinct invariant endpoint map/risk | U1,U4 |
| U6 | Wrong-prior PDE consistency and strictly positive true excess risk | U1,U3 |
| U7 | Differentiated normalized posterior, centered third-moment expansion | U0 |
| U8 | Two realizable posterior laws with equal mean/covariance but different drift; bounded closure claim | U0,U7 |

The dependency graph is acyclic in this order. There are no asymptotic rate,
dimension-uniform, endpoint-limit, statistical convergence or model-quality
theorems. All differential statements are at fixed interior t,r and finite d.
All expectations are posterior means under a specified prior, except U1/U5/U6
risks, whose true evaluation law is explicitly fixed.

Symbols are typed in U0. In U4, J_m differentiates observation y; J_D
differentiates FM input z. In U7, lambda varies with y held fixed; this is a
different slice from holding z fixed while changing external t. The covariance
and third moment are posterior quantities, not hidden-layer matrices.

Initial executor counterexamples: Gaussian posterior mean is not invariant;
a wrong-prior mean satisfies the PDE; equal posterior first/second moments
do not close the precision derivative. The fresh review must independently
attack those calculations and the scope of every claimed implication.
