# Working notes

## Setup

2026-10-25 (Asia/Shanghai). Worktree: /Users/mr.z/ODE-FFN-report-work. No applicable AGENTS.md was found in the root-to-docs path. The worktree already contains many modified and untracked experiment files; this analysis uses a new directory and does not modify those files.

The original layer is the professor's code quoted by the user: T_K(s(h) P) h, with T_K(q) = sum_{k=0}^K q^k/k!, scalar s(h) = 0.8 sigmoid(t_func(h)/10) + 0.2, default K=2, and no bias in P. Thus 0.2 < s(h) < 1 for finite activations. The scalar is computed once, then reused in every polynomial term. The original ODEFFN composes two such layers with SiLU between them.

## Claims requiring special care

- The conditional denoising map is not the deterministic endpoint flow map.
- The input-dependent gate contributes an outer-product term to the layer Jacobian.
- The second-order real scalar polynomial has a 1/2 minimum, but a real matrix with a complex conjugate eigenpair can have a zero second-order polynomial. A blanket 1/2 lower bound for unrestricted P would be false.
- A fixed covariance has fixed eigenvectors, but generally time-dependent eigenvalues and rates. One scalar clock multiplying one fixed generator imposes separability that must be checked.
- A symmetric/negative generator can help interpret directions but may limit the truncated polynomial's expressivity; symmetry is not automatically justified for every hidden matrix.

## Primary sources to verify

- JiT: https://arxiv.org/html/2511.13720 (prediction/loss distinction and interpolation).
- Denoising posterior/Jacobian: https://arxiv.org/html/2309.13598v2 (conditional mean and covariance identities).
- Neural ODE: https://arxiv.org/abs/1806.07366 (continuous hidden-state computation is distinct from an outer generative ODE).

## Launch decision

The derivation does not select a new architecture or predict an ODE-specific advantage over a free time-conditioned linear map. No training process has been launched for this task. Analytic identities and deterministic matrix checks are the authorized mathematical verification, not another empirical architecture sweep.

## Derived conclusions

- Gaussian conditioning: C=t^2 Sigma+(1-t)^2 I; D*=t Sigma C^-1 z; v*=[t Sigma-(1-t)I]C^-1 z. Posterior covariance and Bayes risk are derived explicitly.
- Sampling flow: Phi_(u->t)=C_t^(1/2)C_u^(-1/2), distinct from the posterior mean. A singular endpoint is reached as a limit.
- Fixed-time, direct-data-space K1: P=(1-t)B_t/s for a constant gate. K2 instead satisfies (I+sP)^2=I+2(1-t)B_t.
- Symmetric K2 has a gain floor of 1/2; unrestricted real P can cancel in even-dimensional complex-eigenvalue pairs. Exact preimages of a symmetric target require each gain below 1/2 to have even multiplicity.
- A fixed finite polynomial generator with one continuous scalar gate cannot exactly match a nonconstant family of rank-deficient linear denoisers across an open time interval. The proof uses the Jacobian at zero and the finite roots of det T_K(qP). It is not a full-network impossibility claim.
- External second-order evolution includes B'_t as well as B_t^2. An internal frozen exponential/Taylor solve has a different target.
- Input-dependent gates and hidden coordinate changes prevent transferring physical spectral claims directly to P.
- The Gaussian target is linear and analytically solvable without an internal ODE; it is a calibration problem rather than evidence of architectural superiority.

## Relation to existing work

The previous mathematical reassessment and objective bridge already contained the generic posterior/Jacobian identity, the scalar K2 floor with its real-matrix counterexample, and noncontractive Gaussian directions. They were read and cited. This note makes the full subspace-Gaussian conditional-to-flow calculation explicit and proves the direct-layer scope and temporal restrictions; it does not claim the prior results as newly discovered.

## Environment

Default python3 lacked SymPy. The existing /Users/mr.z/ode-workspace-prep/.venv-test/bin/python supplies SymPy 1.14.0 and NumPy 2.2.6. No dependency installation or environment modification was necessary.

## Verification

The final verification run exited successfully: 107 checks passed, including 48 exact symbolic checks. The current derivation.md and verify_math.py hashes match the recorded inputs in math-checks.json. Local documentation links, Python syntax, and whitespace were checked. Global representability arguments were reviewed with their explicit assumptions; numerical examples are not treated as their proof.
