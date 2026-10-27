# ODE autoresearch notes

## Evidence used at setup
- Canonical starting commit: `2d2259a8e34fb15d48e57f9492f18f4d026e57c8`.
- Official autoresearch: https://github.com/karpathy/autoresearch, local `/Users/mr.z/autoresearch`, revision `228791fb499afffb54b46200aca536f79142f117`.
- `docs/research-joint-dependence-2026-10-26/next-question.md`: the ISRU advantage did not transfer; no new architecture was selected.
- `docs/research-architecture-search/coupled-generator-prior-review.md`: frozen-gate feedback alone was rejected without independent task evidence; a complete nonlinear derivative differs from its frozen-generator approximation.
- The native synthetic backbone uses pre-normalization, time FiLM and residual scaling. The complete hidden field includes their input derivatives. A derivative of the bare FFN is insufficient to describe a higher-order solver for the actual block.

## Interpretation boundary
The next diagnostic tests whether changing the integrator of an already fitted Euler field helps. Its negative outcome would reject a post-training upgrade, not all possible higher-order architectures. Its positive outcome would justify a separately registered training comparison, not an ODE-specific advantage on its own.

## Cycle01 independently replayed outcome
- Registration commit: e0ad302, created before preflight and new outcomes.
- Runtime: `/Users/mr.z/ode-workspace-prep/autoresearch/oct27-cycle01`.
- Preflight passed12 native checkpoint replays and36 float64 complete-field tangent checks.
- Formal process PID32883 exited0 after23.113seconds;204 field cases,100 time panels,20 common-source generation checks.
- Final all-block complete-Taylor MSE / unchanged Euler MSE: Gaussian47.54, Ring15.68, Product8.32, dependent-sign33.81. The registered premise criterion fails.
- Actual complete-Taylor and midpoint dense inference MACs match:130752 per16D example, versus81792 Euler; training MACs389120 versus242240. This is a diagnostic of frozen models, not a matched-training-cost result.
- The common extra-compute controls also worsen all four endpoint field errors. This suggests that treating a fitted Euler residual update as a solver-independent field is unsupported here; independent verification and precise negative-claim scope still matter.
- Audit-v6 completed204 cases/34,680,000 prediction coordinates,100 panels and20 samplings, preserving91 inputs and333 formal files. All failed audit sources/logs remain recorded under the private runtime; deviations.md explains each numerical/assumption correction.
- Exact output-error decomposition: Product cross term-0.0086382, displacement energy0.6848118; sign-code cross term0.0250104, displacement energy1.3681410. The fixed correction's output displacement dominates the small alignment benefit on Product.
- Do not turn this failure into a sign/step-size sweep or start Taylor training without new evidence. The full-support mixture control proposed in the cycle analysis addresses an unresolved interpretation of prior data dependence, not a new architecture claim.
