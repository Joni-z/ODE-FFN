# ODE autoresearch notes

## Evidence used at setup
- Canonical starting commit: `2d2259a8e34fb15d48e57f9492f18f4d026e57c8`.
- Official autoresearch: https://github.com/karpathy/autoresearch, local `/Users/mr.z/autoresearch`, revision `228791fb499afffb54b46200aca536f79142f117`.
- `docs/research-joint-dependence-2026-10-26/next-question.md`: the ISRU advantage did not transfer; no new architecture was selected.
- `docs/research-architecture-search/coupled-generator-prior-review.md`: frozen-gate feedback alone was rejected without independent task evidence; a complete nonlinear derivative differs from its frozen-generator approximation.
- The native synthetic backbone uses pre-normalization, time FiLM and residual scaling. The complete hidden field includes their input derivatives. A derivative of the bare FFN is insufficient to describe a higher-order solver for the actual block.

## Interpretation boundary
The next diagnostic tests whether changing the integrator of an already fitted Euler field helps. Its negative outcome would reject a post-training upgrade, not all possible higher-order architectures. Its positive outcome would justify a separately registered training comparison, not an ODE-specific advantage on its own.
