# Cycle03: exact ODE representation of the gated baseline

Registered before new compatibility/training outcomes. This cycle asks whether an existing strong SwiGLU can be represented by an exactly solvable constrained ODE without changing its function, parameters, gradients or optimization trajectory. It is not an architecture-quality comparison or a claimed innovation in normalizing flows.

## Exact operation

For the input h of one existing FFN, form `a=Wg*h+bg`, `g=SiLU(a)`, and `v=V*h+bv`. Hold h and g fixed during the auxiliary solve. In lifted coordinates y=(u,v), use

```text
P(h) = [[0,diag(g(h))],[0,0]]
y' = P(h)*y,   y(0)=(0,v(h)),   duration=1.
```

P²=0, so exp(P)=I+P exactly. The readout `U*u(1)+bo` is exactly the unchanged SwiGLU. The efficient implementation evaluates this closed form using the original two dense projection calls and product. It never allocates the doubled state or a dense generator during normal training. Preserve all biases and the complete gate derivative. No learned duration or free generator parameter is added.

This conditional latent flow is invertible at fixed h, but its encoder/generator depend on h and its readout is a projection. The complete FFN is not therefore invertible, volume preserving, monotone, nonexpansive or guaranteed to match a physical FM flow. The eigenvalues of P are zero while its singular values include |g_i|; it is generally nonnormal. These distinctions are part of the registered claim.

## Fixed validation scope

1. Independent float64 literal block-matrix exponentials and automatic differentiation at small widths, including input and every parameter derivative. Check P²=0, exp(P)=I+P, conditional inverse/semigroup, exact singular values and the closed-form flow norm. A large-gate witness must demonstrate substantial norm amplification despite zero eigenvalues. Verify the original zero-input/output-bias cases.
2. Replay all nominal SwiGLU64/128/160 checkpoints at0,1000,2500,5000 on Product16, sign-code and the half-mixture, plus SwiGLU128 on Gaussian2D and Ring8:44 historical checkpoints, not44 new runs. Restore native runtime buffers. Require identical state keys and values, parameter order, full validation predictions and metrics.
3. At final endpoints compare all five time-panel predictions and full-network Jacobians and the existing Heun32 generated batch. Check a parameter-gradient probe with a deterministic nonzero readout and actual local inputs so a zero cold head cannot mask a backbone difference. Compare executed dense MACs and parameter counts with native SwiGLU.
4. On the existing half-mixture only, run one cold seed0 native SwiGLU128 trajectory and its paired shear representation for5000 updates. They receive identical batches and the historical optimizer/settings. Require bitwise equal model/optimizer/buffer states at the saved boundaries and exact equality to the historical native snapshots. Verify three-step checkpoint/resume first. This is compatibility evidence, not a new independent repetition or a quality lead.
5. On actual visited FFN inputs, record gate-derived generator singular distributions and conditional flow norm at the four checkpoints. Report the diagonal shear spectrum separately from learned free P matrices in earlier experiments. Do not impose a new norm cap or average the singular values.

Use the existing data, metrics, model seed0, one one-thread Mac CPU process, outside-Git exclusive outputs and sampled memory/CPU monitoring every5seconds. The finite wall ceiling is600seconds for the compatibility run. Never overwrite old runs or restart on an observation timeout. Hash frozen inputs, sources and earlier sealed studies before and after execution.

## Decision

Keep the implementation as an exact constrained-flow baseline only if all value/gradient/optimizer/cost/invariance checks pass. Any mismatch is an implementation problem to resolve with retained failed evidence, not a scientific performance difference. Do not label copied or replayed SwiGLU metrics as a new ODE win. No feedback extension, extra polynomial stage or new training family is selected by a pass. A future changed generator must beat the mathematically identical ordinary baseline under its own registered controls.
