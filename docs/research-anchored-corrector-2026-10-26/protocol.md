# Input-anchored correction with the original ODEFFN

Registered before any new formal outcomes, 2026-10-26. This is one update-rule change with matched generic controls, not a full IIET or LiFT reproduction.

## Evidence and precise question

The audited recurrent-depth screen has no nominees. Its best original ODE endpoints have field MSE .029726 on Ring8 and .325527 on Product16, versus the generic frontier .014535 and .040903. Shared ODE with intermediate supervision improves Ring8 from .069894 to .033807, but Product16 does not improve (.397258 to .401976). More iterations degrade all shared ODE endpoints at 6 and 12 updates. In the final Product16 training window clipping fractions are .004 for both shared ODE losses; the gap does not coincide with pervasive gradient clipping. This does not establish the cause of the gap.

One distinct question is whether the repeated update should accumulate a trajectory or correct an estimate around a fixed input. [IIET, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.453/) supplies a numerical-method precedent; its historical-state components and distillation are not copied. The current experiment isolates input anchoring. No claim is made that anchoring is new, sufficient for good generation, or specific to the professor's P.

## One-factor intervention

Let h0=Embed(x_t), a=1/sqrt(3), and F(h;t) be the exact current conditioned, normalized FFN branch. Three calls use shared weights and final-only direct-velocity regression:

```text
Accumulating control: h_(k+1) = h_k + a F(h_k;t)
Anchored corrector:   h_(k+1) = h0  + a F(h_k;t)
```

Both begin at h0. The first update, all initial tensors, unique parameter counts and executed matrix calls agree within each family. Subsequent gradients and hidden functions need not agree. h0 is not detached; its gradients through every occurrence are retained. The output head and final normalization are unchanged. There is no depth embedding, loss change, learned mixing coefficient, additional feature projection, new activation, spectral constraint or altered P initialization.

The ODE branch is the professor's two scalar K2 frozen-clock solves with middle SiLU, gate hidden64, tau10, range(.2,1). The two generic controls are nominal SwiGLU192 (compute matched) and SwiGLU128 (cheaper and nearer the ODE parameter budget).

## Mathematical scope

For F(h)=A h and B=aA, K accumulating updates give (I+B)^K h0; K anchored updates give (I+B+...+B^K)h0. Under ||B||<1 the latter converges to (I-B)^(-1)h0. For a general smooth branch, the anchored equation is the fixed-point iteration for h=h0+aF(h;t), an implicit-Euler equation with fixed external conditioning. Its convergence needs a contraction condition; the actual network is not constrained to satisfy it. A finite K does not inherit unconditional implicit-Euler stability. Increasing K here changes approximation to a fixed equation; it is not a longer integration horizon. No claim of higher integration order is made.

Check actual linearized-forward algebra, input and parameter gradients, exact accumulated-control optimizer trajectories, repeated-call costs, checkpoint restore and native pipeline dispatch before launch. Record hidden changes and distance to input across thirteen evaluations on the fixed probe bank. For the anchored rule the next-minus-current change is its fixed-point residual; it is not a numerical accuracy certificate or evidence of task quality.

## Fixed Phase A and execution

Run exactly six arms on rotated Product16: accumulating/anchored crossed with ODE K2, SwiGLU192 and SwiGLU128. Keep seed0,5000 updates,batch512,uniform t, independent Gaussian source, cached training targets, pair velocity target, AdamW3e-4/defaults, wd0, global clip1. Keep common surrounding initialization and actual batch hash chains. Use two one-thread CPU workers, monitor processes/memory every10s, freeze source and checkpoints0/1000/2500/5000. Previous measured tiny-model throughput motivates CPU; do not reserve an idle GPU.

Initial host is Mac mini; Lenovo remains offline. Never duplicate an assigned cell. If Lenovo recovers before an eligible later task starts, move a complete later dataset group with all its controls to that host after native preflight. Do not mix model effects with host differences.

Use identical validation banks and field/Jacobian/Heun32-SW2 evaluations as the completed recurrence study. Check input and bank hashes. Reuse the complete previous generic dense frontier only after verifying compatible data, input streams, construction and native endpoint replay; otherwise no pooled comparison. No new test-bank evaluation in this screen. Save per-visit P/activation diagnostics as before. Record field errors at1/2/3/6/12 iterations; the registered training-depth endpoint remains primary.

## Continuation and stop

Phase A advances only the anchored ODE if, at its trained three calls:

- its primary field MSE is at least10% below the best generic endpoint from the six new arms and the verified previous dense frontier;
- its Jacobian error and SW2 are each within10% of the best generic value in that combined comparison;
- its dense forward MACs are no more than5% above the active SwiGLU192 control, with the cheaper SwiGLU128 tradeoff also exposed;
- finite outputs, checkpoint replay, same initialization/stream and cost checks pass.

If no ODE endpoint passes, stop this setting. Do not retune the step, iteration count, loss, width or optimizer to rescue it. An improvement over the old ODE alone is descriptive progress, not nomination. Generic anchoring gains do not establish an ODE contribution.

Only a Phase A nominee triggers Ring8 and Gaussian with the same six arms. It must beat the generic field frontier by10% on at least one nonlinear task, stay within5% on the others, and keep nonlinear Jacobian/SW2 within10% of the generic frontier. Then require20000-update continuation, a fresh frozen test,8192-sample Heun32/128 checks and isolated measured throughput/memory before accepting practical usefulness. No shorter-depth or best-checkpoint result substitutes for this endpoint.
