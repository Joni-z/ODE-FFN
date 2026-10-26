# Forced SwiGLU gate: bounded experiment review

**Conditional GO for one fixed five-arm comparison, after the separate mathematics/prior-art review.** The question is whether a small, function-matched perturbation of the stronger SwiGLU reference is useful. Existing evidence does not justify expecting a gain, but it does not already answer this contrast. No code or training was changed for this review.

## Why this is a different, still unproved question

The [SiLU reference](analysis-reference-silu/analysis-report.md) has Product16 MSE0.153892 versus SwiGLU128's0.092326, and Ring8 MSE0.033714 versus0.015296. The [diagonal sandwich](analysis-diagonal-flow/analysis-report.md) failed to improve that weaker SiLU base. Its independent feature/value product was absent. The proposed gate retains the actual SwiGLU computation `U[SiLU(a)*v]+bU`, including independent affine gate/value projections. Thus neither Stage19 nor the failed [shared-P study](analysis-shared-features/analysis-report.md) directly rules it out. These results support preserving the stronger reference, not choosing phi1 by an FM theorem.

## Minimal controls

Use nominal128, actual hidden85, for every modified arm. Keep the existing SwiGLU96 reference; do not sweep modified widths.

| Arm | Gate replacing b=SiLU(a) | Role |
|---|---|---|
| SwiGLU96 | b | Established smaller baseline |
| SwiGLU128 | b | Exact K1 limit and matched baseline |
| Static | b*phi1(.5*lambda) | Active redundant parameterization |
| Generic K2 | b*(1+.5*lambda*sigmoid(a)) | Same initial adaptive direction without higher terms |
| Exact forced gate | b*phi1(lambda*sigmoid(a)) | Higher-term contrast |

Here lambda=tanh(eta), eta starts at zero, and phi1(z)=expm1(z)/z with its smooth extension at zero. K1 is already the unmodified SwiGLU arm: do not allocate unused eta or run a renamed duplicate. No extra generic offset arm is needed in this first comparison. K2 already supplies the exact leading adaptive perturbation and initial eta tangent; a further generic activation grid would answer a broader question.

Static phi1 is a positive per-channel gain absorbable exactly into U's columns. It has the same function class as SwiGLU, while its extra trainable coordinates change optimizer geometry. Static eta's initial response is `.25*b*v*U[:,i]`, redundant with output-column scaling. It is an active control, not dead parameter padding.

## Initialization, tangent and implementation fairness

Construct the existing fused `w12` and `w3` in precisely their current order, then add only a zero eta vector without RNG draws. Keep the native `F.silu(a)` evaluation; rewriting it as `a*sigmoid(a)` can change floating-point results. Do not replace fused w12 by separately constructed Linear modules. Preserve biases, state keys, independent a/v rows and existing non-FFN reference copying. All modified arms must match SwiGLU128's initial function, input Jacobian and **unclipped common-parameter gradients**, checked with a common nonzero output head. The formal zero velocity head otherwise hides all initial backbone gradients.

Exact and K2 have the same initial extra response:

```text
d output/d eta_i = .5 * U[:,i] * v_i * a_i * sigmoid(a_i)^2.
```

They therefore share the complete initial parameter Jacobian. This is not universally redundant with baseline parameters when a and v are independent. It can be redundant in special cases: at a=v=x, the scalar response is half of `d/value_bias + d/value_weight - d/gate_bias`. No wide-network approximation or full-dataset rank claim follows from the one-unit distinction.

Adding eta changes the total gradient norm seen by the trainer's global clipping. Common raw gradients matching the baseline does not imply identical clipped Adam updates. Exact/K2 should match each other's initial full gradients; static and baseline need not match their full kernels.

**Critical numerical check:** phi1's zero branch must preserve phi1'(0)=1/2 and phi1''(0)=1/3. A constant `where(z==0,1,expm1(z)/z)` branch can suppress eta's starting gradient and leak NaNs from the unused division. Use a smooth local series with safe denominators, then verify zero/near-zero derivatives and nonzero eta updates in a short trainer/resume smoke. Forward equality alone is insufficient.

At width64/hidden85, baseline FFN parameters/MACs are16554/16320; each modified FFN adds85 parameters and no dense matrix MACs. Product16 totals become83853/81792 versus baseline83598/81792. Extra sigmoid, tanh, polynomial or exponential work remains real; equal dense MACs are not a speed claim. Preserve the current training/data/reference protocol and monitoring. Use the established audit machinery rather than expanding the preflight into another research project.

## Predeclare the practical distinction

One five-arm run on the three current tasks is enough for the first decision. Preserve all curves and fixed endpoints. A substantial K2 or exact gain over both SwiGLU sizes **and static** may nominate a generic ODE-inspired activation lead, subject to the existing cross-task/Jacobian/SW2 safeguards and subsequent fresh-holdout and timing confirmation. Only an additional exact-versus-K2 gain supports a benefit from higher terms; register a margin before training. K2 alone succeeding is not evidence for exponentiation. Neither succeeding closes these settings without a clock, lambda-range, width or order rescue. No architecture is accepted from this review, an initial tangent, or a forced-ODE interpretation.
