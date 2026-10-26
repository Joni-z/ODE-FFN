# Function-preserving adjoint implementation and a gate case study

Registered after the completed derivative-orientation study, before the implementation/case-study outputs. These are frozen-weight mathematical/computational experiments; no new architecture success or cold-training effect can be inferred.

## Question and mathematical identity

For the same raw network G and fixed outer time t, reproduce

```text
A(x,t) = .5*(G(x,t) + J_G(x,t)^T x).
```

Use an explicit forward pass and reverse sensitivity sweep. No input-autograd call occurs inside the evaluator. All ordinary tensor operations retain their graphs, so input/time and parameter derivatives of this new implementation remain exact. The underlying parameters, buffers, primal graph and potential do not change.

For the original scalar K2 solve,

```text
L(h)=h+a(h)Ph+a(h)^2/2 P²h
u=Ph+aP²h
J_L^T q=q+aP^Tq+a²/2(P^T)²q + grad(a)*dot(u,q).
```

The last term is an input-clock sensitivity. SiLU, LayerNorm, FiLM, residuals and projections require their full reverse rules in the correct reverse order. Finite differences must include the dependence of the output seed q=x on x. Time t stays fixed in this input adjoint, but ordinary derivatives of A with respect to t must remain correct. Both SwiGLU128 and192 receive exact corresponding implementations; any efficiency comparison uses those controls.

## Validation before task interpretation

Use nonsymmetric affine and nonlinear algebra identities, full nonzero-head trained ODE/SwiGLU weights, and actual parameter/input/time derivatives. Compare values, complete input Jacobians, all parameter loss gradients and at least one actual AdamW update in float64. Verify float32 compatibility on actual task banks, batch independence, no_grad and inference_mode. A manual implementation must not quietly inherit native forward-AD LayerNorm failure. Failures are preserved and investigated.

The exact sweep is evaluated on A-trained and B-trained endpoints for all three families/tasks, with full validation fields and all five512-input derivative panels. On ODE A endpoints only, additionally remove the clock-derivative contribution from the reverse rules for all six ODE solves. Gate values still depend on their actual inputs. This defines an explicit altered vector function, not a detached-gradient training trick. Save per-input native/altered predictions and the exact risk decomposition:

```text
error(altered)-error(exact)
  = 2*mean((exact-truth)*(altered-exact))
    +mean((altered-exact)^2).
```

No gate-omission result is a candidate or proof of retrainability. A large penalty shows dependence of a co-adapted frozen function, not that gates caused the cold-training improvement. Do not select masks, multipliers or further ablation combinations after the result.

## Computational experiment

Report actual executed dense MACs for inference and a pair-target loss forward/backward. Isolated CPU timings compare autograd A and explicit A on the same frozen state and deterministic batch512, with one Torch thread, equal warmup/repeats, alternating method order and no concurrent trainer. Report repeated timings, no single fastest result. Correctness precedes timing. Explicit arithmetic can trade matrix work for pointwise work; no speed guarantee is assumed.

Mac mini handles this bounded evaluation; Lenovo's current Tailscale state is offline. No GPU allocation. Save source/checkpoint/bank hashes and monitor actual process/RSS. Do not relaunch after an observation timeout. After source validation, freeze the implementation used for the case study. Any new training needs a distinct written hypothesis and matched generic controls; a failed A/B setting is not continued.
