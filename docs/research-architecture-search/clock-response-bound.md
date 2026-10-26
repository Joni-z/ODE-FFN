# Fixed response directions are distinct from clock count

Let the six scalar durations be c(x) and f(x)=G(x,c(x)). At a trained anchor set U=partial G/partial c and V=dc/dx. Holding weights and native clock values fixed gives

```text
J_full = J_anchor + U V.
```

The rank-only relaxation allows any rank-six correction. The actual fixed response U restricts its column space as well. For target residual R=J_oracle-J_anchor, allowing arbitrary V gives the least-squares solution U*U^+*R and the exact conditional floor

```text
min_V ||R-U V||_F^2 = ||(I-U U^+) R||_F^2.
```

Its value is at least the unrestricted rank-six singular-value tail, and at most the error of the actual network. Compute U by treating all clock outputs as independent differentiable inputs to the complete network, preserving the native forward output. The measured actual correction projects into this response space with maximum element discrepancy below7e-15. This checks the chain-rule decomposition through all conditioning, normalization, activations and residuals.

## Product16 scalar checkpoint

All values use the frozen20000-step scalar model and the same512 validation anchors at each time.

| FM time | Actual Jacobian error | Rank-only floor | Fixed-response floor | Response floor / error |
|---|---:|---:|---:|---:|
| 0.1 | 0.24432 | 0.02558 | 0.13490 | 55.2% |
| 0.3 | 1.57576 | 0.17743 | 0.80274 | 50.9% |
| 0.5 | 14.28745 | 1.68832 | 8.23934 | 57.7% |
| 0.7 | 71.71839 | 5.21053 | 56.67578 | 79.0% |
| 0.9 | 28.55430 | 2.50736 | 24.38989 | 85.4% |

The rank count alone is a loose relaxation here. At t=.7, an unrestricted rank-six correction has floor5.21, but the trained response directions have floor56.68 against actual error71.72. Even unrestricted clock derivatives cannot attain the rank-only optimum while this response frame is fixed.

This does not establish that P alone causes the error: U depends on all trained weights, clock values, the anchor, and the complete backbone. Retraining changes U and J_anchor. Changing clock values also changes the response. The bound therefore does not invalidate the running range experiment, which retrains the whole model. It does reject interpreting outer-sigmoid saturation as a sufficient explanation of the frozen scalar error.

For the vector model the response space can already span the16-dimensional output, making this particular column-space bound vacuous. Global function fitting and optimization restrictions remain. No vector advantage or new architectural success follows from a zero relaxed floor.

The read-only program is scripts/diagnose_clock_response.py. Tests cover a known wrong response plane and the complete-network correction-span invariant. Exact summaries are in analysis-operator/clock-response-summary.json; raw response matrices and hashes are preserved beside the product16 evidence outside Git. No new loss, spectral constraint or training trick was added.
