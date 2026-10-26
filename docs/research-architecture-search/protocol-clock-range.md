# Stage8: relax only the original clock transfer range

Registered before this stage's training outcomes. This returns to the original two-stageK1 professor composition with independent vector controllers. Failed SiLU deletion, controller reuse and nonlinear-drift return variants are not combined with this test.

## Evidence and mathematical axis

The product16 native-model diagnostic shows frequent outer sigmoid saturation, including57.2% of channel/sample entries at the final vector second solve and83.8%/80.0% in two scalar second solves. Saturation may be useful; this is not proof of harm. Test that possibility rather than declare the range responsible for all error.

Keep the existing MLP logit g, and compare the family s_a(g)=c+a*tanh(k*g/a), with c=.6,k=.02:

- Original: a=.4, the existing sigmoid with shift.2,scale.8,tau10; range(.2,1).
- Wide positive: a=.6, existing sigmoid with shift0,scale1.2,tau15; range(0,1.2).
- Affine limit: a=infinity, implemented as s=.6+.02*g; signed and unbounded.

All share s(0)=.6, s'(0)=.02 and s''(0)=0. Their first difference is cubic: -k^3*g^3/(3a^2). Initial hidden functions are locally matched, not globally identical. Preserve the old sigmoid implementation exactly in the reference so its historical endpoints remain reproducible. No trainable tensor, RNG draw or linear MAC differs between the three ODE arms.

The affine controller retains the MLP's internal SiLU. It removes only the outer saturation and permits signed modulation. The positive wide control changes both bounds, so it does not isolate a lower-floor effect by itself. At a fixed input all arms retain A(x)=D(x)P and F1(x)=x+D(x)Px; their full Jacobians include diag(Px)J_s. These are hidden-operator statements, not proof that a bounded gate makes the full network unable to represent the target. Signed vector durations no longer have a common positive elapsed-time interpretation. No invertibility or stability certificate follows fromK1.

## Five arms and frozen training

`vector_k1`, `vector_wide`, `vector_affine`, `swiglu_budget` (nominal hidden96) and `swiglu` (nominal hidden128). At dim2 the ODE arms have81858 parameters/80192 MACs; full SwiGLU81792/80000 and budget SwiGLU69570/67904. Thus a gain over an ODE parent alone is not the useful frontier.

Datasets: original Gaussian2D/Ring8 v1 and frozen product16 v1. Train all five on each, seed0,5000updates,width64,depth3,gate hidden32,batch512,AdamW3e-4,wd0,clip1 and unchanged time/source pairing. Final5000 random-time validation oracle MSE is primary. The same common512 states/five times, Jacobian error, sample SW2, P trajectories and cost checks remain. Use two monitored CPU workers, immutable source and new output directories. No test bank is opened and no failing job is restarted automatically.

Log duration sign/range use, outer-transform saturation and duration/logit sensitivity normalized to its initial value. For affine gates, zero outer-saturation does not imply that all gradients or inner MLP activations avoid saturation. Negative/large effective durations are observations, not a reason to silently clamp them or change clipping/lr. Preserve failures on nonfinite loss/gradient.

## Advancement and rejection

An ODE range arm may advance only if it lowers Ring8 or product16 oracle MSE by at least10% relative to both SwiGLU budgets, stays within5% of full SwiGLU on the other nonlinear task and Gaussian control, and is within10% of the better SwiGLU common-Jacobian and validation-SW2 metrics on both nonlinear tasks. These are engineering screens, not significance claims. A candidate dominated by the narrower baseline is not accepted because it beats the larger one. Report all five arms and all metrics.

If both relaxed arms pass, choose the one with the lower geometric mean of Ring8/product16 primary errors. If neither passes, stop this range branch; no K2, spectral cap, controller widening, new loss or altered gate family is added to rescue it.

Passing fixes the range before testing the corresponding scalar-clock control, then continues the unchanged lead and references to20000 with exact optimizer/RNG state. Before a completion claim, use fresh independent target/source evaluation banks (SeedSequence([0,dataset_identifier,20261024,8]), disjoint streams), actual latency/throughput, effective D(x)P and complete-field derivative/generation checks. Historical exposed test banks are not called untouched confirmation. Higher Taylor orders remain a separate question.
