# Next focused question: does Muon need the tied quadratic term?

Status: prospective protocol; no third-wave training launched. Implement and
verify before dispatch. This narrows the professor's optimizer question without
adding a new activation or another gate parameterization.

## Evidence and hypothesis

Wave2 shows that Muon beats same-state norm-matched momentum on field MSE, but
this happens for SwiGLU too. Tested decay lowers P and local Jacobians while
harming fitting. These observations leave optimizer benefit versus tied
second-order computation unresolved.

Historical orders were tested with AdamW3e-4 for5k updates on Gaussian, Ring8
and Moons (docs/research-2026-10-20/protocol.md). That study found no uniform
higher-order advantage; it did not cross Muon with order on the current budget
and Product16. This is the specific untested interaction.

At fixed layer entrance h, s=s(h), and all controller parameters held fixed,
perturb P by E. The original second-order operator has directional derivative

```
F2 = h + s P h + (s^2/2) P^2 h
D_P F2[E] = s E h + (s^2/2) (E P + P E) h
D_P F1[E] = s E h
```

The quadratic term couples the matrix update to the current P from both sides.
Muon reshapes the update matrix; this coupling is a reason to test an
optimizer-by-order interaction. It is not a proof that orthogonalized updates
help the quadratic model or that any benefit is unique to an ODE interpretation.

## Eight new configurations, subject to readiness

Train K1 from the same scalar-K2 initialization, changing only each ODE solve's
order attribute to1. On each of Product16/145 and Ring8/Lenovo use selected-matrix
AdamW0.0003/0.001 and official Muon0.006/0.02. This is four arms per dataset,
eight total; reuse the verified K2 controls and both ordinary widths/rates.
Use seed0,20k updates,batch512, identical data stream, auxiliary optimizer,
clipping, evaluation and checkpoint schedule. No decay, new gates, or LR schedule.

Before dispatch verify identical cold tensors and RNG state, F1 dense algebra,
full entrance-gate derivatives, unchanged K2 updates, and native CUDA checkpoint
continuation. Freeze a separate worktree and use two monitored workers per host.
Do not alter completed wave2 source/results. Record wall time and MACs because
removing a matrix application changes compute per step.

Primary comparison: K1 versus K2 at every matching optimizer/rate, accompanied
by the difference in Muon-versus-Adam effects across orders. Report validation
field MSE and independent test SW2 separately, all rates, fixed20k endpoints,
and complete curves. Do not pick a checkpoint on test distance.

If K1 retains the benefit and matches K2, weaken the Taylor/second-order story.
If K2 improves consistently over K1, it still must beat ordinary controls before
claiming an FFN advantage. If rankings depend on rate/task/metric, retain those
interactions. Do not automatically expand to K4/K8 or image transfer.
