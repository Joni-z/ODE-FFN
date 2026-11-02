# Cycle4 registration: what feature does the diagonal provide at K1?

Registered after cycle3 replay and before any cycle4 outcomes. Source and20k
protocol stay fixed; only the deterministic input-diagonal function changes.
The immediate evidence is a small Ring8 raw-K1 field lead, a sampling deficit,
and no Product16 ordinary-control win. The scalar8/8 order result weakens a
tied-second-order mechanism. Do not promote a new method from this premise.

## Three counterfactuals

Keep the two-solve/SiLU FFN and K1. Use the same P/common initialization as rawK1,
remove the controller completely as in rawK1, and add no learned parameters.

| Family | d(h) | One solve F(h) | Question |
|---|---|---|---|
|pd_constant_k1|1|h+P h|Is input dependence needed at all?|
|pd_direct_bounded_k1|.2+.8sigmoid(h/10)|h+P[h*d(h)]|Does bounding the direct input help after removing K2?|
|pd_direct_shift_k1|h+1|h+P(h*h)+P h|Does a tied linear feature address the raw quadratic limitation?|

The fixed shift1 is not fitted or tuned. This is a family contrast, not a pure
causal separation of sign, amplitude and derivatives. Different d give different
initial functions and effective step scales even with identical P. Constant1 is
an ordinary linear residual control within the same two-solve skeleton; the
SiLU between solves and surrounding native backbone remain present.
In fact its full FFN is exactly a bias-free two-linear-layer SiLU FFN with
W1=I+P1,W2=I+P2. Interpret a win as an ordinary feature/control outcome, not a
new ODE architecture.

For bounded d, J=I+Pdiag(d+h*d'); for shifted d, J=I+Pdiag(2h+1);
for constant d, J=I+P. Native witnesses check these dense full derivatives at
zero and signed inputs, not only the matrix polynomial with frozen d.

## Registered cells and controls

Each family crosses AdamW.0003 and Muon.006/.02 on Ring8/Lenovo and Product16/145:
18 new trajectories. Reuse completed rawK1, rawK2, boundedK2, scalarK1 and all
ordinary64/128/192 controls. Swi64 and rawK1 are close in both parameters/MACs.
Dataset caches, seed0, batch512,20k final checkpoint, optimizer groups, clipping,
validation and Heun32 sampling remain unchanged. No new loss, spectral penalty,
decay, scheduler, embedding, packages or image training.

Provisional decisions use validation only: constant and shifted compare to
rawK1 at matching optimizer/rate; boundedK1 compares to its boundedK2 parent
for an order contrast and additionally reports rawK1. Require >=5% validation
improvement for provisional keep. Sampling tradeoffs are always reported.
Accepted generative advantage still requires the ordinary controls and both
tasks; changing the denominator is not a rescue. Publish every endpoint.

## Execution plan

- [x] Implement only cycle4/trial4.py numerical adapter and fixed grid.
- [x] Verify exact inherited tensors/RNG, full derivatives, native optimizer
      checkpoint resume, original parent arm and actual8-to16 trainer continuation.
- [x] Freeze/push runtime; deploy new detached worktrees and fresh documented
      CUDA witness on both hosts; never overwrite completed runtimes.
- [ ] Run two monitored workers per free GPU, complete all18, replay independently.
- [ ] Analyze D shape, optimizer interaction and ordinary controls; choose the
      next question from complete outcomes rather than a favorable test sample.

Local preflight diagnosis: the first adapter reused the archived feature_mode
attribute, activating its unrelated shared-controller path. The dense derivative
witness caught missing t_func before deployment or training. Renamed the new
attribute diagonal_mode; archived source remains unchanged. Preserve this failed
preflight as a code issue, not a training trial or a quality result.
The next check reached a witness-only parent lookup error: cycle3's fallback
does not expose the historical bounded-K2 name. The witness now imports that
unchanged cycle2 builder explicitly. Neither issue reached a formal queue.

CPU witnesses pass: all three full derivatives, exact inherited weights/RNG,
eight original optimizer-arm equivalences, and physical model/optimizer/RNG
continuation. Full native8-to16 trainer smokes remain required on both GPUs.
