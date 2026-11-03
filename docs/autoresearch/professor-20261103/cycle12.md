# Cycle12 registration: frozen outer-transform swaps

Cycle11 is complete and no new candidate qualifies. Its Product16/AdamW corner
has a controller/output interaction despite similar P/PD/response/range summaries.
Test whether a single present-day output-transform replacement repairs the fixed
trained function. This is read-only sensitivity analysis, not another cold-training
method, a range-strength sweep or continuation of the rejected radial setting.

## Fixed cohort and one intervention

Use all24factorial checkpoints: four controller/output corners, two tasks and
AdamW.0003/Muon.006/Muon.02. Each is final20k and hash-bound to its source CSV.
Evaluate its native output and exactly one opposite sigmoid/affine output.
Total48evaluations,24native anchors/24swaps. No favorable family, rate or
checkpoint is selected. All tensors, runtime buffers, P, internal controller
parameters, middle SiLU, K2, data and native field/sampler stay unchanged.

For a common controller logit g,

```
d_sig(g)=.6+.4 tanh(g/20)
d_aff(g)=.6+.02g
d_aff-d_sig=g^3/60000+O(g^5) near zero
(d_aff'-d_sig')=.02 tanh(g/20)^2.
```

The derivative equation is scalar logit sensitivity, not a positive-semidefinite
ordering of full network Jacobians. On a whole-network swap later hidden states
and logits are recomputed; they are not held at their native values. The operation
changes output values, bounds/signs and feedback derivatives together, so it cannot
isolate saturation alone or decompose the full training cause.

## Native identity and numerical checks

Before interpretation every native field/test endpoint must replay its bound
checkpoint; generated/target arrays must match archived examples. Apply the
involution twice and verify exact restored probe predictions and unchanged
parameters/runtime buffers. Every solve retains its original PD placement/order.
Native nonfinite/error is a readiness failure, not a zero-quality result.
A nonfinite swapped field or rollout is retained as a failed intervention with
finite fractions and null score, never clamped, dropped or silently retried.

CPU fixtures test both scalar transfer formulas/derivatives and round-trip state
identity for all four families with a nonzero fixture readout. Fresh documented
CUDA witnesses additionally load one fixed AdamW checkpoint/family/host and
compare native/round-trip probes plus small native Heun rollouts. These are
readiness only and use separate paths. Formal results require all24native anchors.

## Evaluation and interpretation

Reuse the unchanged validation oracle and independent test sampling bank:
4096samples,256directions,Heun32. Field is the validation readout; test distances
remain separate and cannot select a model/rate/checkpoint. Save paired native/
swapped/target arrays and source/config/manifest/sample hashes. Retain native
activation/conditional-spectrum summaries for both actions; raw P is fixed.
An independent saved-array CPU replay recomputes all finite sliced-W2 values.

Report the entire48-row table, paired changes, all six factorial settings and
the earlier near-parameter-matched ordinary references. Show every Product16
AdamW corner on predeclared axes0/7/15 and all Ring8 corners at the registered
Muon.02 setting, with identical display indices. Examples do not select a winner.

If swapping does not repair the field gap, reject this immediate-output repair;
that does not prove which part of training caused the gap. A favorable frozen
swap is sensitivity evidence only, not a trained-method result. No cold-training,
new optimizer, rate/strength, loss, data, sampler, K1 or image rescue is selected.
Do not append these evaluations as training trials to the external ledger.

- [x] Implement minimal attribute-swap producer and hash-bound cohort; CPU checks.
- [ ] Publish/deploy immutable source; fresh documented CUDA/native-checkpoint
      witnesses and source/availability checks on145andLenovo.
- [ ] Run both monitored frozen cohorts, verify24native anchors, round-trip
      identity and saved-array replay; publish all48outcomes and case figures.
- [ ] Select the next question only from the completed comparison.

CPU readiness passed on all four families, including nonzero readouts, actual
native scalar transfer derivatives and exact tensor/probe round trips. All24
records/five source CSVs are checked. These are implementation tests only.
