# Cycle8 registration: can the model adapt to weak spectral relaxation?

Cycle7 rejects flattening a trained checkpoint as a general repair. That is
different from allowing all parameters to adapt while a weak P constraint acts
during optimization. This is one bounded test of that remaining professor-led
question, not a response that adds a new gate or loss to rescue a failed model.

## Fixed design: twelve trajectories

Keep the native scalar K2 and raw K1 models, with their already fixed Muon rates:
Ring8 scalar .006/raw .02; Product16 scalar .02/raw .006. For each of these four
parents run three arms: unchanged native updates, shape relaxation, and scalar
contraction matched to the shape transform's resulting operator norm.
No rate or strength search. Keep seed0, 20k updates, batch512, FM loss,
auxiliary AdamW, clipping and all existing data/evaluation settings unchanged.
The native anchors must reproduce the archived 20k numerical state exactly.
Retain the previous native SwiGLU64/128/192 results as ordinary controls; no
claim can be based solely on a gain over one of these ODE parents.

After every tenth native optimizer update, use the cycle7 Frobenius-preserving
singular-value blend on each P, with k=1-exp(-10/10000). The registered relaxation
time is 10k updates, half the fixed training budget. The cadence amortizes CPU
SVD overhead; it is fixed before outcomes. All arms begin at the identical native
function, with no initial projection. Norms are preserved relative to the
just-updated P, not clamped to the initial norm. The scalar arm contracts each
of its own updated P matrices to the same leading singular value its shape
transform would have produced. Across diverging trajectories these are local
counterfactual controls, not globally identical operator-norm histories.

Official Muon and native auxiliary AdamW still compute their original steps.
The relaxation is a separate post-step operation; momentum is not transported
or reset. This is an explicit optimization constraint, not a modified FM loss
or an assertion that vanilla Muon itself constrains P. It incurs additional SVD
cost and changes the update direction. Report elapsed time separately.

## Logging, verification and decision

The optimizer adapter must record the **total** post-relaxation DeltaP under the
existing delta key, with native and relaxation deltas separately. Disabling the
inner logger avoids presenting pre-relaxation weights as the final update.
Retain clipped gradients, Muon momentum and raw-P/conditional-PD histories.
No SVD is differentiated through. Every saved optimizer state must resume with
the same ten-update cadence and all native moments/RNG states.

Before formal runs verify the matrix identities, exact no-op native states and
an actual native trainer checkpoint8->16 continuation crossing the step10
relaxation boundary. Compare constrained resumed states to uninterrupted16-step
states, and native8-step anchors to their original runners. A fresh documented
CUDA witness is required on both hosts; no formal training before it passes.

Replay every final20k checkpoint with the unchanged evaluator, independently
check anchor states and the relaxed update logs, and publish all twelve results.
Validation field MSE determines parent-relative provisional decisions; test SW2
is separate. Do not nominate a model if ordinary controls or sampling reveal
the same earlier tradeoff. If this single relaxation setting fails, retain the
negative evidence rather than automatically sweeping strengths, rates or gates.

- [ ] Implement small optimizer adapter and source-bound grid; CPU witness.
- [ ] Publish/deploy immutable source; fresh documented native CUDA witnesses.
- [ ] Baseline-first monitored queues on145/Lenovo, two workers per free GPU.
- [ ] Finish/replay all12, verify four exact native anchors, then assess the
      adaptation question against parents and ordinary controls.
