# Does sharing remove useful depth-dependent P directions?

Registered before this diagnostic's outcomes. It follows a negative architecture experiment, not an extension of that setting. Saved shared-potential ODE error is.23353 versus function-matched cloned-potential .03423; corresponding SwiGLU penalties are smaller. This does not itself prove gradient interference or select an internal-depth conditioner.

## States, data and exact matched functions

Use all six shared trajectories on Product16: ODE K2/SwiGLU128/SwiGLU192, direct/complete-gradient readout, at steps0/1000/2500/5000. The two old direct ODE/Swi192 trajectories retain their recorded output-product paths; other states come from the new recurrent-potential runtime. Every snapshot's data RNG must match the regenerated training stream.

At each state, form three independent copies of its actual shared block, keeping all common parameters and buffers unchanged. This cloned function must reproduce outputs and complete input derivatives; it is not the independently trained cloned endpoint. For a batch loss, shared block gradient equals the sum of the three independent-copy gradients, and every common-parameter gradient agrees. No parameter is updated.

Starting at that snapshot's saved data RNG, generate two successive banks A and B, each eight batches of512, using the exact cached-target/uniform-time/Gaussian-source training law. Save actual target/source/time arrays and stream hashes. Use the same banks for every model at a given step. Bank A defines gradient projections; B checks their directional transfer. These are training-objective diagnostics, not a held-out generalization or new test score. Preserve individual-batch and bank-mean results; do not replace noisy pair labels by a privileged teacher.

## Depth metric and groups

Let g1,g2,g3 be gradients with respect to independent copies of the same block coordinates. Under the Euclidean metric that charges all three parameter copies, use the orthonormal depth basis

```text
u0=(1,1,1)/sqrt(3)
u1=(-1,0,1)/sqrt(2)
u2=(1,-2,1)/sqrt(6).
```

E0, E1 and E2 are the squared norms of the three gradient coefficients. Their sum equals total independent-copy gradient energy. The tied subspace retains fraction E0/total; an affine-in-depth parameter subspace retains (E0+E1)/total. Also record all pairwise dot products/cosines and the exact shared-gradient sum. At the true zero-output-head initialization, block gradients can be exactly zero: report undefined fractions as missing, not as conflict or perfect agreement.

Use actual native parameter coordinates. Disjoint ODE groups are P, clock, block normalization and time-condition projection. SwiGLU groups are input/gate projection, output projection, normalization and condition. Retain aggregate FFN and complete-block values, clearly marking them as overlapping summaries. These groups are not equivalent function classes across families. Euclidean projections do not model Adam's full update geometry or establish an optimization theorem.

For each group, bank A supplies unit-norm descent directions in the duplicated coordinates for the common-only subspace and common-plus-linear-depth subspace. Record their bank-B gradient alignment, normalized by the bank-B gradient norm. This measures a first-order direction on a second pair bank; it is not an actual optimizer rollout or a final task-quality result.

## Prospective direction rule

Only one possible new structural lead is screened: allow each original P to vary affinely with internal computational depth, preserving the original layer/controller/readout otherwise. This is not selected unless all of the following hold at both2500 and5000 for the shared-gradient ODE's bank-mean gradients:

1. P retains at most50% of its independent-copy gradient energy under tying, and contributes at least50% of the complete block's energy outside the tied subspace.
2. The linear-depth mode captures at least70% of P's lost energy, E1/(E1+E2).
3. Bank-A common-plus-linear P descent aligns positively with bank B by at least.2 and improves normalized alignment over common-only by at least.05.

Keep individual-batch consistency, direct ODE and both generic readouts visible. These thresholds are a bounded design-selection convention, not statistical significance or a proof of causality. If the rule fails, do not launch P-depth training or select another parameter group/mode after outcomes. A pass only warrants a separately registered matched cold experiment with both SwiGLUs and explicit parameter/compute controls; it is not a successful architecture.

## Verification and resources

Before interpretation, verify function/gradient identities on every actual batch/state, Parseval/group energy identities and finite loss-direction derivatives for representative nonzero states. Preserve all block gradient vectors, coordinate names, modes, input/checkpoint/source bindings and complete summaries. Use float64 for algebra from actual saved weights. One Torch CPU thread, bounded memory and process telemetry on Mac mini; no GPU allocation while Lenovo remains offline. Do not restart a live process on an observation timeout.
