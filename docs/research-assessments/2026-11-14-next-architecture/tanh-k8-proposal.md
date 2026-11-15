# Proposed CelebA candidate: Tanh-D K8

**Superseded by the user-requested [response-family comparison](response-screen-proposal.md).**
Retain tanh as one candidate and comparison anchor. The direct tanh-only
image run below is not selected or submitted.

15 November 2026. One candidate proposed in response to the request for a
concrete image architecture. No implementation or training is launched by
this document. The ISRU-midpoint pair is not the selected next comparison.

## Operator and FFN

Keep the widened ISRU-D K8 topology and replace only its entrance diagonal:

```text
c = sqrt(1.2)
d_i(h) = tanh(c*h_i)/(c*h_i), with d_i(0)=1
A(h) = P diag(d(h))
F8(h) = h + A(h)h + A(h)^2 h/2! + ... + A(h)^8 h/8!
```

D is computed once at each solve entrance and frozen across its terms;
gradients still pass through D. This is a frozen-generator Taylor map,
not a refreshed nonlinear solver or the original scalar-clock s(h)P layer.
Copy the already verified near-zero tanh-ratio implementation from
`studies/muon_pd_20261102/followup_20261115/train.py`.

Full FFN: 768 -> linear 1200 -> Tanh-D K8 -> SiLU -> Tanh-D K8 ->
linear 768. The two solves have independent P matrices; each P is reused
across its own eight terms. No new trainable tensors are needed. Expected
total parameters remain 130,402,560 and FFN dense MACs match widened ISRU-D
K8; verify counts and benchmark pointwise/total cost before a formal run.

## Reason for selecting this comparison

At the registered synthetic 20k endpoint, tanh-K8 has lower field MSE and
pooled SW2 than ISRU-K8 on both tasks. Pooled SW2 is 0.14090 versus 0.16779
on Ring8 and 0.06427 versus 0.07321 on Product16. Earlier checkpoints are
mixed; this is a concrete transfer hypothesis, not an established image gain.

The question is whether the response shape contributes to the observed image
gap while the PD structure, order, width and optimizer stay fixed.
This test neither establishes PD superiority over the original scalar clock
nor an ODE-specific advantage. The [synthetic evidence](../2026-11-15-order-response/findings.md)
and [qualified image references](../../celeba128-wide-20261114/continuation-160/matched-results.md)
remain separate.

## Proposed comparison and decisions

Use unchanged author JiT-B/16, unconditional CelebA128, internal width1200,
seed0, batch256, AdamW and the exact existing image recipe. Start from the
same fresh initialization tensors as the existing wide ISRU-D K8 construction;
do not fine-tune its trained checkpoint. Verify RNG and backbone identity.

Propose one new 100-epoch trajectory on Torch. Reuse the fixed native SwiGLU
and widened ISRU-D K8 as the practical and response controls. Evaluate FID50k
every20 epochs using the same EMA, reference, sampler and sampling seed.
Only add the FFN selector and necessary existing-wrapper plumbing.

Before training: inspect existing checkpoint response/error profiles, verify
the copied diagonal and full derivatives including zero, compare parameters
and initial tensors, test save/resume and evaluation RNG, benchmark on the
same GPU, then register walltime and monitoring. These are readiness tasks,
not additional architecture searches.

At100 epochs, native FID is35.91785 and ISRU-D K8 is38.13574. Beating only
ISRU-D identifies a useful response change but does not qualify the FFN.
Beating native makes a160-epoch confirmation worth considering against
native24.30357 and ISRU-D K8 28.08343. A tiny lead needs persistence and measured
cost; K8 still costs substantially more FFN matrix work than native.
If native remains better at100, end this candidate at that budget without
a width, order, optimizer or coefficient sweep. Further runs require a new
evidence-based decision. Finalized reports and all existing controls stay fixed.

