# Image-transfer recommendation: ISRU-D K1

Status: architecture recommendation, 11 November 2026. No training dispatched.

Subsequent [Taylor-order clarification](2026-11-11-taylor-order.md): K1 is a
performance candidate, not the settled final order. The user asks to preserve
the original closed-form-approximation question. ISRU-D has a frozen-diagonal
exponential interpretation, and its higher orders have not been ruled out by
the old scalar-clock K4/K8 study. The transfer order comparison remains open.

The user wants a concrete architecture to test on a larger image dataset.
Recommend transferring the existing ISRU-D K1 FFN now, alongside ordinary
controls. Complete causal attribution on synthetic data is not a prerequisite
for this exploratory transfer. Image performance remains to be measured.

## Exact operator

For each token of width d, independently:

```text
q(u) = u / sqrt(1 + 1.2 * u^2)       # elementwise; fixed coefficient
a    = h + P1 q(h)
b    = SiLU(a)
FFN(h) = b + P2 q(b)
```

P1 and P2 are separate, bias-free d-by-d matrices. Keep their existing
Linear/Kaiming-uniform initialization distribution; do not zero-initialize
both matrices or introduce extra projection, normalization or trainable rate
parameters. The input and output shapes are identical. The complete JiT block
retains the author's pre-normalization, conditioning and outer residual gate:

```text
z = modulate(RMSNorm(x), shift(t, c), scale(t, c))
x = x + gate(t, c) * FFN(z)
```

Here c is absent/dropped for unconditional training. Attention and the rest of
the author's block remain unchanged. The FFN receives the already conditioned
features; it does not introduce a second global-time controller.

This is the existing synthetic design, not a new combination of tricks. The
[cycle13 builder](../../studies/muon_pd_20261102/autoresearch_20261103/cycle13/trial13.py)
replaces the input diagonal in two K1 maps around SiLU. Its inherited
[forward](../../studies/muon_pd_20261102/reference/synthetic/models.py)
applies the diagonal before P. Numerically compare a minimal image-module copy
with that archived operator before training. Preserve historical RNG records;
matching the initializer distribution does not reproduce old RNG positions.

## Why choose it

q is the exact scalar response of dy/dr = -y^3 at r = 0.6. It is bounded and
has derivative (1 + 1.2u^2)^(-3/2), between zero and one. This compresses the
features sent to P without imposing a spectral constraint on P. The complete
residual FFN is not thereby guaranteed to be bounded or contractive, and it
is not an exact solve of a coupled linear/nonlinear flow.

The engineering hypothesis is that a compact residual FFN with this response
can preserve image-generation quality at less FFN matrix work than native
SwiGLU. Test this directly rather than asserting image benefits from synthetic
MSE. Ring8 gives MSE 0.009773 versus selected Swi64's 0.011279; sampling nearly
ties, and Product16 favors ordinary controls. These results motivate a trial,
not a guarantee. See the [complete comparison](../autoresearch/professor-20261103/cycle14-results-complete/analysis-report.md).

The initial recommendation uses K1, fixed coefficient 1.2 and the existing
SiLU placement. K1 alone does not test the value of accurate exponential
approximation; the order comparison needs a separate explicit decision as
described above. Keep Muon, adaptive frames and additional spectral penalties
out of that structural comparison. Use the same upstream AdamW recipe across
the image arms.

## Capacity and minimum comparisons

At fixed model width d, ISRU-D has 2d^2 matrix weights. Native JiT's ratio-4
SwiGLU has three projections of intermediate width floor(8d/3), approximately
8d^2 weights. ISRU-D therefore has roughly one quarter of the native FFN's
dense matrix work; this is not a whole-model FLOP or wall-time claim.

For d = 768, ignoring biases and pointwise operations:

| FFN | Matrix weights / dense MACs per token | Purpose |
|---|---:|---|
| Native SwiGLU, intermediate width 2048 | 4,718,592 | Author architecture and practical reference |
| ISRU-D K1, two 768-by-768 maps | 1,179,648 | Selected candidate |
| SwiGLU, intermediate width 512 | 1,179,648 | Ordinary control at equal FFN matrix budget |

Retain the original SwiGLU equation in the smaller control; change only its
intermediate width. Do not report a lower parameter count as a speedup without
timing full training and generation. Free ISRU remains a useful attribution
control if ISRU-D passes the first image screen; its initialization and
residual placement must be recorded separately.

## Image protocol and launch prerequisites

The recommended destination is ImageNet 256 with the pinned author JiT-B/16
implementation. The [author README](https://github.com/LTH14/JiT/blob/cbc743a2ada5e9762697da2c83f8c4f8379e8c17/README.md),
read directly for this decision, provides a class-conditioned 600-epoch recipe
tested on eight H200s. That is a reference recipe, not a measured budget for
145 or Lenovo. No runtime estimate is made here.

The user's standing preference is unconditional training. Retain it unless
the user changes it: fully drop class labels and sample without guidance.
This is an explicit change from the published conditional recipe, so published
conditional FID does not qualify the local unconditional baseline. Keep the
same task, optimizer, noise/time law, EMA and sampler across all comparison
arms. Use seed 0 only.

Before selecting a long run, establish dataset availability and available
devices on 145, measure full-step throughput/memory, and register epochs,
effective batch, evaluation checkpoints and a finite compute budget. Lenovo
can provide compatibility and short-run checks; do not assume its 4060 can
efficiently carry the full ImageNet study. No hardware availability was checked
or resource allocation made in this selection turn.

The first screen needs native SwiGLU, ISRU-D and budget-matched SwiGLU under
the same training schedule and standard FID50k evaluation. Apply the user's
baseline gate of FID50k < 50 before drawing architectural conclusions. Stop
for nonfinite training; if the ordinary baseline does not qualify within the
registered budget, resolve that protocol/budget rather than ranking immature
models. If it qualifies and ISRU-D loses to budget-matched SwiGLU in both FID
and measured cost, do not start an unregistered coefficient/order rescue.
A favorable screen warrants the Free-ISRU attribution control.

The exact epoch budget and dispatch manifest remain unset. This is a candidate
selection with a concrete implementation target, not an executed experiment.

## Current code identity

The active [ffn.py](../../ffn.py) implements **FreeISRU**, and `--ffn isru`
selects that ordinary independent-matrix function. It is not ISRU-D. Add a
distinct `isru_d` option when implementing this recommendation; do not rename
the old option or imply its old checkpoints test the selected architecture.
The migration should stay confined to the replaceable FFN and its CLI hook,
with no historical synthetic trainer imported into image training.

This decision moves exploratory image transfer ahead of the pending
boundary-flux calibration. It does not change the existing evidence or the
finalized Report2.

## Verification of this recommendation

Checked the active image hook, archived diagonal-before-projection operator,
four local evidence links and the matrix-count arithmetic above. All eleven
Report2 protected-file hashes match finalization.json. This pass changes
documentation only; no FFN implementation, checkpoint or training result was
created or modified.
