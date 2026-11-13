# What to test after the CelebA128 comparison

13 November 2026. Audit and proposed decision, not a training launch.

**Keep K8 as the selected reference, but do not extend it blindly. The next
useful training comparison is ISRU-D K1 versus a narrow ordinary SwiGLU.**
First inspect the saved K8 checkpoint to measure what its additional Taylor
terms actually do. This sequence separates order from parameter budget and
tests the operator that produced the earlier synthetic result.

## What is established

The completed native JiT comparison uses the same CelebA128 training exposure,
AdamW recipe, sampler, EMA and FID reference. At 100 epochs, native SwiGLU has
FID50k **35.91785**, versus **43.90767** for K8. K8 takes 199.06 seconds per
steady training epoch, versus 113.73 seconds. This is a worse quality/time
tradeoff at the measured budget, with a smaller parameter count.

[Results and primary records](results.md) bind the measurements to the frozen
source and final checkpoints. The reference is the official training split,
not a held-out generalization test. The two full intermediate checkpoint
histories were not retained: there is no recoverable FID curve for epochs
20/40/60/80. Existing P snapshots cannot reconstruct those models.

The last two ten-epoch mean training losses change from .030379 to .030058
for SwiGLU and .032073 to .031695 for K8: improvements of 1.06% and 1.18%.
Both still learn; this does not show that either FID has converged. It also
does not establish that another 100 epochs would remove K8's FID gap.
[Recomputed windows, costs and input hashes](next-step-measurements.json).

A fresh [integrity review](EXPERIMENT_AUDIT.md) independently checked remote
data checksums/splits, final checkpoint hashes, sample counts, original logs,
TensorBoard and the executed metric. No invalid final measurement was found.
The [review receipt](EXPERIMENT_AUDIT.json) records both the integrity review
and a separate evidence-to-decision review. Neither review recomputed all FID
features. These checks do not depend on the proposed follow-up performing well.

## What was transferred from synthetic experiments

The positive Ring8 result was **ISRU-D K1 with AdamW**, not K8 with Muon.
Its field MSE was .009773 versus .013717 for same-rate AdamW Swi64. Its
common-bank SW2 was .171104 versus .194535. Selected Swi64/Muon reached
.171453, nearly the same sampling score. On Product16, ISRU-D's .089003
SW2 was worse than same-rate Swi64's .085821 and Free ISRU's .085964.
[Complete comparison](../autoresearch/professor-20261103/cycle14-results-complete/analysis-report.md).

That model was a width-64, three-block, time-conditioned velocity MLP. The
image model is a width-768, twelve-block transformer predicting clean images,
with upstream RMSNorm, attention, adaLN gates and a different data/time scale.
The image comparison changes polynomial order as well as domain and backbone.
It is an experiment on a new K8 candidate, not a successful synthetic K8 model
failing to reproduce. Earlier scalar-clock K4/K8 results concern a different
operator; [the order note](../research-assessments/2026-11-11-taylor-order.md)
keeps these experiments separate.

The image pair already shares AdamW. Switching only K8 to Muon now would mix
architecture and optimizer effects. The registered synthetic Muon .02 setting
also worsened fitting for ISRU-D and Free ISRU on both tasks; it provides no
positive reason for that immediate switch. This is a result for that setting,
not a general ranking of optimizers. [Optimizer results](../autoresearch/professor-20261103/cycle13-results/analysis-report.md).
The existing synthetic transport study
also found improved fitting alongside changed mode allocation in ordinary
models: this is a reason to retain FID, not a demonstrated mechanism for the
current image gap. [Transport evidence](../autoresearch/professor-20261103/cycle15-results/analysis-report.md).

## Why fewer parameters cost more here

For width d=768, each K8 FFN contains two learned d-by-d matrices and applies
each eight times. The native SwiGLU has three projections with intermediate
width 2048. The counts below include only dense FFN matrix work, not attention,
pointwise operations, backward passes or whole-model training FLOPs.

| FFN | Whole-model parameters | Dense forward MACs/token/FFN |
|---|---:|---:|
| Native SwiGLU, intermediate 2048 |130,405,632 |4,718,592 |
| ISRU-D K8, completed |87,879,936 |9,437,184 |
| ISRU-D K1, proposed control |87,879,936 |1,179,648 |
| SwiGLU, intermediate 512, proposed control |87,901,440 |1,179,648 |

K8 has a quarter of the native FFN matrix parameters but twice its dense
forward MACs. At whole-model level the parameter saving is 32.61%, while
measured epochs are 1.75 times longer. GPU utilization was 96–98%; low
utilization does not explain this observed cost.

The narrow SwiGLU retains native biases: its extra 21,504 total parameters
amount to 0.0245% relative to ISRU-D. Preserve the original SwiGLU equation
rather than remove biases to make the count cosmetically exact.

There is also an initialization difference: native linear weights receive
Xavier initialization; replacement P matrices use Linear's Kaiming-uniform
default. Native FFNs have biases and multiplicative gates, while the candidate
has two internal residual maps around SiLU. The existing result compares
these complete FFN choices. It does not isolate any one of these differences.

## What the mathematics says to measure

One internal layer uses

```text
d(h) = (1 + 1.2 h^2)^(-1/2)
A(h) = P diag(d(h))
T_K(h) = sum_{k=0..K} A(h)^k h / k!
```

The diagonal is fixed during this series. Increasing K improves approximation
to the conditional linear exponential; it need not improve the learned image
distribution. K1 is h + P ISRU(h). K8 is not eight applications of ISRU to an
evolving state, nor eight integration steps of the global generation flow.

At K1, the exact input Jacobian is `I + P diag(d(h)^3)`. For higher orders,
the derivative includes all positions where the input-dependent A appears
inside each power. Consequently raw P alone does not describe sensitivity.
Neither a larger spectral norm nor a lower stable rank establishes instability.

For any entrance h, `||A(h)||2 <= ||P||2`. The exponential-action remainder
is at most `||h||2 * sum_{j=9..infinity} ||P||2^j / j!`. The largest final raw
P norm, 2.74828, gives an upper bound of .033604 times the entrance norm in
exact arithmetic. This is a conservative bound, not measured error relative
to the output. It uses raw training weights; FID uses EMA weights. It does
not justify either declaring K8 inaccurate or discarding its higher terms.

**Checkpoint diagnostic, before training:** use the saved K8 EMA and a fixed
set of 128 official validation images, common noise seed 0 and times
0, .1, .25, .5, .75, .9, .95. Record the sampling null label. For eight
preselected images retain entrance activations in all 24 internal layers,
then measure term norms, cancellation, diagonal distributions and relative
K1/K2/K4/K8 action differences. Compare K8 against K16/K24 in FP64 on a fixed
token subset; verify the reference difference before interpreting the gap.
Do not construct a dense matrix for every token.

Also compare final predicted-image/velocity changes and the outer gated FFN
increments on the same states. Repeat the selected cases in FP32 and training
precision to distinguish truncation from arithmetic effects. Bound this to one
short diagnostic allocation, at most 30 minutes after environment readiness;
record uncompleted measurements instead of silently extending it. No FID sweep
or training is needed for this probe. Truncating a trained K8 is a sensitivity
intervention, not evidence of how a freshly trained lower order performs.

## Smallest useful training decision

Subject to approval, run only these two new controls first:

1. **ISRU-D K1:** identical two P matrices, coefficient 1.2, SiLU, initialization
   and image recipe; change only the Taylor order from 8 to 1. Verify identical
   initial P and shared backbone tensors against K8 initialization.
2. **SwiGLU with intermediate width 512:** the author's unchanged activation
   and three projections, initialized with the native Xavier/zero-bias policy.
   Replace only FFNs after constructing the normal backbone, inside an isolated
   RNG context, so the narrower constructor cannot change attention or other
   shared initial weights. The effective intermediate width is 512; the upstream
   constructor applies a two-thirds conversion to its nominal argument.

Keep seed 0, 100 epochs/63,500 updates, batch 256, AdamW 5e-5, warmup 5,
the existing time/noise law, crop, EMA .9999 and Heun50 unchanged. Retain the
completed K8 and full SwiGLU endpoints as fixed references. New full
checkpoints and FID50k are required at 20/40/60/80/100. Earlier FID curves
will compare only the two new arms; old intermediate points must stay blank.
The registered primary comparison is epoch 100, not the best checkpoint
selected after viewing the curve.
Stop this budget at 100 epochs; stop earlier for nonfinite values or failed
source/data/checkpoint integrity. Do not change the recipe in response to an
unfavorable intermediate FID.

Do not extrapolate training speed directly from FFN MACs. Before launch,
measure both complete new models on the allocated GPU, including training,
sampling throughput, data loading and memory. Budget one continuous allocation
per arm as measured training time plus five FID evaluations, startup and a
stated margin. Existing native/K8 timings are references, not measurements
of either new arm. Retain utilization and job monitoring.

| Observation at the registered endpoint | Decision |
|---|---|
| Narrow SwiGLU beats K8 with lower cost | K8 has no advantage over the near-equal-parameter ordinary control; parameter reduction cannot explain the entire deficit. |
| K1 improves over K8 | The extra Taylor terms do not justify K8's measured cost at this budget; use the narrow ordinary control to assess whether K1 itself is useful. |
| K1 beats narrow SwiGLU | A useful small-FFN candidate exists. This supports the complete ISRU-D parameterization, not an ODE-exclusive explanation; test a matched response/initialization control next. |
| K1 and narrow SwiGLU both trail full native | Their reduced budgets are a plausible limitation; do not claim capacity is the only cause. Do not launch an optimizer rescue automatically. |
| K1 loses to K8 | Higher order is useful relative to K1 at this recipe, but K8 still loses to full native. Use the measured term profile to decide whether one intermediate order can retain the benefit at lower cost. |
| K1 offers no quality/time benefit over narrow SwiGLU | Pause new image architecture variants. Revisit the proposed mechanism before spending another full run. |

Use the size of the FID difference together with measured training/sampling
cost and checkpoint trajectories; do not call a numerical near-tie an
architectural win. K2 alone would leave both the synthetic K1 transfer and
ordinary capacity control unanswered. No K1/K2/K4/K8 grid is proposed.
K8 remains the selected reference until the user makes a new method decision.

## Launch prerequisites and preserved artifacts

The active periodic-evaluation code is present, but the existing Slurm wrapper
still names the completed 20261112 outputs and checks their old readiness/source
identities. **It must not be submitted as a new run unchanged.** Register new
output names, a clean frozen source and readiness records for the two controls;
do not bypass the guards or overwrite completed checkpoints. The new narrow
SwiGLU hook is proposed here, not implemented by this audit.

No extension to 200 epochs, Muon switch, spectral constraint, new dataset or
formal training job is selected in this audit. The finalized reports, frozen
runtime, raw checkpoints and historical measurements are preserved.
