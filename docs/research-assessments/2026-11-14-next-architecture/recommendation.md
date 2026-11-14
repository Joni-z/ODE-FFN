# Next architecture decision after the matched CelebA runs

14 November 2026. Recommendation, not a training launch. Finish the authorized
160-epoch comparison. Do not extend the present architectures automatically.

**The next question should be whether repeated computation needs a changing
feature response, rather than a more accurate frozen matrix exponential.**
First inspect the saved image checkpoints. If they support this question,
test one frozen-versus-refreshed ISRU comparison at identical parameter and
matrix-work budgets. Do not combine it with Muon, new normalization or a new
loss.

## What the image results establish

The [matched FID50k comparison](../../celeba128-wide-20261114/continuation-160/matched-results.md)
has the following completed common endpoints:

| Epoch | Native SwiGLU | Wide K1 | Wide K8 |
|---|---:|---:|---:|
| 100 | 35.91785 | 37.92486 | 38.13574 |
| 120 | 29.47602 | 32.50213 | 32.85067 |
| 140 | 25.98559 | 29.62906 | 30.08220 |

The models have essentially equal total parameters: 130,405,632 for native,
130,402,560 for either ODE. K1 also nearly matches native FFN matrix work;
K8 uses 5.27 times native FFN forward dense MACs. This is an FFN work ratio,
not whole-network latency. The baseline is already below the user's FID50
qualification threshold.

These results weaken two proposed explanations: insufficient parameter count
and insufficient Taylor accuracy. They do not prove that any ODE architecture
must lose, or that training has converged. All three still improve with
training. At the latest live read for this note, baseline had completed 154
epochs, K1 had completed 160 with FID27.78868, and K8 was generating its final
160-epoch evaluation. No complete 160-epoch ranking is asserted here.

## What is structurally different from SwiGLU

Native SwiGLU learns separate gate and value projections, with intermediate
width 2048. The current ODE uses width1200 and two square P matrices around
SiLU, plus entrance/exit projections. Equal parameter count does not make
these function classes or optimization problems equivalent.

For each current ODELayer:

```text
d0 = 1 / sqrt(1 + 1.2*h0^2)
A0 = P diag(d0)
output = T_K(A0) h0
```

The diagonal is held fixed in internal time, but remains differentiable in
backpropagation. The map is nonlinear in its input. Its internal auxiliary
system is conditionally linear: y' = A0 y. K8 repeatedly mixes features under
the same A0; it does not reevaluate the ISRU response along the internal path.
That is a correct exponential construction, not an implementation error.

Two plausible restrictions deserve measurement: high-order terms may add
little useful correction, and the fixed ISRU scale may suppress informative
amplitude differences in the learned branch. Neither is an established cause
of the FID gap. Each map retains its identity term, so saturation of ISRU
does not mean that the entire block loses amplitude information or gradients.

## A precise alternative, if the checkpoint diagnosis supports it

Keep P, width1200, entrance/exit projections, the two-map composition and
intervening SiLU. Change only the local evolution to

```text
q(y) = y / sqrt(1 + 1.2*y^2)
dy/ds = P q(y),     s in [0, 1].
```

Use one explicit midpoint step for the first controlled test:

```text
u = P q(h)
mid = h + 0.5*u
output = h + P q(mid)
```

The exact control is frozen-D K2:

```text
d0 = 1 / sqrt(1 + 1.2*h^2)
u = P (d0*h)
mid = h + 0.5*u
output = h + P (d0*mid)
```

The control equals h + A0 h + A0^2 h/2. Both variants have the same trainable
tensors and exactly two P applications per solve. The only functional change
is whether the second response uses d(h) or d(mid). They generally do not
have identical initial hidden functions, even when initial weights match.
This comparison tests the training outcome of the operator change, not an
isolated optimizer effect at an identical initial function.

Both FFNs use 7,603,200 forward dense MACs per token: about 1.61 times native,
versus 24,883,200 for K8. Pointwise arithmetic and actual GPU latency must be
measured separately. The proposed operator is a second-order nonlinear-ODE
solver, **not an eighth-order Taylor approximation**. Existing K8 remains the
tested fixed-exponential reference.

The distinction is visible in the second coefficient. With D=diag(d(h)):

```text
frozen conditional flow: h + s Pq(h) + (s^2/2) P D   Pq(h) + O(s^3)
state-dependent flow:   h + s Pq(h) + (s^2/2) P D^3 Pq(h) + O(s^3)
```

This follows from q'(h)=d(h)^3. At |h_i|=2, d_i is about0.415 but q'_i is
about0.0716. Recomputing ISRU changes the directional response; it is not just
a better approximation of the previous exponential. Positive D alone does
not guarantee contractivity for arbitrary P. The new field is globally
Lipschitz, with norm bound ||J_f|| <= ||P||, and bounded coordinate response;
these properties alone do not predict lower FID.

## Earlier tests and literature constrain this recommendation

- The [original vector-clock midpoint experiment](../../research-architecture-search/analysis-state-clock/analysis-report.md)
  already tested feedback for f(y)=D_sigmoid(y) P y. It failed: Product16
  MSE0.32117 versus frozen0.30185 and near-compute SwiGLU0.05867. Feedback
  was active, so inactivity cannot explain that failure. The proposed
  P ISRU(y) field differs in response and matrix order, but this is a reason
  for a small conditional test, not evidence that it will win.
- The [cubic split experiment](../../research-cubic-split-2026-10-26/analysis-report.md)
  used y'=Py-y^3; its ordinary free-ISRU control was stronger. It is a
  different ODE and remains negative evidence against claiming that an
  ODE interpretation automatically supplies a useful inductive bias.
- [JiT, section4.4](https://arxiv.org/html/2511.13720v2#S4.SS4)
  explicitly retains general-purpose Transformer improvements, including
  SwiGLU. Its clean-image prediction argument does not imply that hidden
  FFNs must integrate the outer flow-matching dynamics. Preserve the native
  predictor, loss and sampler in the next comparison.
- [DiffEqFormer](https://arxiv.org/html/2503.01329v1), ICLR2025, explicitly
  defines continuous hidden dynamics with depth-dependent weights. It is
  relevant prior art for ODE Transformers, not evidence for this local image
  operator. A state-dependent hidden ODE is not itself a new invention.
- [Stollenwerk, EACL2026](https://aclanthology.org/2026.eacl-short.48/)
  derives Dynamic ISRU from a decoupled RMSNorm construction and analyzes
  outlier compression. This supports checking feature-response scale, not
  claiming that ISRU or learnable ISRU scaling is novel. The paper does not
  test our generator.

## Execution order and decision

1. **Close the current comparison.** Collect baseline/K8 FID160. Keep the
   native baseline architecture, checkpoints and evaluation fixed.
2. **Inspect existing EMA checkpoints at100/140/160.** Use the same saved
   noisy inputs at t=.1,.3,.5,.7,.9. Measure Taylor-term magnitudes and their
   effect on complete predicted-image/velocity error; d and q' distributions;
   actual feature-weighted P action; and the finite frozen-versus-refreshed
   midpoint correction. Preserve all input gradients when measuring the full
   Jacobian. Fixed-checkpoint interventions are diagnostics, not retraining
   results or proofs of attainable FID improvement.
3. **Choose from those measurements.** If repeated terms and the ISRU
   response change are appreciable in the time/depth regions with excess
   error, the two-arm comparison above is a concrete next test. If the
   correction is negligible, do not repeat a feedback experiment on wording
   alone. If saturation dominates, refreshing the same saturation is not an
   automatic remedy; first examine its ordinary feature-response control.
4. **If selected, train only the two arms.** Same seed0, width1200, AdamW,
   JiT/CelebA recipe and FID50k every20epochs; first endpoint100epochs, using
   the existing native/K1/K8 endpoints. Record training speed and sample
   speed on the same GPU. No optimizer, loss, spectral constraint, time
   conditioning or baseline change is bundled into this test. Measure runtime
   before choosing the Slurm walltime; no allocation is authorized by this
   recommendation itself.
5. **Use an outcome that changes the decision.** Beating only frozen K2 is
   mechanism evidence, not a useful replacement FFN. Beating native at100
   selects a matched continuation to160 to check whether the gain persists;
   a tiny endpoint difference is a lead, not a finished result. An eventual
   efficiency claim must also beat or justify the added1.61x FFN work. If
   both new arms lose to native, stop this architecture family rather than
   add more order, width or optimizer changes to it.

Muon is a separate later comparison, applied to both ordinary and ODE arms
under a registered optimizer budget. Raw P norm/flattening penalties are not
the next default: prior constraints damaged fitting, and current image
evidence has not identified runaway amplification as the cause of the gap.

The publishable question is whether a particular evolving feature response
improves the measured quality/cost tradeoff. Better numerical integration and
the phrase "ODE-inspired" do not answer it. A distinct connection between
internal dynamics and flow-matching time would need its own evidence; this
candidate does not claim that connection.
