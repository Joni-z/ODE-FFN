# What the synthetic experiments establish

1 November 2026. **Overall verdict: WARN.** The fixed-budget Product16 result
is reproducible. It supports the complete FreeISRU configuration in a small
velocity network. It does not establish that the original ODEFFN, a tied ODE
generator, or an image Transformer benefits.

This audit started no training. It preserves the old runs, active training
code, Report1, and Report2 LaTeX/PDF. Numerical checks are in
[verification.json](verification.json); the executable check is
[verify_existing.py](verify_existing.py). A fresh read-only reviewer inspected
the primary artifacts separately. Semantic review is same-family/provisional;
it is not a guarantee of experimental correctness.

## The model, and what “original” means

The synthetic scaffold is a vector residual MLP:

```text
noisy point x_t (2D or 16D), diffusion time t
  -> linear map to 64 hidden channels
  -> 3 blocks: LayerNorm + time scale/shift + FFN + residual
  -> LayerNorm + linear map back to 2D or 16D
  -> predicted velocity
```

It has no attention, image patches, or token sequence. Each residual has scale
1/sqrt(3); the final velocity readout starts at zero. Time uses Fourier
features and a small MLP. This is a controlled test environment, not a small
version of the complete JiT image model.

The professor's original **FFN** was nevertheless tested as an actual arm:
two independent K=2 ODELayer maps separated by SiLU. Each map is
`h + s(h) P h + 0.5 s(h)^2 P^2 h`, with
`s(h) = 0.2 + 0.8 sigmoid(g(h)/10)`. Removing the surrounding Transformer
does not remove this local operator. Its transfer to a Transformer remains
a separate experimental question.

The early screen used Gaussian2D, Ring8 and TwoMoons, comparing ordinary SiLU,
original ODE and spectral-cap ODE; SwiGLU was added separately. The ISRU screen
used Gaussian2D, Ring8 and Product16, followed by dependent signs and a half
mixture. The original screen and ISRU screen must not share an invented common
checkpoint-selection rule: ISRU uses the fixed 5,000-update endpoint.

Historical source: [VelocityNet and FFNs on the archive branch](https://github.com/Joni-z/ODE-FFN/blob/research-archive-20261101/synthetic/models.py),
[initial configuration](https://github.com/Joni-z/ODE-FFN/blob/research-archive-20261101/configs/synthetic/train_v1.json),
and [the actual ISRU builder](../research-cubic-split-2026-10-26/cubic_study.py).
Replay uses the preserved `conservative_readout_20261026/source-v2` source,
whose recorded bindings were checked, rather than assuming the archive is
identical to every historical runtime.

## The positive FFN is FreeISRU

```text
u = W1 h
out = W2 (u / sqrt(1 + 1.2 u^2))
```

W1 and W2 are independent, bias-free 64x64 matrices. Both initialize to
`I + 0.3 P + 0.045 P^2`; they then train freely. There is no adaptive duration,
tied generator, or Taylor recurrence in this trained FFN. The response can
be derived as the exact duration-0.6 solution of `dy/dr = -y^3`. That supplies
an ODE-inspired interpretation, not evidence that ODE integration causes the
observed gain. A trained generator P cannot be uniquely recovered from W1/W2.

The separate `cubic_split` arm retains a shared P and adaptive duration.
It is not the Product16-winning free-matrix arm. See
[the two implementations](../research-cubic-split-2026-10-26/cubic_core.py).

For 16D inputs, the actual full-model budgets are:

| FFN arm | Parameters | Dense forward MACs/example |
|---|---:|---:|
| Original ODE K2 | 83,862 | 106,944 |
| FreeISRU | 58,512 | 57,408 |
| SwiGLU64 | 58,572 | 57,024 |
| SwiGLU128 | 83,598 | 81,792 |
| SwiGLU160 | 95,820 | 93,888 |

SwiGLU64/128/160 denote nominal widths; actual gated widths are 42/85/106.
Swi64 is a near-parameter and near-matrix-work control for ISRU. Swi160 is
a larger ordinary control. Matrix counts exclude pointwise operations and
do not establish wall-clock speed. [Budget source](../research-cubic-split-2026-10-26/costs.csv).

## Training and ground truth

The ISRU comparison ran on **CPU on the Mac mini**, unlike the later MPS image
study. Each new arm used seed0, 5,000 updates, batch512, AdamW3e-4, weight
decay0 and clipping1. The same stream provides target-bank examples, Gaussian
noise and uniform times to each arm:

```text
x_t = (1-t) noise + t target
training target = target - noise
loss = mean((predicted_velocity - training_target)^2)
```

The evaluation oracle is the analytic conditional velocity of the specified
distribution; it is not a model-generated reference and is not used as the
training label. The raw oracle MSE measures vector-field fitting. Sampling
uses 32-step Heun and reports sliced Wasserstein distance (SW2) with 2,048
samples and 128 fixed unit directions. A reference-to-reference SW2 is a
finite-sample comparison scale, not a strict lower bound.

Product16 is a rotated distribution with independent bimodal latent
coordinates. The dependent-sign law preserves those one-coordinate marginals
but changes dependence. The half mixture keeps all original component
locations at positive probability. These are constructed laws with known
truth, suitable for testing specific mechanisms. They do not stand in for
natural images. Sources: [training](../research-cubic-split-2026-10-26/cubic_study.py),
[dependent-sign law](../research-joint-dependence-2026-10-26/joint_core.py),
[half-mixture law](../autoresearch/cycle02-full-support/mixture.py).

## Results and their scope

Raw oracle MSE at the fixed 5k endpoint; lower is better. Compare arms within
each target, not error magnitudes across different target laws.

| Target | Original ODE K2 | FreeISRU | SwiGLU64 | SwiGLU160 |
|---|---:|---:|---:|---:|
| Gaussian2D | .004449 | .005429 | .004192 | .004557 |
| Ring8 | .032503 | .016694 | .017904 | .013921 |
| Product16 | .325527 | **.037159** | .164398 | .078178 |
| Dependent signs | .109867 | .085073 | .051075 | **.039790** |
| Half mixture | .398957 | .163557 | .192344 | **.113254** |

Sources: [original three targets](../research-cubic-split-2026-10-26/comparison.csv),
[dependent signs](../research-joint-dependence-2026-10-26/comparison.csv),
[half mixture](../autoresearch/cycle02-full-support/results/comparison.csv).

On Product16, ISRU reduces field MSE by 77.40% versus near-matched Swi64 and
52.47% versus larger Swi160. This is meaningful fixed-budget evidence; the
small backbone alone does not invalidate it. Its sampling distances are much
closer: ISRU/Swi64/Swi160 = .092898/.092758/.097835. ISRU slightly loses to
Swi64 on that metric. On dependent signs, ISRU's field error is over twice
Swi160's despite a better SW2 (.105373 versus .113178). Field fitting and
these finite-sample generation scores do not have an identical ordering.

The remaining gaps are specific:

1. **Convergence.** Between 2.5k and 5k, ISRU field MSE falls .131870 to
   .037159; Swi160 falls .159333 to .078178. This is a budget comparison,
   not a comparison of converged limits. It is valid to report the budget
   result; it is not valid to silently convert it into an asymptotic claim.
2. **Attribution.** Surrounding tensors/buffers and data streams were matched,
   but ISRU has polynomial near-identity initialization while SwiGLU has its
   standard initialization. Equal zero initial outputs do not make hidden
   features or optimization geometry equal. The existing tied `silu_split`
   does not isolate activation in the free-matrix FFN.
3. **Optimization.** The final recorded Product16 clipping fractions are .980
   for ISRU and .000 for Swi160. Shared hyperparameters define a controlled
   screen; they do not prove each arm is adequately optimized. No causal
   explanation follows from clipping frequency alone.
4. **Selection and transfer.** These validation banks were reused during
   exploratory architecture work. Historical references are reused trajectories,
   not fresh replications. The evidence does not establish independent
   confirmation, general generation superiority, an attention interaction,
   or JiT's clean-image prediction benefit.

## Integrity checks

| Check | Verdict | What was checked |
|---|---|---|
| Ground truth provenance | PASS | Analytic distribution-derived velocities, independent of predictions. |
| Score normalization | PASS | Raw MSE and raw SW2; no own-output denominator in these performance metrics. |
| Existence and arithmetic | PASS in stated scope | 688 historical file bindings match; 20 full checkpoint field predictions replay exactly; all 15 Report2 numeric rows match source CSVs. |
| Executed metric paths | PASS in stated scope | Snapshot/endpoint calls lead to field, Jacobian and sampling outputs; three key Product16 sampling arrays replay exactly. |
| Claim scope | WARN | Fixed-budget, task-specific configuration result; attribution and transfer unresolved. |
| Evaluation type | simulation_only | Constructed data with analytic truth, not model-derived proxy truth. |

The fresh replay independently recomputes SW2 from saved sample arrays for
all 20 endpoints. It does not rerun optimization. Half-mixture values were
checked against the source CSV, not its checkpoints. The receipt separates
these scopes explicitly; numerical agreement does not certify causal claims.

One report correction is needed in a later edit: `reports/report2/report2.tex:38`
says the tied split beats the old ODE without naming the target. It improves
Ring8 and Product16, but is worse on Gaussian2D (.005900 versus .004449).
The Report2 source/PDF were not changed or rebuilt during this audit.

## Decision before any next run

Keep ISRU as a **Product16 lead**, not an established general ODE method.
Do not select another image-scale run from this evidence alone.

First specify a longer-budget comparison of the existing ISRU, Swi64 and
Swi160 configurations, keeping the same data, optimizer and seed0. Predeclare
the budget and a plateau check, retain field error and sample metrics, and
show all checkpoints rather than selecting a favorable endpoint. Keep at
least the existing dependent-sign negative control visible.

If that supports a persistent advantage, isolate one factor: use the same
free two-matrix FFN with ISRU versus SiLU, crossing polynomial initialization
with a common standard initialization. Keep the external backbone fixed and
the existing SwiGLU controls. This tests response versus initialization;
changing clocks, clipping, solver and width simultaneously would not.

A useful ODE-specific claim then requires a retained generator/flow structure
to outperform its free-matrix control under those checks. Only after that
decision should a small upstream-derived Transformer comparison be designed
with a competent ordinary baseline. These are proposed tests, not launched
experiments or predicted outcomes.
