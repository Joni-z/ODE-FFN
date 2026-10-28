# ISRU follow-up: preserve and understand the positive result

Updated October 29, 2026. **Current research focus: revisit the existing ISRU
synthetic result before inventing another architecture.** This update records
the evidence and next work; it launches no training or new model.

**Reporting scope:** this follow-up belongs to [Report2](../../../reports/report2/README.md).
Report1 is finalized and unchanged.

## The implemented model

The winning arm is `free_isru` in
[cubic_core.py](../../research-cubic-split-2026-10-26/cubic_core.py):

```text
u = W1 h
response(u) = u / sqrt(1 + 1.2 u^2)
output = W2 response(u)
```

W1 and W2 are independent, bias-free learned 64x64 matrices. Both start at
`I + 0.3 P + 0.045 P^2`, using the original first P's cold state. The response
comes from the exact local cubic subflow `y'=-y^3` at duration 0.6. The separate
`cubic_split` candidate retains a tied P and adaptive clock; its result must
not be substituted for the winning free-matrix ISRU arm.

This identifies the actual successful computation. It does not establish a
new activation, an ODE-specific necessity or a converged-generator advantage.

## Positive result and completed follow-ups

All rows below are recorded 5,000-update oracle field MSEs; lower is better.
Compare models within each row. Historical references are reused, not fresh
replications.

| Target law | Free ISRU | SwiGLU160 | Interpretation |
|---|---:|---:|---|
| Independent-sign Product16 | **0.037159** | 0.078178 | ISRU field error is 52.47% lower in this screen. |
| Dependent signs | 0.085073 | **0.039790** | The advantage reverses after cold retraining. |
| Half Product16 / half dependent signs | 0.163557 | **0.113254** | The reversal persists with all 65,536 original component locations retaining positive probability. |

Sources: [initial experiment](../../research-cubic-split-2026-10-26/analysis-report.md),
[dependent-sign transfer](../../research-joint-dependence-2026-10-26/analysis-report.md),
and [fixed-component-support control](../cycle02-full-support/analysis.md).

ISRU has 58,512 parameters and 57,408 dense forward MACs per 16D example,
versus 95,820 and 93,888 for SwiGLU160. The smaller SwiGLU64 is a separate
near-parameter/work control. These are matrix-work counts, not end-to-end
latency measurements. The Product16 field gain does not translate into an
equally clear sampling gain: ISRU SW2 is 0.092898 versus 0.097835 for Swi160
and 0.092758 for Swi64 in the original comparison.

The line was therefore followed up through transfer controls. What was not
resolved is why the positive case works, whether it survives adequate
baseline training, and which component of the implementation supplies it.
The failed transfers constrain the claim; they do not erase the positive case
or settle whether ISRU is worth understanding.

## What remains unresolved

- **Training adequacy.** Product16 ISRU MSE falls from 0.131870 at 2.5k to
  0.037159 at 5k; Swi160 falls from 0.159333 to 0.078178. These endpoints do
  not establish convergence. In the last stored window, clipping fractions
  are 0.980 and 0.000. That difference is an observation, not a demonstrated
  cause or authorization to tune clipping.
- **Activation versus parameterization.** ISRU uses a special matrix
  initialization and independent maps. The same-family `silu_split` control
  keeps tied polynomial maps and an adaptive clock, so it does not isolate
  the activation in the free-matrix model. These studies do not attribute
  the whole gain to saturation, initialization or untying alone.
- **Task dependence.** Product16 has separable latent component signs;
  the other laws introduce different posterior interactions. The controls
  do not isolate one unique causal property. Do not repeat the claim that
  component count alone explains the reversal.
- **Generation and ODE interpretation.** Oracle field fit, sample quality and
  a useful ODE mechanism are separate claims. The current image FID screen
  supplies no missing validation; its [baseline limitation](../baseline-validity-review.md)
  remains in force.

## Next work

Follow [task_plan.md](task_plan.md) and [notes.md](notes.md). First recover the
exact frozen ISRU/baseline states and assess learning curves, oracle error
scales and sampling quality. Then specify one focused follow-up with adequate
ordinary controls. Do not add a stack of tricks, assume that 5k is sufficient,
or move to images using the weak current image baseline. No new training
protocol or launch is selected by this note.

The original cubic screen and its transfer records remain unchanged. The
current task is an explicit reassessment of a positive lead, not a claim that
an old failed gate has passed. Raw runs, source snapshots and Report1 stay
untouched. The broader ODE-FFN objective remains active.
