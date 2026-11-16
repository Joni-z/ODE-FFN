# Do the crossed field-risk changes reach target sampling?

Registered after the full field diagnostic and before new crossed samples.
No new training or response function. Four retained 20k weight sets, same P and
surrounding network; each is evaluated with the OTHER discrete hidden solver.
Every local solve still uses eight matrix applications. Existing native solver
samples and strong ordinary controls are reused on the same bound banks.

## Predictions fixed from the observed field risks

| Task | Trained weights | Lower field-risk solver | Crossed sampling prediction |
|---|---|---|---|
| Ring8 | Midpoint | Midpoint | Changing to RK4 worsens target SW2 |
| Ring8 | RK4 | Midpoint | Changing to midpoint improves target SW2 |
| Product16 | Midpoint | Midpoint | Changing to RK4 worsens target SW2 |
| Product16 | RK4 | RK4 | Changing to midpoint worsens target SW2 |

This tests the specific field-based ranking, not a general theorem that lower
field MSE guarantees better generation. Retain all four cases even if predictions
fail. Results are frozen-checkpoint interventions, not retrained architectures.

## Unchanged evaluation

Use original selection and confirmation streams: eight common 4096-source banks,
Heun 128 and 256 directions, 32,768 pooled samples. Native 20k Heun32 final samples must
replay exactly BEFORE changing any solver. Copy the existing evaluate_screen.py
and change only the native model builder, explicit solver substitution after
that anchor, and recorded inference metadata. Its dataset loading, field loss,
Heun sampler, SW2 and case metrics stay unchanged. Independent NumPy recomputes
all new metrics from raw generated arrays and original targets.

Primary comparison per split: signed pooled SW2 change and mean paired-bank SW2
change must both follow the prebound table. Report every bank, exact conditional
bootstrap 95% and four-comparison-adjusted intervals per split; also 8-comparison
sensitivity over both splits. No arbitrary 5% margin, selected best checkpoint,
new seed, altered baseline or sampler tuning. Sampling supports generative value
only when actual target metrics improve; field-risk direction alone is insufficient.

Short witnesses use one original bank's first 64 sources and verify restoration,
source/counterfactual bindings and independent metric reconstruction only. Formal
evaluation retains every original source. Run on Mac CPU;145 physical 0-3 prohibited,
4-7currently occupied and unused. Keep finalized reports and image code unchanged.
