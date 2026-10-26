# Can a linear readout recover the missing nonlinear response?

Registered before new frozen-feature extraction or fitting, 26 October 2026. The preceding response-basis result cannot show that the native network contains the missing nonlinear feature: even a linear predictor's error lies in that privileged basis. This study asks what the existing learned feature tensors actually permit through a new linear readout.

## Fixed models and inputs

Use the four final direct predictors from conservative_readout_20261026: original scalar K2 ODE, SwiGLU128, SwiGLU192 and SwiGLU384, all step5000. Restore the exact frozen source and nonpersistent buffers. The three gradient-readout arms have a different output function and are outside this linear-head question; they are not silently treated as ordinary linear heads.

Use all4096 A/B row pairs and all five times(.1,.3,.5,.7,.9) saved by coordinate_dependence_20261026. All are previously exposed diagnostic inputs, not an untouched final test. Rotate only to reconstruct physical inputs/labels and to analyze errors; the probe gets no extra oracle coordinate features. First reproduce20 saved float32 native panels. During float64 extraction, reproduce every saved A prediction and retained hybrid output, without changing parameters.

The same fresh-population input path supplies two calibration labels:

- exact population conditional-mean velocity, a privileged diagnostic label;
- ordinary sampled pair velocity, reconstructed from the already saved source/target draws.

Both labels use identical inputs. These population calibration laws differ from the original cached-bank training objective. A successful fit therefore establishes linear accessibility at these fixed features, not a defect in the original optimizer or a gain under the original training procedure.

## Feature sites and readouts

Capture three existing tensors: input embedding, hidden state after the third residual update before final LayerNorm, and final LayerNorm output. Capture actual native-forward activations; do not append coordinates, time features, nonlinearities, basis functions or extra hidden states.

At each site fit [1,h] to both calibration label matrices. The ordinary model's final readout belongs to the last site's affine class. Probe two predetermined temporal scopes:

1. shared: one readout across all five times;
2. separate: an independent readout at each time.

The second is a privileged capacity diagnostic with five sets of coefficients and access to discrete time identity; it is not the same-cost native model or a proposed architecture. Input-embedding probes provide a known affine-only negative control.

Split original row pairs0:2048 and2048:4096. A rows from one half calibrate; A and all coordinate-retaining hybrids from the opposite half evaluate. Swap halves. No hybrid or evaluation label enters fitting. Rows across time retain their fold membership.

Fit float64 SVD least squares with fixed relative cutoff1e-10, retaining every declared condition and recording ranks, singular values, coefficient norms and residual normal-equation checks. Do not tune a penalty, cutoff, feature site or temporal scope after outcomes. Structural rank deficiency from input embedding or LayerNorm is expected and must not be counted as a learned failure. This is a finite-bank accessibility probe, not a population approximation lower bound.

## Measurements and the primary convention

For every model/site/scope/calibration-label combination record A/H predictions and paired total/conditional/other-coordinate risks, both folds separately and the pooled mean. Retain native predictions as the reference. All4*3*2*2=48 probe settings are reported, not only the best head.

The primary question uses **shared, final-normalized features, mean labels**. For each direct generic reference and each evaluation fold let

```text
Delta_native = R(ODE native)-R(reference native)
Delta_probe  = R(ODE probe)-R(reference same probe)
gap_closure  = 1-Delta_probe/Delta_native.
```

Require positive native denominators. A majority linear-readout-accessibility account survives only if closure>=.50 for all three references on both held-out folds. Otherwise it is not established by this probe. Negative and above-one values remain visible. Also report absolute ODE/generic risks: closing half a gap need not match the practical baseline.

The separate-time and pre-normalization results diagnose two explicit rival explanations if the primary fails. They cannot replace its outcome or nominate an architecture. Success only with separate-time heads is consistent with temporal readout incompatibility; success only before final normalization is consistent with feature accessibility lost at that normalization. Neither establishes causality of a training failure or superiority of a changed architecture.

The two folds share fixed trained models and have reciprocal calibration sets. Show their individual values/ranges as descriptive robustness checks, not independent model replications or confidence intervals. No significance test is selected for this finite diagnostic, and no winner claim beyond its explicit scope is made.

## Verification and resource plan

Before outcomes test known available/missing nonlinear features, rank-deficient designs, held-out isolation, temporal-shared versus separate fitting, and orthogonal output-coordinate equivalence. Freeze this protocol and analysis source. Actual hooks must reproduce native outputs without altering the forward. Verify the original source and model/input hashes again at completion.

An independent evaluator manually traverses the immutable backbone, rather than using extraction hooks. Pivoted QR with the declared SVD rank must reproduce fitted predictions, training normal-equation residuals and every held-out A/H risk. Rank-boundary ambiguities or unstable coefficients are reported explicitly rather than repaired through cutoff selection. QR coefficient vectors need not equal SVD minimum-norm vectors in rank-deficient classes; compare their realized functions.

Use one Torch/BLAS CPU thread with bounded per-model memory on Mac mini. Record process/time/memory at each model/time. Preserve source, coefficient/singular-value arrays, all held-out predictions, tables and verification receipts. No GPU allocation or model optimizer steps. Lenovo is offline and is not counted as running work. Either outcome leaves the broader useful-ODELayer objective open; a later architectural experiment needs its own mechanism and cold protocol.
