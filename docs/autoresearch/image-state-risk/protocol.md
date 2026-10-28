# Shared-state image risk: fixed diagnostic

Zhizhe Zhang · October29,2026. Registered before new cross-state predictions.

## Question and evidence
Does native K1's FID advantage coexist with lower conditional-mean error on
the states encountered during sampling, despite its higher validation error?
The cold image controls establish that fixing P's clock during training can
retain the image margin, while matched-initial-function ordinary SiLU also
improves substantially. These are mandatory ordinary controls. No exclusive
adaptive-clock mechanism or successful new architecture is established.

The rejected rotation candidate does not justify a learned-frame rescue.
Earlier synthetic spectral, nonlinear/order, frame, covariance, response-basis
and loss settings retain their recorded stop decisions. This new diagnostic
asks about the actual image regression/sampling mismatch, not another operator.

## Models and fixed states
Use exactly five frozen20k models: native K1, cold fixed06, matched-init
SiLU64, Swi86 and SiLU131. The original checkpoints/runtime buffers and native
float32 model arithmetic are unchanged. Swi86 matches native K1's FFN work;
SiLU131 is the declared near-parameter ordinary control. Neither is omitted.

Use exactly four domains at Heun indices13,38,64,90,115:
1. Paired validation: the original gate-feedback64-image/64-source inputs.
2. Paired training: the first64original training-split indices and the same
   original64Gaussian starts, with identical float32 interpolation arithmetic.
3. K1 generated: stored64-image predictor-stage states from the sealed
   gate-feedback analysis, not Euler trial states or regenerated paths.
4. Swi86 generated: its corresponding stored states from the same analysis.

All five models see every identical state:100model/domain/time batches and
6,400image predictions. No favorable time, image, token or checkpoint is selected.
Train/validation paired diagnostics are not substituted for quality banks.

## Reference and measurements
Compute the exact finite-training-bank conditional mean for the unchanged
45,000normalized training images and independent standard-Gaussian corruption:

```text
logit_i = [t*z dot x_i - .5*t^2*||x_i||^2]/(1-t)^2
w = softmax(logit)
m_bank(z,t) = sum_i w_i*x_i
```

Use native float32 target normalization before promoting the bank to float64.
No covariance approximation, nearest-neighbor truncation, fitted temperature,
rescaled noise, learned prior or change in component weights. Stream matrix
products; no need to store all posterior weights. Preserve posterior mean,
effective support1/sum(w²), entropy, top weight/index and conditional trace.
The reference is exact for the empirical training law only. It is not the
unknown population conditional mean and is not an architecture-success metric.

Save every raw model prediction and per-image clean MSE to m_bank, and its
velocity-scaled counterpart divided by(1-t)^2. On both paired domains also
save MSE to the actual clean image for every model and for m_bank. Keep each
time separately, plus equal-time averages whose weighting is explicit.
Report all100rows; no p-values or independent-trial claims from these states.

## Prespecified interpretation
An empirical-risk reversal requires K1's equal-time bank-reference clean MSE to be
at least10%below Swi86 on BOTH generated-state domains, while K1 is not better
on paired-validation states. Otherwise this specific reversal hypothesis fails.
All ordinary controls and every per-time result remain context, not new cutoffs.

Before using that reference to motivate a population-quality architecture,
check its held-out behavior. If m_bank's paired-validation clean MSE exceeds
BOTH native K1 and Swi86 by10%or more at at least three of five times, classify
the empirical reference as unsuitable for that purpose. Report any empirical
reversal but do not treat proximity to m_bank as a quality advantage. The
training/validation contrast and posterior support are reference diagnostics,
not evidence that any fitted neural model memorizes or that FID is explained.

No result promotes an architecture, justifies a spectral penalty, or changes
any previous criterion. A failed reversal or unsuitable reference requires a
different missing-premise test; it is not license to tune this one. No new
neural training is selected by this protocol.

## Verification and resources
Bind raw data/split, checkpoint/buffer state, stored inputs and every parent
seal/source before measurement. Preflight replays each native20k prediction,
checks the full bank/split identity and seeded posterior/conditional-risk
identities against direct Gaussian likelihoods. A fresh documented native
invocation must pass without fixes/retries before the formal cohort starts.

Independent audit replays every model prediction bitwise and recomputes all
posterior means with direct squared-distance likelihoods and a different batch
shape, checking mean/support/entropy to fixed float64 atol3e-9/rtol3e-9. Recompute
every per-image and aggregated metric, preserve all formal hashes, and consume
actual terminal tool exits before interpreting the output. Failed runtimes
are preserved. No tolerance or domain change after outcomes.

One one-thread Mac mini CPU worker in the unchanged image-training environment;
plotting in the existing check environment. Monitor actual CPU/RSS and batch
progress every5seconds. Measurement ceiling1,200s, audit1,200s, expected new
storage below250MB. Do not restart a live handle on an observation timeout.
Preserve old data, source, runs, figures and submitted Report1.
