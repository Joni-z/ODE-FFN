# One cold comparison of the clock input source

Zhizhe Zhang, October 29, 2026. Register before implementation outcomes.

## Evidence and hypothesis
Native K1 has FID 173.44, fixed-clock cold training 174.71, matched SiLU
178.06 and matched Swi86 185.85. The fixed-core optimizer can be transported
exactly to an ordinary FFN; this does not identify the cause of the FID gaps.
Earlier frozen time-only replacement gave 180.86, better than its pooled
200.05, but missed its original gate. Its calendar remains stopped.

The new question is cold learning of a time-only clock, not a finer calendar,
calibration or frozen rescue. Hypothesis: conditioning the amount of shared
channel mixing on known noise time, while removing feature-dependent clock
feedback, can improve generation at comparable regression error. The original
AdaLN already uses time; this tests time-conditioned mixing matrices in
addition to that existing modulation. This is a hypothesis, not a theorem.

## Exactly two new arms
Both keep the professor's K1 ODEFFN: two h+sPh maps with SiLU between them,
the original six 64x64 P matrices and six 64→64→1 clock MLPs. Keep tau=10,
scale=0.8, shift=0.2, orders=1, original biases/dropout and shared JiT body.
Restore the original native step 0 state, then set only each clock MLP's last
linear weight and bias to zero. Both arms have the identical initial state
and initial constant clock represented by float32(0.6). No input normalization or feature
rescaling is added.

1. `feature_clock_zero`: original local clock inputs, h before the first
   local solve and SiLU(h1) before the second. This is the matched-init
   feature-clock control, not an accepted new method.
2. `time_clock_zero`: both clock MLPs consume the existing shared c(t),
   broadcast their scalar output over patches. In this unconditional run
   the label embedding is the same fixed label 0 for every image.

All clock parameters remain trainable. No trained weights, time table, new
encoder, loss term, activation, generator constraint or solver is introduced.
Active-backbone preflight probes may use inherited trained tensors solely to
expose gradients; formal training always rebuilds the paired cold state.

## Fixed training, costs and measurements
Use the original 45k / 5k CIFAR split, model seed 0, data stream 100000, batch 64,
weighted clean-image loss and capped delta 0.05, logit-normal times, original
AdamW with learning rate 1e-4, betas (0.9, 0.99), zero weight decay and clip 1,
constant learning rate, no EMA or AMP. Train each arm for exactly 20k updates.
Store full states at 0, 1k, 2.5k, 5k, 10k, 15k and 20k:
model/optimizer/RNG/buffer states and the six immediately preceding states.
Use the unchanged validation bank and all five denoising panels.

Record all six physical P spectra, six clock-output weight arrays, clock
mean/std at each fixed panel time, gradient/clipping windows and resources.
Do not label a feature-clock mean matrix as its full input Jacobian.
Measure actual trainable counts and forward/backward dense matrix calls.
Expected learned count is 211094 for both arms; predicted FFN forward work
is 16512 MACs/token for feature clocks and 8322 for time clocks at 64 patches.
These are predictions to verify, not whole-model speedup claims.

Generate the same 1024 Gaussian starts with native CPU Heun128 at 5k / 10k / 20k
for both arms. Retain all six banks; 20k alone is primary. Keep the same 2048
real references, uint8 conversion and MPS Inception/FID implementation.
Retain the full 35-row completed comparison and all stronger ordinary/ODE
endpoints, including matched SiLU, matched Swi86 and both native K orders.

## Quality and stopping rule
Let best_control be the lowest 20k FID among all existing controls and the new
feature_clock_zero arm. Time-clock support requires its 20k FID <=0.95 times
best_control, with weighted and clean MSE each <=1.05 times the corresponding
loss of that same reference. With the current controls this FID limit is
164.7651, and it becomes stricter if the matched feature-clock control is
better. Report every continuous difference and both arms regardless of gates.
Nonconstant learned clock profiles are descriptive evidence; a flat profile
cannot support a claim that noise-time adaptation caused an improvement.

A passing initialization-only feature control is reported as such. A passing
time-only arm is a candidate conditional FFN, not proof of a unique ODE
function class. Failure closes this learned-clock source setting. No tau,
input normalization, raw-time feature replacement, calendar, width, order,
optimizer, EMA, duration or checkpoint rescue follows.

## Execution and audit
Mac mini, at most two one-thread CPU training workers; no GPU reservation.
Metrics use the existing MPS extractor. Warm specs 475428 fb/dc 729 a 50 unchanged.
Each trajectory has 7200 s ceiling, cohort 7500 s; training audit 1200 s; sampling
and its audit each 3600 s; metrics and its audit each 1200 s. Expected new disk
below 500 MB. Monitor primed per-process CPU/RSS every 5 s and progress every 100
updates. Preserve actual tool exits; poll live handles rather than restarting.

Before launch, a fresh agent runs documented native and metric preflights
once each. Verify paired initial functions/states, nonvacuous active gradients
including c(t), independent conditional-matrix and full-derivative checks,
unchanged baseline predictions, physical resume and measured costs. After
training, audit all stream boundaries, checkpoint predictions/panels and
last-update states; independently replay all six image and feature banks and
reference features/FID. Preserve source snapshots, all raw states, old stop
rules and submitted Report 1. Publish code and curated evidence through normal
GitHub main pushes.
