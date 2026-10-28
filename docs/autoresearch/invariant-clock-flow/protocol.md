# Fixed old-law invariant-clock screen

Register before formal outcomes. Sole candidate: two invariant-clock planar
rotation layers with the professor's middle SiLU. This tests an architectural
constraint, not new action-angle mathematics, a posterior model or a claim
that exact integration must improve quality.

## Exactly 21 fresh cells, with old controls retained

Use the unchanged Product16, dependent sign-code and half-mixture banks from
the earlier studies. All have the original component locations/variance and
registered physical rotation. No new targets, labels, mixture weights,
validation banks, oracle supervision or held-out test access.

On each law run exactly these seven arms:

1. `rotation`: width64, two pair flows, each clock MLP32→128→32 with SiLU,
   tau10/range(.2,1), learned frequencies initially zero, inner duration1.
2. `euler`: same parameters and gates, one Euler application of the same field.
3. `normalized_euler`: same parameters/gates, normalize each Euler pair by
   sqrt(1+angle²). It preserves amplitude and has the unit-endpoint atan-rate
   interpretation; it is a conserving parameterization control.
4. `silu_identity`: bias-free width64 W2SiLU(W1h), W1=W2=I initially.
5. `fixed06_identity`: same initial function in W=I+.6P coordinates, P=0.
6. `swiglu129`: ordinary native factory at nominal129 (rounded internal86),
   selected arithmetically for near-matched parameters/dense work.
7. `native_k1`: the literal original scalar-input-clock ODEFFN, order1,
   its ordinary seed0 initial matrices/clocks and native middle SiLU.

All five initial-identity arms match their initial hidden function and shared
backbone gradients. The three flow arms also share their entire clock state
and initial parameter tangents. Plain/shifted SiLU do not have the same
trainable tangent space as constrained frequencies. NativeK1/Swi129 are not
called function-matched. Clock parameters initially have zero gradient at
zero frequency; nonzero-frequency and actual-update probes must discharge
active routes before formal training. No frequency initializer sweep.

Retain all five historical endpoints per law: nativeK2, ordinary freeISRU
and Swi64/128/160. These 15 rows are reused evidence, not independent reruns.
Fresh initializations copy every shared tensor/runtime buffer from the old
Swi128 cold reference; new FFN tensors are seed0. Preserve native width64,
depth3, LayerNorm/time-FiLM, residual scale, head and all arithmetic.

## Budget and quantities

Seed0, historical training stream100000, 100,000 old training targets,
batch512, 5,000 updates, AdamW3e-4/wd0/clip1, pair-velocity loss. Save states
at0/1000/2500/5000, optimizer/RNG/runtime buffers and stream digests. Do not
select a favorable checkpoint. Keep native complete-bank oracle/pair MSE,
five512-pair Jacobian panels,2048-source Heun32 samples and128-direction SW2.
These are low-dimensional synthetic metrics, not image FID.

Analytic 16D whole-model counts to verify before training: rotation/Euler/
normalizedEuler84240parameters/81984inferenceMACs; Swi12984180/82368;
nativeK183862/82368; plain/shifted SiLU58512/57408. Dispatch measures backward
work too. Pointwise sin/cos/radius work is excluded from matrix MACs, and
actual draw/update/end-to-end timings are separately recorded. The near-
matched ordinary comparison is not an equal-wall-time assertion.

Record every learned generator P at all snapshots, with zero-spectrum CV
explicitly undefined at initialization. Record actual visited clocks, pair
amplitude changes and full gate-feedback derivative gains. Preserve native
P diagnostics too. No P-norm or singular-value certificate replaces the
complete network Jacobian/generation checks.

## Acceptance and failure interpretation

For each law define the practical reference set as Euler, new plain/shifted
SiLU, new Swi129, new nativeK1 and all five historical endpoints. The sole
candidate must reduce final oracle MSE by at least10% against the best of
this set on at least one law, remain within5% of the best on every other law,
and keep complete Jacobian error and SW2 within10% of the best on every law.
It must also stay within5% of normalizedEuler's MSE and10% of its Jacobian/
SW2 on every law. Its parameters and both inference/training dense work
must stay within5% of new Swi129. All conjunctions apply to the same fixed arm.

NormalizedEuler is reported separately as a preserving-family control, not
hidden from comparisons. A tie there could support the amplitude constraint,
not an exclusive ODE-solver benefit. A fixed-clock or weak-adaptation outcome
would also limit any adaptive-clock claim. Passing this screen only justifies
separately registered old-task/image confirmation and an unopened holdout;
it cannot establish image gains or complete the successor objective.

Failure closes this setting. No extra basis, reflection, clock width/range,
frequency initialization, norm loss, solver step, learned duration, optimizer
or training-length rescue. Keep every failed source, model and outcome.

## Readiness, audit and resources

First pass the local operator checks. Then source-register the full runner,
pin/warm-reuse the existing synthetic-cpu stack and obtain a NEW fresh
agent-follows-doc native preflight: exact old-reference predictions, actual
costs, active-head shared gradients, nonzero-frequency routes and physical
three-step resume for all21cells. Those short update witnesses are readiness,
not warm starts or formal trajectories. Controller requires the actual
documented-pass receipt/hash and zero doc mismatches.

After training independently reconstruct every84checkpoint prediction and
all105new panels,21manual-Heun batches,210double-precision manual-chain
Jacobian anchors,15reused endpoint evaluations and every stream/counter/
matrix summary. Do not claim a full optimizer-trajectory rerun. Quality
interpretation waits for actual terminal controller/audit exits.

Mac mini only unless Lenovo is reachable and independently ready. Use at
most two one-thread CPU workers, no GPU reservation, primed process/RSS
telemetry every5seconds and saved counters every100updates. Cell ceiling600s,
training cohort7200s, audit1200s; expected new disk below1GiB. Never overwrite
an existing runtime or restart a live handle after an observation timeout.
Preserve old sources, datasets, metrics, sealed studies and Report1.
