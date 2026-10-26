# Plain SiLU FFN reference at the original ODE budget

Registered before this reference is trained. This is a missing comparison for the existing architecture search, not a new ODE candidate.

## Question

Does the existing two-projection SiLU FFN already fit these tasks substantially better than the original vector ODE block at the same parameter and dense-matrix budget? If so, the observed ODE gap does not require SwiGLU-specific gating to appear. If plain SiLU improves over ODE but remains behind SwiGLU, distinguish that generic FFN benefit from the remaining SwiGLU advantage. A worse SiLU result alone would not prove that multiplicative gating is uniquely necessary: the compared families and native FFN initializations differ.

## One arm, three tasks

Use the unchanged `StandardFFN(width=64, hidden=128)` through `kind='ffn'`, with label `ffn_silu`. Its block is `Linear(64,128) -> SiLU -> Linear(128,64)`, including the existing biases. Each FFN has16576 parameters and16384 dense MACs, exactly matching the two original vector32 K1 solves. The full models also have matching budgets at each data dimension. Do not confuse this expanded128 FFN with the earlier collapsed/static width64 map.

Run Gaussian2D, Ring8 and rotated product16, split into the existing v1 and product16 child studies. Copy only the exact non-FFN initial tensors and runtime buffers from the same hash-bound Stage15 vector_mul step0 references used in Stage17. Keep native FFN draws, with no trained-weight transfer. Initial zero velocity follows the common zero output head; no nonzero-head function equivalence to ODE or SwiGLU is claimed.

Retain the existing data manifests, cached-bank independent Gaussian/target pair labels, seed0,5000 updates,batch512,AdamW3e-4,wd0,clip1,width64/depth3. Keep all current validation settings: primary uniform-time oracle MSE, five-time field/Jacobian checks, validation SW2, Heun32, and saved checkpoints every1000 updates. Use two one-thread CPU workers with actual10s psutil telemetry. Product16 has one job, so it uses only one worker. No GPU allocation and no width, temperature, order, loss or optimizer sweep.

## Historical controls and interpretation

Reuse the immutable Stage17 parent ODE, shared/untied ODE and all three SwiGLU sizes, keeping the established SwiGLU96/128 comparisons explicit. Source, default behavior, common initial surrounding tensors and all saved data-RNG states must match the new protocol's comparison conditions before combining tables. The new reference retains its own FFN parameters. Do not rerun18 duplicates merely to obtain concurrent wall times; report recorded wall times as contextual, not controlled latency measurements.

Report every new endpoint and curve, together with oracle-MSE, Jacobian and SW2 ratios to the parent and SwiGLU controls. Use the fixed5000 endpoint; no retrospective checkpoint selection. A single reference trajectory per task is the experimental unit; anchors, output coordinates, snapshots and layers are not independent experimental repetitions. This is a descriptive architecture control, without a new statistical or architecture-acceptance rule. The previous ODE nomination decisions remain unchanged, and a good ordinary FFN does not satisfy the goal of designing a useful ODELayer.

## Required preflight and audit

Freeze Stage17 source unchanged, adding only this protocol and the three plans. Verify every inherited file hash, reference checkpoint hash, parameter/MAC equality, unchanged native FFN/RNG after the common-backbone copy, actual monitor output, and runner config propagation before the one authorized launch. Verify the unchanged implementation's default trajectories and a bounded full-trainer/checkpoint/resume smoke. After terminal completion, audit18 snapshots and15 diagnostic arrays, replay all three endpoint fields/Jacobians, and compare every saved data-RNG state with all six Stage17 arms. Preserve the frozen source, inputs, logs, hashes and negative outcomes.
