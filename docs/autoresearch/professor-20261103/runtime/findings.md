# Measured run cost

The90 completed cycle1–5 trajectories now have their native training-loop wall
times attached to the replayed endpoints. Recorded field and sampling values
were checked against the independent replay before attaching the times. The
sum of process wall times is16,458.6s; workers overlap, so this is not GPU hours
or elapsed queue time.

Examples from the same native task/host,20k updates with two workers per GPU:

| Host/task | Optimizer/rate | Raw K1 seconds | SwiGLU64 seconds |
|---|---|---:|---:|
|145/Product16|AdamW.0003|101.1|94.7|
|145/Product16|Muon.02|174.5|178.3|
|Lenovo/Ring8|AdamW.0003|112.9|105.0|
|Lenovo/Ring8|Muon.02|201.7|182.5|

These times include the native loop's periodic evaluation, spectra and checkpoint
work; initialization/initial evaluation lie outside its timer. They are descriptive
under concurrent dispatch, not an isolated latency benchmark. Dataset and host
are paired, so cross-host rows cannot establish a hardware speed comparison.

The protocol matches updates rather than wall time. In particular Muon's local
matrix computations add measurable cost here; equal-step fitting improvements
are not automatically equal-time efficiency gains. Analytic dense-MAC counts
exclude optimizer and diagnostic work. The recorded90–94% nvidia-smi utilization
is active time, not a measurement of peak compute efficiency or MFU.

[All run times and config hashes](trajectories.csv). No training was rerun to
collect these records. Cycle5 is included; cycle6 uses a separately registered80k budget and remains live.

## Matched80k-budget extension

The table now includes all98completed training trajectories from cycles1–6,
with eight80k cases distinguished by cycle6. Cycle7 is a frozen diagnostic
and is not counted as training. Cycle8 is ongoing and is not in this table.
Times remain per-process wall measurements under two-worker sharing, including
periodic native evaluation/logging/checkpoints. Their sum is not GPU-hours,
and mixed budgets must not be used for an equal-time efficiency claim.
