# Measured run cost

The76 completed cycle1–4 trajectories now have their native training-loop wall
times attached to the replayed endpoints. Recorded field and sampling values
were checked against the independent replay before attaching the times. The
sum of process wall times is14,363.3s; workers overlap, so this is not GPU hours
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
collect these records. Current cycle5 costs will be attached after completion.
