# Measured run cost

The table covers122 completed training trajectories from cycles1–6,8and10:114 at20k
updates and eight at80k. Native wall times are attached only after recorded
field/sampling scores match independent replay. Their summed process wall time
is26,528.4s; workers overlap, so this is not GPU hours or elapsed queue time.
Cycle7 is frozen analysis and is excluded from training counts.

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
collect these records. Cycle5 is included; cycle6 completed its separately registered80k budget.
Cycle9 is a frozen diagnostic and is also excluded. Cycle10 completed and is included after native replay.

## Budget and constraint extensions

Cycle6 adds eight80k cases, clearly distinguished from20k comparisons in the
CSV. Cycle8 adds twelve20k native/shape/contraction cases; its SVD cost is part
of the native timer. Frozen cycle7 cases are not training trajectories. Mixed
budgets cannot establish equal-time efficiency, and two-worker dispatch is not
an isolated latency measurement.

Cycle10 adds twelve20k raw/radial/static/ordinary cases. Scalar reduction and
square-root work is included in their native wall timer. No equal-time or
isolated-hardware speed claim follows from these concurrent observations.
