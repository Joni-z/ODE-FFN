# Audit time allowance, before finer-grid outcomes

The registered native-32-step replay takes 32.96 seconds for the first arm,
including fixtures/startup. Serial independent replay of all eight finer-grid
banks entails 24 baseline-equivalent model/grid units. A 600-second allowance
for the whole sampling audit may be too short even when every output is valid.

Before any N=64 or N=128 fitted output exists, raise **only the sampling-audit
phase** wall-time ceiling to 1,200 seconds. Sampler arms remain capped at
1,200 seconds each; feature extraction and feature audit retain 600 seconds.
The original prepared source/protocol and ongoing preflight are preserved.
Bind this addendum and the one-shot controller before launching finer grids.

This changes no model, step count, data, metric, sampler arithmetic, acceptance
gate, tolerance or restart policy. An observation timeout still does not imply
a failed process. Do not change any ceiling after observing finer-grid values.
