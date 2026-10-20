# Post hoc Ring8 solver check

Prepared after inspecting the 66-cell frozen-order stage, before this check's outcomes.
Ring8 is selected because frozen K4/K8 substitutions change SW2 by about 3% despite
similar separately trained endpoints. This selection is exploratory.

Freeze the same six seed-0, update-5000 Ring8 checkpoints. Evaluate K1 and K2 at
their trained order and exact exponentiation; K4 and K8 at both orders and exact
exponentiation; SiLU and SwiGLU unchanged. Twelve checkpoint/evaluation modes at
32, 64, 128 and 256 Heun steps give 48 cells. Two velocity evaluations occur per
Heun step; steps and NFE must not be conflated. Retain the same 2048 source/target
points and 128 directions, float64 operator actions and float32 outer network.
Reproduce the 12 corresponding Heun32 results and endpoint arrays within 5e-4.

Report all cells, SW2, endpoint RMS versus the preceding resolution and versus
the same checkpoint at its trained order and resolution. Examine whether the
direction and approximate size of frozen-order effects survive resolution
increases. A small 128-to-256 difference is finite-resolution evidence, not a
proof of convergence or a statistical confidence bound. No new training,
architecture, seed, or selected-best checkpoint is introduced.

Run one worker on one otherwise idle/reclaimed-authorized GPU, with ten-second
telemetry and a 30-minute timeout. Hash the script before launching; import the
unchanged scientific code from the previous frozen runtime. Write only to a new
output directory. Verify checkpoint/data/source hashes before and after.
