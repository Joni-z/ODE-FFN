# Task plan: A fixed nonlinear local field

Goal: test one explicitly nonlinear ODE-inspired FFN against the original layer and strong ordinary controls, after the paired-matrix and bias-placement diagnostics failed to close the nonlinear gap.

- [x] Complete and seal the bias-placement study; no hidden-specific recovery.
- [x] Check known inverse-square-root activation and numerical-integration prior art.
- [x] Specify the field, split update, controls, unchanged FM budget and stop rule before new outcomes.
- [x] Implement and verify derivatives, small-step consistency, native references and exact resume.
- [x] Freeze sources and input bindings; launch one monitored 15-cell screen with six historical references.
- [x] Independently replay checkpoint outputs/streams and report all outcomes, trajectories, costs and sampling diagnostics.

Constraints: training seed 0 only; same native width64/depth3 backbone and ordinary pair-velocity FM objective. No continuation, coefficient/order/width/clock/learning-rate rescue on failure. Preserve old artifacts, submitted Report1 and dirty worktree. No commit/push or subagents. Use at most two one-thread Mac workers; Lenovo SSH timed out, so no remote launch is currently possible.

Status: complete and negative for the registered ODE candidate. All15 new cells completed in160.6 seconds, launched once through session58085/PID4714. Independent audit session21468 passed in12.7 seconds; analyzer session70604 exited0. All21 outcomes are reported. Free ISRU supplies an ordinary nonlinear-response lead, with no ODE-specific gain or follow-up rescue selected.
