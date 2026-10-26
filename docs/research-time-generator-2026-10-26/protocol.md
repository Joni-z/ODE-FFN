# Time-dependent original generator, with static and ordinary controls

Registered before formal outcomes,2026-10-26. This is a separate architecture question after a fixed-checkpoint sensitivity diagnostic, not a change to that diagnostic's radius or stopping boundary.

## Operator and mathematical boundary

Preserve the professor's scalar K2 layer, both independent solves, SiLU between them, tau10 and duration range(.2,1). Replace only the projection action by

```text
P(t) = P0 + phi(t)*U*V
phi(t) = sqrt(12)*(t-.5)
out = h + s(h)*P(t)h + .5*s(h)^2*P(t)^2 h.
```

U is64x4 and V is4x64. Do not materialize a different dense matrix for every input; evaluate P0*h+phi(t)*U(Vh) at both polynomial terms. The identical U,V and fixed external t must be reused inside both powers. The two separate original layers retain separate generators and clocks. The clock is still computed once from each solve's initial input. t is the ordinary external generation time, not internal iteration count.

At a fixed local input, a perturbation H of P has derivative

```text
d out = s H h + .5*s^2*(P H + H P)h.
```

Both noncommuting terms matter. The temporal operator supplies structured polynomial dependence on phi through its repeated action. Ordinary gated FFNs also have nonlinear temporal responses; no approximation superiority is inferred from this identity.

Under uniform training times, phi has mean0 and second moment1. There is no learned time embedding, expert router, extra loss, nonlinear clock, spectral constraint, changed activation or optimizer change. All parent weights are trained. Low-rank down factors use their ordinary random linear initialization and up factors start atzero. This gives a live up-factor gradient, not two dead zero factors. No teacher/checkpoint gradients, target rotations, fitted matrix factors or analytic initial weights enter construction.

## Controls and cost

Cross ODE K2 and nominal SwiGLU192 with static versus time-dependent rank4 adapters: four new Product16 cells. In the static arm use phi=1 everywhere. Since P0 and the generic base projections are unrestricted, static adapters mainly add parameter coordinates without broadening the static linear-map family. This separates temporal dependence from extra optimization coordinates.

SwiGLU receives identical-style low-rank updates to w12 and w3, including its gate and value projections. Its native actual hidden width is128. Per full three-block forward, both adapted families add6144 executed dense MACs. ODE adds3072 parameters; SwiGLU adds6144 because its two rectangular projections store separate factors. Actual hooks verified these counts. Elementary operations, kernel calls and hardware latency are not represented by dense MAC equality.

Reuse the previous same-host unadapted ode_k2_native_final and swiglu192_native_final5000-step endpoints as the two parent controls, only after native checkpoint replay and source/initialization/data-stream checks. Also retain all six previous practical dense SwiGLU controls, including the cheaper SwiGLU128 deep result. They are reused controls, not new training repetitions. The generic time-adapted model must not be omitted if it improves. No fresh duplicate baseline training is needed solely to obtain concurrent wall times; the ordinary adapted models run alongside ODE.

## Frozen Phase A

Product16 only: seed0,5000updates,batch512,width64,three independent blocks, uniform t, independent Gaussian source and immutable cached target bank, direct pair velocity, final-only MSE, AdamW3e-4/defaults,wd0,global clip1. Copy the exact parent core tensors and nonpersistent buffers. Static/time members of each family have identical actual adapter initialization. Initial hidden functions, full input derivatives and common-parameter gradients match their unadapted parent even under a nonzero output head; subsequent trajectories need not match.

Run on Mac mini with two one-thread CPU workers and10-second monitoring. Frozen source and native preflight must pass before launch. Lenovo is offline; if it returns before a qualified later task starts, a complete later dataset group may move after native preflight. Do not duplicate assigned cells or mix host differences with model effects. No GPU reservation is required for this tiny CPU-efficient screen.

Save checkpoints0/1000/2500/5000, optimizer/data RNG/runtime buffers and actual batch hashes; complete curves; per-visit actual adapted clock/term summaries. spectrum_*.json explicitly contains base P0 spectra. In a separate effective-generator-trajectory.json derive full P(t) singular/eigenvalue spectra at .1/.3/.5/.7/.9 for all four checkpoints, retaining adapter norms. Never call base P0 the complete time-dependent generator or its eigenvalues the full-model Jacobian.

Evaluation matches the immutable previous banks: primary full uniform-time oracle MSE, five512-input Jacobian panels,2048-sample Heun32 validation SW2 with128 directions and target-vs-target floor. No analytic oracle is a training label. No new test data in this screen.

## Decision rule

Only the ODE time arm may advance from Phase A, and only if all hold:

- Primary MSE at least10% below the best generic endpoint across the new and verified reused controls.
- Primary MSE at least5% below both its static-adapter and unadapted ODE controls.
- Full Jacobian error and SW2 each within10% of the best generic values.
- Actual dense forward MACs within5% of the adapted SwiGLU192 time control, finite training and complete source/stream/replay checks.

If it fails, stop this fixed setting without rank, time-basis, scale, loss, optimizer, width or training-budget rescue. A static-only gain is not temporal evidence; ordinary-adapter gains are generic, not ODE-specific. A reduction in matrix-gradient conflict or adapter activity alone is not practical success.

A nominee triggers the same four new arms and verified parent/generic controls on Ring8 and Gaussian. The same ODE time setting must beat the generic frontier by10% on at least one nonlinear task, stay within5% on the others, keep nonlinear Jacobian/SW2 within10%, and improve its own static/parent comparisons. It then requires20000-update matched continuation, a frozen fresh test,8192-sample Heun32/128 evaluations and isolated throughput/memory before acceptance. No screen alone completes the research goal.
