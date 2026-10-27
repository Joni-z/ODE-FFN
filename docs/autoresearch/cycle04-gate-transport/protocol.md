# Cycle04: exact gate transport, fixed cold-training screen

Registered before new training outcomes. This tests the specific computational change in [design.md](design.md), not cycle03's unchanged baseline identity. Only the rectangular gate-transport matrix C is added; its fixed block topology makes the joint generator cube-zero. The professor-style degree-two exponential polynomial is therefore exact. The new term shares an existing activated gate across additional value/output directions. A fixed-gate-dictionary rank witness motivates testing this computation; it is not a full-network lower bound or a predicted win.

## Arms and data

Three new arms on each of Gaussian2D, Ring8, Product16, dependent signs and the half-mixture:

- `transport`: native SwiGLU128 (actual hidden85), plus zero-initialized C of shape42x43 in each of three blocks; exact endpoint `U[(g+P_C*g/2) elementwise v]+bo`.
- `diagonal`: the same native SwiGLU128, with zero-initialized ordinary per-channel `scale_delta`, gate multiplier `1+scale_delta/2`. This preserves the ordinary function class and tests redundant scaling/optimization, with fewer new parameters.
- `swiglu143`: native SwiGLU architecture at nominal143, actual95, as an ordinary matched-parameter/inference-cost control.

All ten existing SwiGLU128/160 endpoints remain explicit reused practical controls. They are not independent reruns. The original ODE may be discussed only as historical context; beating it alone cannot qualify the candidate.

Keep width64/depth3, LayerNorm, external-time FiLM, residual scaling, output head, original targets, independent Gaussian source, uniform FM times, pair-velocity labels and all evaluation banks. Training is seed0,5000 updates,batch512,AdamW3e-4,weight decay0,clip1 and historical data-stream identifier100000. No new dataset, loss, teacher label, duration, coefficient, initializer sweep or norm constraint is added. Transport/scaling inherit every historical Swi128 cold tensor and buffer; their new parameters consume no RNG draws. Wider Swi143 uses seed0 and copies all surrounding parameters/buffers from the same reference, but is not falsely called hidden-function/tangent paired.

## Verification before launch

Verify literal matrix exponentials, K³=0, complete input/parameter derivatives including C, zero-C native equality, a nonzero-head initial shared-parameter tangent probe, analytic C gradient, the two-dimensional fixed-basis rank witness and executed forward/backward MACs. Test paired fresh/resumed three-step optimization for every arm/dimension. Explicitly assert the measured16D counts: transport89016 parameters/87210 inference MACs; Swi14389418/87552. Record diagonal cost separately; fewer parameters are not a matched-budget ablation.

Freeze source, config, banks, training targets, reference checkpoints and previous seals before launching. Preserve cold/1000/2500/5000 snapshots, optimizer/RNG/runtime-buffer state, stream hash chains, training/clipping curves, learned C singular/eigenvalue-related diagnostics and actual conditional generator visits. Check K³=0 at real visited inputs and distinguish learned rectangular C, the square-zero gate operator P_C and the complete generator K(v). Zero eigenvalues are a structural identity, not stability evidence.

## Execution and evaluation

15 fresh trajectories, at most two one-thread CPU workers on Mac mini; Lenovo may receive whole groups only if SSH is verified before assignment. Never depend on MBP/torch. Monitor process CPU/RSS and progress every5seconds. Each cell has a600-second ceiling, with no automatic experiment restart. Publication uses local GitHub credentials when available and is reported separately from computation.

Use every row in existing field banks:10000 per2D law,20000 per16D law. Retain five512-row fixed-time field/Jacobian panels and2048-source Heun32 generation with128 slice directions and target-reference floors. No test bank is opened. Audit all60 new snapshots,75 panels,15 generated batches, consumed streams, matrix diagnostics and immutable bindings before selecting a continuation.

## Fixed decision

Only `transport` can qualify. It must reduce primary oracle field MSE by at least10% against every ordinary control (`diagonal`,new Swi143,reused Swi128/160) on at least one16D law. It must remain within5% of the ordinary frontier on every other law, keep complete Jacobian error and SW2 within10% of the best ordinary control on every nonlinear law, and stay within5% of Swi143's measured inference matrix work. All conditions must hold for the same fixed candidate. These are screening rules, not significance tests.

A survivor still requires separately frozen longer-budget continuation, an unopened holdout and isolated throughput/memory checks. A diagonal or wider-baseline gain is not an ODE result. The candidate is algebraically an ordinary partial gate mixer; an identical formula is not an independent control or evidence of an ODE-specific causal effect. If it fails, stop this setting without changing partition, coefficient, rank, feedback, widths, clipping, optimizer or training length to rescue it. Preserve every result and leave the broader research goal active.
