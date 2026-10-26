# Stage11: test clock-coordinate conditioning with the original operator

Registered before these training outcomes. The mathematical argument and its limits are in clock-coordinate-mathematics.md. Stage10 found a noiseless fitting gap; it did not prove an expressivity lower bound. This screen returns to ordinary cached-bank pair-target FM and changes no loss or training-data law.

## Five fixed arms

- `vector_k1`: unchanged professor-derived vectorK1, tau10.
- `vector_reparam`: same model, W_last and bias divided by8 after all reference draws, tau divided by8 to1.25. Every initial function matches, checked beyond the zero output head.
- `vector_tau`: tau1.25, original unscaled initialization. Same feasible function class and budget; initial functions differ.
- `swiglu_budget`: nominal hidden96/actual64.
- `swiglu`: nominal hidden128/actual85.

All three ODE arms retain independent controllers, middle SiLU, K1 and the original positive sigmoid range. No failed covariance field, clock sharing, activation deletion, affine duration or nonlinear-drift rearrangement is combined. Their parameter counts and linear MACs are identical. The sole deterministic scale8 is fixed before outcomes, motivated by a power-of-two function-preserving transformation toward unit temperature, not a sweep.

The corrected-lr/epsilon/clip-metric Adam equivalence is verified only as a mathematical unit control. Formal training uses the same ordinary optimizer and clipping for all arms. Do not add those coordinate corrections to the candidate or claim they are performance tricks.

## Training and measurements

All five arms on Gaussian2D, Ring8 and product16, seed0,5000updates,width64,depth3,gate hidden32/reference64,batch512,AdamW3e-4,wd0,clip1. Use the original cached training bank, independent fresh Gaussian source and uniform time. No analytic teacher is used for training. Evaluation remains the common validation panels and Heun32 sampling; no fresh test bank is opened during this screen. Two monitored CPU workers, immutable mirrored source and existing data/source/config/checkpoint hashes remain required.

Primary endpoint: final random-time validation population-oracle MSE. Keep complete curves, common-field/Jacobian error, sampling SW2, P trajectories, durations, raw-logit sensitivity and invariant controller output-weight/bias norms divided by tau. Initial tensors match the reference except the exactly scaled last-affine weights/bias of the paired arm. Its nonzero-output initial forward must be bit-identical on native preflight inputs; otherwise record the numerical difference and do not claim exact matching. Snapshot data-generator states must agree across reference/candidate/control; the code uses the same model-independent bank sampler.

## Fixed decisions

A new coordinate arm may advance only if it improves Ring8 or product16 primary error by at least10% relative to **both** SwiGLU sizes, stays within5% of full SwiGLU on the other nonlinear task and Gaussian control, and is within10% of the better SwiGLU common-Jacobian error and sampling SW2 on both nonlinear tasks. Report both new arms and the unchanged parent even if neither passes. If both pass, choose the lower geometric mean of Ring8/product16 primary errors and freeze that choice before further evaluation.

A gain by the paired transform identifies a parameterization/optimization effect at fixed initial function and fixed function class. A gain shared only by the unpaired tau arm leaves initial function changes as an alternative. No result is described as new expressive power or an ODE stability guarantee. If neither passes, stop this scale; no temperature sweep, scheduler change, cap, larger controller or K2 rescue is added. A passing lead still needs exact-resume20000-update references, new independent evaluation banks and actual throughput/latency before acceptance. Historical Report1 remains untouched and the full architecture objective remains in force.
