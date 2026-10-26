# Stage9: test covariance geometry inside the original K1 layer

Registered before training this family. The derivation, implementation invariants and failure boundaries are in covariance-operator-mathematics.md. The six-paper assumption check is in literature-operator-assumptions.md. Previous failed SiLU deletion, shared clocks, nonlinear-drift return and clock-range variants stay stopped.

## Hypothesis and controls

The independent Gaussian-source objective has a data-space velocity Jacobian equal to a signed isotropic matrix plus posterior covariance. We test whether a similar algebraic structure is useful for the hidden field replacing P*h. We do not assume hidden and data coordinates coincide or that the complete velocity becomes conservative.

Retain F(h)=h+D(h)*f(h), the original independent vector controllers, K1, both solves and the middle SiLU. Six arms are fixed:

- `vector_k1`: unchanged original field P*h.
- `cov_tied`: f=gamma*h+U^T(softmax_group(Uh)-uniform), four groups of eight atoms.
- `cov_untied`: same field with an independent decoder V initialized exactly to U^T.
- `cov_linear`: origin-tangent replacement of softmax, a static structured P0*h.
- `swiglu_budget`: nominal hidden96, actual hidden64.
- `swiglu`: nominal hidden128, actual hidden85.

Gamma starts at zero and is unconstrained. Temperature is fixed at1. U copies the first32 rows of each corresponding initialized reference P. All gates, surrounding tensors and RNG streams match the vector parent. The tied/untied initial hidden functions match exactly, including a nonzero output-head check. The linear control matches only their value/Jacobian at zero; the parent is not function-matched to this family. Record initial hidden-output differences rather than claiming otherwise.

Four groups/eight atoms are a single fixed budget, not a fitted choice or a dataset-supplied latent basis. The code never accesses target rotations, component locations or oracle velocities during training. Tied/linearized have69576 parameters at dim2, untied81864; their80192 linear MACs match each other. Both SwiGLU sizes remain because the tied arm saves parameters but not linear MACs. Softmax costs are additional and must appear in measured runtime.

## Frozen screen

Use the existing Gaussian2D/Ring8 v1 and product16 v1 datasets. Each arm trains5000 updates, seed0,width64,depth3,gate hidden32,batch512,AdamW3e-4,wd0,clip1, unchanged fresh Gaussian source/time pairing. Use the current monitored two-CPU-worker protocol and immutable mirrored source. Keep all checkpoint/trajectory/source/data hashes. Log the actual internal field Jacobian at zero and at32 actual hidden probes, its symmetric/skew parts, U singular values, gamma and duration/increment statistics. These are not called spectra of a constant P or of the full velocity Jacobian.

The primary endpoint is final random-time validation oracle MSE. Supporting metrics remain the common512 states at each of five times, full-field Jacobian error, Heun32 validation SW2, parameters, MACs and measured wall time. No test bank is used for selecting this screen.

For tied/linearized updates additionally record the measured change in E=-Phi and the sufficient descent margin1-max(-gamma,0)*s_max/2. This frozen-weight internal energy is not the FM loss and is not shared across layers. Do not enforce the margin or clamp gamma during this screen.

## Decisions fixed before outcomes

Each nonlinear new arm may advance as a practical lead only if it improves Ring8 or product16 oracle MSE by at least10% versus both SwiGLU sizes, stays within5% of full SwiGLU on the other nonlinear task and Gaussian control, and is within10% of the better SwiGLU Jacobian and validation SW2 on both nonlinear tasks. It must also avoid nonfinite training/diagnostics. A smaller parameter count without comparable quality and measured computation is not sufficient.

To attribute a useful result specifically to tied nonlinear covariance geometry, `cov_tied` must additionally reduce the geometric mean of Ring8/product16 primary errors by at least5% relative to both `cov_untied` and `cov_linear`. These thresholds are engineering continuation rules, not statistical significance claims. A tie with the linear control does not establish a nonlinear-field mechanism. A win by the untied arm rejects tying as the claimed source of benefit; its mechanism would require a separately frozen matched untied-linear control before acceptance.

If neither nonlinear new arm passes, stop this family at this budget. Do not rescue it with higher Taylor order, added heads/atoms, changed initialization scale, a temperature sweep, caps, new losses or enlarged controllers. If both pass, prefer the lower geometric mean nonlinear-task error, retaining all references. A passing lead is frozen before exact-resume20000-update and fresh-holdout verification, with actual throughput/latency checks. Passing the screen alone cannot complete the goal.
