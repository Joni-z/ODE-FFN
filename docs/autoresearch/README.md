# ODE autoresearch

The user-authorized autonomous research loop follows [program.md](program.md), adapted from Karpathy's official autoresearch. The data, evaluation and scientific controls remain specific to ODE-FFN; this does not run the upstream language-model task.

| Cycle | Question | State | Decision |
|---|---|---|---|
| [01](cycle01-complete-field/analysis.md) | Does a complete nonlinear hidden-field Taylor correction help existing SwiGLU checkpoints? | Completed and independently replayed | Reject this frozen-checkpoint upgrade; no new training |
| [02](cycle02-full-support/analysis.md) | Does the ISRU reversal persist with all65536 component locations retaining positive probability? | Five unchanged models trained and independently replayed | Yes; literal component count alone is insufficient, without a sole-cause claim |
| [03](cycle03-gated-shear/analysis.md) | Can an exact constrained ODE preserve the strong gated baseline? | Completed and independently replayed | Yes; exact SwiGLU equivalence, not a quality improvement |
| [04](cycle04-gate-transport/analysis.md) | Does exact gate transport improve quality at matched budget? | Fifteen cold trajectories completed and audited | No; stop this setting |
| [05](cycle05-feature-flow/analysis.md) | Do integrated nonlinear gate responses outperform ordinary extra features? | Ten cold trajectories completed and independently replayed | No; numerical accuracy passes but quality fails; stop this setting |
| [06](cycle06-curvature-attribution/analysis.md) | What does fitted nonlinear curvature contribute at fixed weights? | Five-checkpoint read-only study and saved-array audit complete | Full-bank risk changes stay below1%; no training or nomination |
| [07](cycle07-manifold-heads/analysis.md) | Does the ODE/Swi ranking depend on prediction space and thin/thick geometry? | Eighteen unchanged-model trajectories completed and audited | Thin/clean ODE is9% better than parameter control; fixed conditions fail; no new architecture |
| [08](cycle08-readout-invariant/analysis.md) | Can hidden P change every generated direction at fixed readout? | Eighteen endpoints, six zero-P interventions and two dimension controls verified | No; exact affine sampling law in65 directions, shared by SwiGLU; no new training |
| [09](cycle09-input-attribution/analysis.md) | Does the learned input map explain the thin/clean gap? | All72checkpoints conditionally decomposed and independently replayed | Only3–5% of the gap is input loss;95–98% remains after conditioning; no new method |

The [feature-flow review](next-feature-flow-review.md) led to the completed cycle05. Its accurate integral did not outperform ordinary controls. No additional quadrature, initialization or width rescue is selected. The next architecture needs a task-relevant functional justification beyond a valid internal ODE representation.

Cycle07 adds a paired128D geometry/head benchmark and an input-rank bound. It does not replace earlier tests or establish a new ODE method. [Cycle08](cycle08-readout-invariant/analysis.md) verifies a fixed-readout sampling restriction, without new training. [Cycle09](cycle09-input-attribution/analysis.md) finds that95–98% of the thin/clean gap remains after accounting for the learned input information. The largest residuals occur in later tangent-field bins. No new architecture is selected.

The active worktree is `/Users/mr.z/ODE-FFN-autoresearch`, branch `autoresearch/oct27-ode`. GitHub main is last verified at800d09f; newer cycle03–09 commits are local pending Mac mini GitHub authentication. Private data, checkpoints and raw logs are stored under `/Users/mr.z/ode-workspace-prep/autoresearch/`. Use the repository task_plan.md and notes.md for current work. The broader useful-ODE-architecture objective remains open.
