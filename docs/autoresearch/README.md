# ODE autoresearch

The user-authorized autonomous research loop follows [program.md](program.md), adapted from Karpathy's official autoresearch. The data, evaluation and scientific controls remain specific to ODE-FFN; this does not run the upstream language-model task.

| Cycle | Question | State | Decision |
|---|---|---|---|
| [01](cycle01-complete-field/analysis.md) | Does a complete nonlinear hidden-field Taylor correction help existing SwiGLU checkpoints? | Completed and independently replayed | Reject this frozen-checkpoint upgrade; no new training |
| [02](cycle02-full-support/analysis.md) | Does the ISRU reversal persist with all65536 component locations retaining positive probability? | Five unchanged models trained and independently replayed | Yes; literal component count alone is insufficient, without a sole-cause claim |
| [03](cycle03-gated-shear/analysis.md) | Can an exact constrained ODE preserve the strong gated baseline? | Completed and independently replayed | Yes; exact SwiGLU equivalence, not a quality improvement |
| [04](cycle04-gate-transport/analysis.md) | Does exact gate transport improve quality at matched budget? | Fifteen cold trajectories completed and audited | No; stop this setting |
| [05](cycle05-feature-flow/analysis.md) | Do integrated nonlinear gate responses outperform ordinary extra features? | Ten cold trajectories completed and independently replayed | No; numerical accuracy passes but quality fails; stop this setting |

The [feature-flow review](next-feature-flow-review.md) led to the completed cycle05. Its accurate integral did not outperform ordinary controls. No additional quadrature, initialization or width rescue is selected. The next architecture needs a task-relevant functional justification beyond a valid internal ODE representation.

The active worktree is `/Users/mr.z/ODE-FFN-autoresearch`, branch `autoresearch/oct27-ode`. GitHub main is last verified at800d09f; newer cycle03–05 commits are local pending Mac mini GitHub authentication. Private data, checkpoints and raw logs are stored under `/Users/mr.z/ode-workspace-prep/autoresearch/`. Use the repository task_plan.md and notes.md for current work. The broader useful-ODE-architecture objective remains open.
