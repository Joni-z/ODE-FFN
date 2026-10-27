# ODE autoresearch

The user-authorized autonomous research loop follows [program.md](program.md), adapted from Karpathy's official autoresearch. The data, evaluation and scientific controls remain specific to ODE-FFN; this does not run the upstream language-model task.

| Cycle | Question | State | Decision |
|---|---|---|---|
| [01](cycle01-complete-field/analysis.md) | Does a complete nonlinear hidden-field Taylor correction help existing SwiGLU checkpoints? | Completed and independently replayed | Reject this frozen-checkpoint upgrade; no new training |
| [02](cycle02-full-support/analysis.md) | Does the ISRU reversal persist with all65536 component locations retaining positive probability? | Five unchanged models trained and independently replayed | Yes; literal component count alone is insufficient, without a sole-cause claim |

The active worktree is `/Users/mr.z/ODE-FFN-autoresearch`, branch `autoresearch/oct27-ode`. Curated milestones are published to GitHub main. Private data, checkpoints and raw logs are stored under `/Users/mr.z/ode-workspace-prep/autoresearch/`. Use the repository task_plan.md and notes.md for current work. The broader useful-ODE-architecture objective remains open.
