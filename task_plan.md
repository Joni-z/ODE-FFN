# ODE autoresearch task plan

## Goal
Use the autoresearch experiment loop to develop an evidence-supported ODE architecture for flow matching. A useful successor must face strong SwiGLU and compute controls, including both Product16 and dependent-sign data. Installing a loop or completing one diagnostic does not complete this goal.

## Current cycle: oct27-complete-hidden-field
- [x] Read the official autoresearch protocol, project conventions, and preceding transfer/failed-generator reviews.
- [x] Create isolated branch `autoresearch/oct27-ode` and worktree `/Users/mr.z/ODE-FFN-autoresearch`.
- [ ] Write an ODE-specific program with immutable evaluation, explicit budgets, preserved failed experiments, and publication rules.
- [ ] Register a read-only checkpoint diagnostic of the complete hidden vector-field derivative, including normalization and gate derivatives.
- [ ] Verify analytic derivatives against automatic differentiation and exact checkpoint replay.
- [ ] Execute the registered diagnostic with monitoring; assess the result without coefficient or checkpoint selection.
- [ ] Publish code, all outcomes, limitations, and the next justified question to GitHub.
- [ ] Select and execute the next justified cycle; keep the overarching goal active until useful architecture evidence exists.

## Decisions
- Adapt the official research protocol, not its GPT task, tokenizer, val_bpb, CUDA environment, or unrestricted hyperparameter search.
- Preserve historical data, seed 0, native model semantics, and all sealed evidence.
- Use the actual residual duration c=1/sqrt(3), without tuning it.
- First test frozen checkpoints. This can identify whether higher-order integration of the learned hidden field is useful, but cannot establish the behavior of a freshly trained higher-order model.
- Record failures rather than reset shared/published history. Never overwrite existing runtime outputs.

## Errors encountered
- None in this cycle. The expected absence of root planning files was established before creating these files.

## Status
Registering the task-specific autoresearch protocol and a strictly read-only derivative diagnostic. No new architecture training has started.
