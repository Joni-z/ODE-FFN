# Autoresearch for the synthetic ODE-FFN

This protocol adapts the fixed-budget loop in [Karpathy autoresearch](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/program.md) to the existing Mac mini CPU study. The worktree uses the dedicated branch `autoresearch/efficient-solvers-20261122`.

The pinned dataset, minibatch order, loss, optimizer and outer sampler come from the existing synthetic trainer. Three full-width solvers change only the inner integration of the same learned ISRU vector field. All models use seed 0, width 64, batch 512 and AdamW 0.0003. The internal horizon is 1.

1. Score the historical 10,000-update two-step midpoint baseline on the fixed Product16 selection stream.
2. Give each candidate the arithmetic cost of those 10,000 updates. Profile each forward and backward pass, clipping and optimizer with the existing scalar FLOP estimator. Save the nearest whole-update checkpoint.
3. Rank by pooled SW2 on the original selection arrays. Compare Ralston (four drift evaluations), two midpoint/Adams–Bashforth steps (three), and one third-order SSP Runge–Kutta step (three). The screen keeps the best candidate for a full-budget check, alongside the original reference. Use the lower number of evaluations to break an exact tie.
4. Resume the selected Product16 candidate to the established 20k four-step midpoint training FLOP budget. Train the same solver on Ring8 from seed 0 to its corresponding budget.
5. Score both on the original selection and confirmation arrays with Heun128, eight banks of 4,096 and 256 projection directions. Assess endpoints against the existing midpoint and ordinary FFN controls.

The tracked protocol and trial source are committed before training. Keep raw logs, checkpoints and the live TSV under the work directory. Publish endpoint evidence and keep/discard decisions. The selection and confirmation stages use fixed arithmetic budgets because CPU wall time varies when the Mac mini runs multiple studies. This is one research cycle for the ODE-FFN project.
