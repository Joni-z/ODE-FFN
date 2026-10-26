# Figure catalog

All four figures have PNG and PDF versions and were rendered from the recorded study artifacts by scripts/plot_architecture_followup.py. They have no inferential error bars.

- heldout-comparison: compare final field fit, derivative fit and sampling for both tasks. Notice that the Ring8 velocity advantage does not produce a sampling advantage and does not transfer to16D. Keep metric scales separate; empirical target discrepancy is not a theoretical floor.
- training-curves: all41 recorded validation checkpoints, including the exact Ring continuation. Notice convergence of Ring curves and sustained16D SwiGLU advantage. The lines are trajectories, not independent repetitions; do not cherry-pick their minima.
- mode-samples:4096 target/generated points, common generation inputs, Heun32. The bottom row uses the declared population rotation for analysis only. Notice inter-mode leakage without claiming missing clusters; projections omit14 dimensions, which are separately quantified.
- operator-trajectories: all six P spectral norms of vectorK1 at every recorded evaluation. Notice gradual, unequal growth. These are generator-matrix diagnostics, not norms of the complete field Jacobian or stability certificates.

Every figure supports the same decision: the frozen lead is insufficient for the goal, and the next test must isolate a structural explanation rather than expand a parameter sweep.

- composition-ablation: final5000-step validation oracle errors for the two-factor activation screen and SwiGLU. Hatched bars omit only the middle SiLU. Notice that Ring8 worsens for both gate granularities. This figure documents the registered branch stop; it does not test larger budgets or other activation functions.
