# Matched-law supervision diagnostic

Purpose: determine whether conditional target noise closes the original vectorK1 fitting gap to both SwiGLU budgets. Pair/mean arms use identical fresh inputs, initial weights and optimizers. The exact teacher and pair teacher share the same population risk minimizer before clipping; the earlier empirical-bank teacher is not substituted into that identity.

The registered second interpretation applies: ODE remains at least25% worse than the better mean-supervised reference on both nonlinear tasks. Removing conditional noise improves absolute errors but does not establish a useful ODE lead. The full numbers, derivative checks, final-interval clipping and limits are in ../population-noise-readout.md.

Claim candidate: conditional noise alone does not close this fixed-budget architecture gap.

- Evidence: all12 final results, identical regenerated input streams, matched initial tensors and native/Mac audits.
- Allowed wording: mean-supervised ODE still has2.43/14.42 times the better reference error on Ring8/product16 under the registered budget.
- Forbidden wording: the ODE cannot represent the target, clipping is the cause, or an oracle-dependent method is deployable.
- Uncertainty: a fixed-budget nonconvex fitting gap does not identify expressivity versus parameterization/optimization.
- Next check: invertible function-class-preserving clock-coordinate control under ordinary pair-target FM.
- Decision: keep with scope.
