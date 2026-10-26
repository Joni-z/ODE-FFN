# Gaussian denoising and the original ODELayer

This directory connects the population denoising objective to the professor's original Taylor-polynomial layer. It contains mathematical derivations and deterministic verification, not a training study.

- [Technical derivation](derivation.md): Gaussian posterior, optimal velocity, exact sampling flow, and scoped statements about P, gates, Taylor truncation, and temporal expressivity.
- [Chinese walkthrough](explanation-zh.md): a plain-text explanation using an elliptical Gaussian supported on a plane.
- [Verification script](verify_math.py) and [recorded results](math-checks.json).
- [Plan](task_plan.md) and [working notes](notes.md).

The central fixed-time K1 correspondence is `P=(1-t)B_t/s` when a single constant-gate layer directly predicts the clean data in data coordinates. Default K2 requires a different matrix equation; neither formula identifies an arbitrary hidden P with the physical velocity Jacobian. A full Transformer is outside the single-layer impossibility statements.

No new architecture has been selected and no training has been launched from this derivation. The analytic Gaussian is a calibration problem that a time-conditioned linear model can solve; it does not establish an ODE-specific advantage.

Run the verification with a Python environment containing NumPy and SymPy:

```bash
python docs/theory-gaussian-ode-layer/verify_math.py
```
