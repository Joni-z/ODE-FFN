# Figures

- `generator-endpoints`: all six arms on all three tasks; each metric is
  normalized separately by the better same-machine SwiGLU. Values below one
  are favorable. No uncertainty bars are implied.
- `generator-learning`: both nonlinear tasks and all six complete validation
  oracle trajectories. Log vertical axis, common update horizon.
- `local-witness`: analytic product16 latent FM velocity and its ideal
  constructive additive coefficient (fixed cP plus the controller term s-c)
  for the predeclared coordinate at t=.5. It is
  an illustrative capacity witness, not a trained prediction or full-network
  lower bound. It uses the fixed-across-time certificate.

Both PDF and PNG are available. `plot-provenance.json` records exact script,
audit/trajectory/certificate input hashes. `comparison.csv` retains all
numeric endpoints and budgets.
