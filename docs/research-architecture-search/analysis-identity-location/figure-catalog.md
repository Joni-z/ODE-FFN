# Figure definitions

- `location-factorial`: all30 primary endpoints, normalized to the better
  same-task SwiGLU. A cell reports its absolute MSE and ratio. No uncertainty
  intervals or inferential claim.
- `location-learning`: every nonlinear-task ODE trajectory and both controls;
  log vertical axis, shared declared5000-update horizon.
- `initial-spectral-equivalence`: first vector solve on product16, MM versus II;
  raw P singular values and the matching gamma*I+.6P reference maps.
- `centered-map-trajectories`: means across six vector solves, separately for
  raw P and the fixed-reference map. These are not actual conditional maps or
  full network Jacobians.

PDF and PNG versions are saved together. `plot-provenance.json` records the
plotter and audited input hashes; `operator-summary.csv` retains numeric
six-layer summaries. Read-only posthoc operator checks do not change training.
