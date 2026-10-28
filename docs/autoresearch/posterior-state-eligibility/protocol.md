# Data-free posterior-state identity checks

Register source before outputs. No data, fitted prior, neural model, image
quality evaluation or changed benchmark. This does not supply formal
experiment readiness for a later training family.

Use six explicitly fixed priors: symmetric atoms, the cycle22 skew atom
counterexample, a 2D shared covariance mixture, a 3D unequal covariance
mixture, a 2D singular mixture and a single 2D Gaussian. Three fixed
observations and precisions0/.2/1/4 give72fixed-observation cases. Three
fixed FM states and times.15/.5/.85/.98 give72time-coordinate cases, including
the active denominator cap. No case is added or removed after outcomes.

Checks:
- NumPy solve-based conditioning against independent Torch eigen-coordinate
  conditioning, complete mean/covariance/weight states.
- Exact degree-three posterior moments from product three-point Gaussian
  cubature, not random Monte Carlo.
- AD precision derivatives of every component state/weight; the complete
  global mean drift and state Jacobian, with the third-moment contribution.
- Ordinary atom/shared-covariance attention and unequal-covariance quadratic
  gates/affine values. Check the nonzero logit-ratio Hessian witness rather
  than claiming all priors are linear-key attention.
- FM fixed-z, complete component/weight time derivatives, state and
  capped-material derivatives, retaining the moving observation term;
  check ordinary endpoints there too and record the omitted-term discrepancy.
- Integrate the component/log-weight/covariance ODE from precision0 to
  .2/1/4 for each prior/observation with DOP853,rtol1e-10,atol1e-12, giving54
  endpoint checks. Compare complete state and normalized weights to the
  direct ordinary evaluation. No probability/covariance clipping.

Float64 algebra/AD/cubature comparisons require rtol/atol3e-10. Integrated
state comparisons require max absolute difference below5e-8. Preserve all
errors and the fixed counterexample. These are frozen arithmetic checks,
not model-selection thresholds or statistical inference.

Use the unchanged warm .venv-test CPU interpreter, one thread and no package
installation. Expected runtime below30seconds, ceiling120seconds and
below0.5GiB new disk. Save frozen source, its commit, output arrays/tables,
exact command and actual consumed terminal tool exit. Never overwrite an
existing runtime or restart a live handle after an observation timeout.

Decision: if identities pass and the exact endpoint is the stated ordinary
map, do not launch a trained ODE candidate based solely on this construction.
If an identity fails, preserve the failed source/results and correct the
mathematical scope before any follow-on work. Neither outcome completes the
broader ODELayer research goal. No Gaussian image-reference rescue is selected.
