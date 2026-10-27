# The field derivative relevant to this check

Write D(x,t) for the whole trained clean-image predictor, including attention,
normalization and external time conditioning. Define delta(t)=max(1-t,.05),
v=(D-x)/delta, and follow the numerical trajectory x'=v. Away from t=.95,
the material acceleration satisfies

    a = partial_t(v) + J_x(v) v
      = (D_t + J_D v - v)/delta - (delta'/delta) v.

Therefore

    t < .95:  a = (D_t + J_D v)/delta
    t > .95:  a = (D_t + J_D v - v)/delta.

If D is continuously differentiable, v is continuous at the cap boundary,
but the one-sided acceleration jumps by -v/.05. The original uniform grids
do not place a node exactly at .95. A naive smooth-field error expansion
across that interval is unjustified. This does not by itself prove that
Heun loses global second-order convergence: a single kink interval can still
have O(h^2) local contribution. It does make a clean factor-four ratio on
these three meshes an observation to check rather than an assumption.

Even away from the kink, local integration error depends on derivatives of
the entire external velocity, not only eigenvalues or singular values of a
local P. For one professor solve F=T_K(sP)x, its full derivative includes

    J_F = T_K(sP) + [P T_(K-1)(sP) x] outer grad(s).

SiLU, the second solve, residual/AdaLN gates, attention and image readout
compose with this derivative. Spectra of P alone cannot identify the cause
of the image FID ranking or bound full-field sampling error.

Finally, each fixed-N score is a distributional function of the numerical
endpoint Phi_N(e), not the paired clean-regression objective. A monotone
relationship between validation MSE and FID does not follow algebraically.
This check varies only Phi_N while holding D and all sources fixed; it
does not measure population oracle error or causal training dynamics.
