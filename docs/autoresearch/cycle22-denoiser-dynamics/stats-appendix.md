# Deterministic verification scope

There are no new trained models, sampling scores or independent training
repetitions. The observation unit is a fixed input/time derivative fixture.
The three unchanged analytic laws contribute 8 validation clean samples
crossed with 5 times each. Fixture Gaussian noise uses seed0. The original
dataset-generation seeds and splits remain unchanged; no test data are read.

All 120 cases are retained. Means are checked against the separate existing
NumPy marginal-velocity oracle; the reconstruction uses uncapped 1-t, matching
the analytic independent interpolation. The variance and FM PDE identities
are checked by complete float64 automatic derivatives, including every spatial
Hessian trace coordinate. There is no stochastic trace estimate, selected
Jacobian direction or fitted model whose error could be confused with an
identity residual. Maximum covariance/Jacobian discrepancy is 2.67e-15.

The independent audit retains 240 finite-difference records: both h=1e-4 and
h=5e-5 for every case. The registered gates are max absolute derivative error
<=2e-3*(1+max absolute AD derivative) and PDE residual
<=1e-2*(1+largest participating term). Both steps pass separately. Across
all records, maximum Jacobian, Laplacian, variance-derivative and time-derivative
discrepancies are 1.03e-7, 6.00e-6, 2.46e-5 and 1.47e-6. The largest FM residual
0.00339 is not a direct AD residual; coordinate factors amplify finite-
difference subtraction errors. All raw values and both steps remain visible.

The Gaussian and atomic controls are closed-form mathematical examples, not
additional benchmark datasets or empirical quality comparisons. Their risks
are exact population excess risks under the explicitly specified scalar
Gaussian law and loss denominator. They are not measured model MSE or FID.
The two atomic examples use different fixed priors. Their closure counterexample
does not exclude a prior-conditioned mechanism or additional hidden state.

The 8 probe values per time are shown individually, with a descriptive mean.
They are not independent trained replications, confidence intervals or a
population-risk estimate. No inferential test, p-value, multiple-comparison
winner selection or model-quality significance claim is appropriate here.

The proof is quantified over a stated finite-GMM class and fixed finite
dimension at interior t/r. Numerical rotation orthogonality is checked to
1e-12 for the product implementation, not asserted as exact real arithmetic.
Neither data-law application nor numerical kernels were certified by the
semantic reviewer; their receipts are separate. Semantic review is native,
same-family and provisional, with complete source-unit coverage but no
claim that execution coverage proves all possible obligations were found.
