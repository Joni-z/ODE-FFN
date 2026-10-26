# Raw-affine error as a falsifier of a linear-dynamics explanation

Registered before computing this decomposition's outcomes. The prior Gaussian physical-flow review proposed but did not implement this question: if inability to fit physical-coordinate affine dynamics explains most ODE excess error, that excess should reside mostly in the raw-affine error component. The current P-depth screen is negative; this is a distinct error-localization test, not another parameter group/mode or rescue of sharing.

## Fixed inputs and comparisons

Use all21 final predictors from conservative_readout_20261026: ODE K2, SwiGLU128 and SwiGLU192, each direct/complete-gradient, plus direct SwiGLU384, across Gaussian2D/Ring8/Product16. These preserve the original professor scalar K2 layer in the ODE arms. At each of five saved times .1/.3/.5/.7/.9, use all512 actual common panel points and the previously saved predictions/oracle fields. No new predictor output, random data, model training, labels for training, or checkpoint selection.

Bind each final checkpoint, result, panel input/output, source manifest and study/provenance. Reproduce the recorded native field MSE for every panel. This five-panel mean is distinct from the uniform-time primary field metric.

## Exact finite-panel decomposition

For Q=[1,X] and prediction error E=prediction-oracle, use a float64 thin SVD of Q to project E onto the column space:

```text
E_aff = Q*Q^+*E
E_nonlin = E-E_aff
MSE_total = MSE_aff+MSE_nonlin.
```

Keep signed ODE-minus-reference differences in both components and every time. No clipping of negative excess components or per-time selection. Cross-check via independent QR least squares, orthogonality and invariance to a deterministic invertible rotation/translation of input coordinates. This projector concerns raw inputs, not learned features, the P spectrum, or previous final-head fitting.

## Cross-fit control

Fit the affine error on rows0:256, evaluate that affine predictor on rows256:512; then swap the halves. Save both coefficient matrices and all out-of-fold affine predictions. Record the decrease in MSE from subtracting this out-of-fold affine error prediction; it can be negative. This checks whether the localized error is reproducible across samples rather than a same-panel projection artifact. The oracle error labels are privileged diagnostics. Subtraction is not an achieved model, deployable improvement or proposed new training objective.

## Falsification convention and limits

Primary hypothesis: the Product16 **direct** ODE's five-panel mean excess is mostly affine relative to each of the three direct SwiGLU controls. For each positive total excess, require both the same-panel affine-excess share and cross-fit affine-reduction-excess share to be at least .50. Every reference must meet both requirements for this narrow explanation to survive. If a denominator is nonpositive, retain signed components and mark the share undefined rather than rank a negative gap.

The .50 boundary operationalizes "most"; it is not a significance test or an architecture-nomination threshold. A failure rejects the linear-error account as a sufficient explanation under this finite bank. A pass only localizes a deficit and does not distinguish capacity, optimization or feature interference. All gradient-readout and other-task rows remain visible as controls; do not replace the primary comparison after outcomes. No linear bypass, covariance teacher, spectral basis, time generator, optimizer or stopped architecture is selected by either outcome.

## Verification and resources

Before outcomes, check an exactly affine error field with known coefficients and an error with a constructed orthogonal residual, including out-of-fold recovery. Save source/protocol hashes. During the run, record process/time/memory, every input hash, projection coefficients/out-of-fold arrays and complete per-time/group tables. Rehash inputs at completion. A second QR implementation verifies all raw decompositions and cross-fit predictions. Run one bounded-memory local process, no GPU allocation.
