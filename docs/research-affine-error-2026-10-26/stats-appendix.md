# Statistical and verification scope

The unit is a fixed predictor evaluated on an existing common panel. There are21 predictors,105 model/time panels and18 prespecified aggregate ODE/reference comparisons. Five fixed times and512 samples per panel are repeated conditional evaluations, not105 independently trained models. No p-values or confidence intervals are claimed. Training seed0 and the unchanged training/data histories are part of the input provenance.

All summary energies are exact float64 finite-panel descriptions. The five-time mean differs from the uniform-time primary metric of the original training study. The raw-affine projector uses error against analytic population-oracle fields at those saved inputs. Its coefficients do not become labels, initialization or weights for any model. Orthogonal remainder means orthogonal to `[1,X]` in that finite sample geometry, not a proof about every nonlinear feature class.

Two-fold cross-fitting uses contiguous halves with roles swapped, exactly as registered. Predictions for each test half use coefficients fitted on the other half only. Native-minus-corrected MSE can be negative; no result is clipped. No cross-fit split, time or degree was tuned. A constructed exactly affine error transfers exactly in the pre-outcome algebra check.

The frozen driver verified SVD projections against a distinct QR solution, orthogonality, energy conservation, input-coordinate rotation/translation invariance and every recorded native field MSE. It binds165 original inputs, including all final checkpoints/results, model panels, common banks and the original source/provenance. Its actual run completed in.589s with sampled peak RSS47.64MiB and no GPU allocation.

An external script then rebuilt all105 errors from the original predictions and oracle banks, independently computed QR coefficients and out-of-fold estimates, checked all105 saved decomposition arrays and21/18 summaries/decisions, and rehashed every original output. Maximum metric discrepancy is1.11e-16. This verifies the saved-prediction arithmetic; it does not claim a new neural-network evaluation or a generalization result for a trained correction.

Runtime and evidence:

```text
/Users/mr.z/ode-workspace-prep/architecture-search/affine_error_20261026
  source/                       frozen driver and pre-outcome protocol
  READY.json                    pre-outcome algebra checks
  output/arrays/                105 saved error/projection/cross-fit arrays
  output/input-bindings.json    actual source/checkpoint/prediction bindings
  output/audit.json             driver verification
  output/decision.json          all three primary references fail
  independent-verification.json QR replay and all original output hashes
```

The50% threshold operationalizes "most" for this diagnostic, not a significance test or a gate selecting a trained architecture. Undefined shares for nonpositive aggregate gaps remain blank; their signed components remain in the tables.
