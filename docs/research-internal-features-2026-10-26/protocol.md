# Does the controller contain a missing response that the final state does not expose?

Registered before new internal-feature extraction or outcomes, 26 October 2026. The preceding final-feature probe closed only about1% of the original ODE deficit. This does not reveal whether useful nonlinear responses exist at an earlier internal site.

## Immutable models, labels and folds

Use the same four direct step5000 predictors: original scalar K2 ODE and SwiGLU128/192/384. Restore conservative_readout_20261026's actual source and runtime buffers. Reuse all4096 population row pairs and five times from coordinate_dependence_20261026. These inputs are already exposed diagnostic data, not a fresh final test.

Use the original two reciprocal row folds. Only A rows in the calibration half fit coefficients. Held-out A points and all coordinate-retaining hybrids evaluate them. Keep each row's coordinates/times together. Use both exact population mean labels and ordinary pair labels on identical inputs, preserving their distinction from the original cached-bank training law. No oracle coordinates or nonlinear target basis are added to the features.

## Fixed-width residual probes

At each site retain the actual64-dimensional native feature h. Fit an affine correction [1,h] to the missing physical-space velocity:

```text
calibration target = velocity_label - frozen_native_prediction
evaluated field = frozen_native_prediction + affine_correction(h).
```

This keeps the existing predictor as a fixed offset. Every site gets the same65*16 free affine coefficients per label. It is a diagnostic of additional linearly accessible residual information, not a new same-cost architecture or deployable privileged model.

At the final normalized feature this class is exactly the ordinary affine-head class already tested: the native output is itself affine in that feature. Both label choices, all models/times/folds and every A/H prediction must reproduce the preceding final-head probe to numerical precision. At another site, the fixed native offset is an additional function; do not equate that setting with a standalone64-feature predictor.

ODE sites, for each of the three blocks:

1. normalized/time-conditioned FFN input z;
2. first controller hidden SiLU output;
3. first solve output;
4. first solve increment, computed as actual output minus input;
5. middle SiLU output;
6. second controller hidden SiLU output;
7. second solve increment;
8. FFN output;
9. complete residual-block output.

Also retain the input embedding and final normalized feature:29 ODE sites. The three generic backbones expose their corresponding conditioned FFN input, FFN output and residual-block output at every block, plus input embedding and final normalized feature:11 sites each. No subset of their wider internal activation channels is chosen. Each probed feature has64 channels; scalar clocks and concatenated feature banks are excluded.

Fit one correction shared across all five times, for each label, using float64 SVD with the existing1e-10 relative cutoff. Record all ranks, singular values, coefficient norms and normal-equation residuals. No new cutoff, penalty, temporal scope, width or feature subset is selected after outcomes. All62 sites and both labels are reported.

## Primary question and fixed decision

The final block's two controller hidden SiLU sites are primary, fixed before outcomes. Earlier blocks and other sites describe where accessibility changes and cannot replace the primary.

For each of the three generic references and each evaluation fold, let R0 be the original ODE's final-normalized residual probe risk, Rref the matching generic final-normalized probe risk, and Rc one primary controller's probe risk. Mean labels define the primary comparison:

```text
gap = R0-Rref
closure = (R0-Rc)/gap.
```

Require a positive gap. The late-controller-reservoir account survives only if closure>=.50 for BOTH controllers, BOTH folds and ALL three references. Otherwise it is not established by this test. Keep negative or above-one values. Do not nominate another earlier site or label after a failure. Absolute risks and other-coordinate leakage remain visible; closing half a gap is not baseline superiority.

The comparison is a finite diagnostic convention, not a population-capacity theorem or significance test. Show both reciprocal-fold values/ranges without calling them independent trained models or confidence intervals. Either result leaves open whether end-to-end retraining can change the representations.

## Interpretation and verification

For the original layer, its clock contribution to the local Jacobian is rank at most1; this does not bound the full network's expressivity. The middle SiLU and compositions remain nonlinear, and earlier vector-clock architectures already failed practical tests. A rich controller probe would establish residual accessibility, not prove information destruction or select a controller bypass.

Before outcomes test a constructed hidden-feature/compression witness, the exact K2 Jacobian rank-one identity via directional differences, residual/direct final-head equivalence, rank deficiency and fold isolation. Actual native hooks must preserve20 original float32 panels and the existing float64 predictions. All recorded sites must have64 channels.

Freeze new source/protocol and bind the immutable inputs. An independent manual forward pass must reconstruct every internal tensor; pivoted-QR fits must reproduce the main SVD prediction functions, all held-out A/H risks, and old final-probe compatibility. Do not compare rank-deficient coefficient vectors as though they were unique. Report numerical rank-boundary problems rather than changing the cutoff.

One local Torch/BLAS CPU thread, bounded per-model memory and process/time/memory telemetry. Save the A features, fit coefficients/spectra and all A/H predictions. Rehash sources/inputs at completion. No GPU reservation, network optimizer steps, old-model writes, stopped-branch rescue or new architecture nomination. Lenovo is offline; local work is available.
