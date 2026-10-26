# Estimand, uncertainty and verification

## Population risk and sampling unit

The target is the original analytic Product16 population, expressed in its known independent latent coordinates. No labels or coordinates are supplied to training. The exact cached-bank conditional mean is generally coupled, so D is excess-coordinate dependence relative to the population oracle, not automatically an optimization mistake under the bank objective.

The independent sampling unit is one pair of standardized latent draws. All16 coordinates, five times and seven models within a pair remain together during resampling. N=4096 paired rows and512 bootstrap resamples were fixed before evaluating new probes. Training seed remains0; the NumPy diagnostic and bootstrap seeds are0 as well.

C estimates E[(E[e_i|q_i,t])²], including conditional bias; it is not the variance of a centered conditional expectation. D estimates E[Var(e_i|q_i,t)]. T is the paired componentwise mean risk, not the earlier512-point panel metric or the uniform-time validation primary. The finite identity T=C+D holds before averaging; finite-sample C can be negative and is not clipped.

## Bootstrap and comparisons

Percentile95% intervals use the same512 paired row resamples for all outputs/times/models. Reported contrasts are effects for these fixed trained functions, conditional on the evaluation population and architecture states. There are no training-population p-values or claims of joint95% coverage over all comparisons. The prespecified all-reference convention operationalizes a majority explanation rather than performing an architecture-nomination test.

The point ratio DeltaD/DeltaT is reported only for positive DeltaT. Its interval is left undefined if bootstrap denominators cross zero; no resamples are discarded. This denominator boundary was an analysis validity convention and does not affect any primary comparison, whose denominators are all positive in every resample. Negative secondary total gaps retain signed components without a misleading ratio.

## Actual computation and checks

The first compatibility attempt terminated before generating new probes: saved gradient panels used create_graph=True, but the first replay used no_grad. Its maximum float32 discrepancy was4.77e-7. Source/output v1 and the explicit failure receipt are preserved. Correcting only the replay context reproduced the previously failing five panels bitwise, then source-v2 performed the actual experiment. No weights, cutoffs, probe count or target was changed.

Source-v2 completed normally in19.450s on one Torch CPU thread, chunks256, sampled peak RSS262.39MiB. All35 original native panels replay exactly. Thirty explicit-adjoint cases on actual new original/hybrid inputs agree with the automatic potential gradients within6.66e-15. Six frozen source files, original source provenance, seven checkpoints/results and35 original input/output panels are bound. All new standardized draws, latent A/B banks and actual used native/hybrid output coordinates are saved.

The analysis independently reconstructed T/C/D from actual fields and the analytic coordinate oracle. The external verifier then reconstructed C as T−D rather than using the cross-product implementation, and computed bootstrap means by explicit resampled-row averaging rather than the analyzer's bincount-weighted averages. All35 cases,7 pooled means,35 time means,6 contrasts and512 bootstrap draws pass. Maximum row and bootstrap differences are1.11e-15 and5.88e-15, respectively.

Runtime root:

```text
/Users/mr.z/ode-workspace-prep/architecture-search/coordinate_dependence_20261026
  source/ and output/              preserved failed compatibility attempt
  launch-v1-failure.json            terminal failure before new probes
  source-v2/ and READY-v2.json      corrected frozen evaluator
  output-v2/                       completed actual experiment
  analysis/                        row metrics, bootstrap indices and decisions
  independent-verification.json    separate replay and source hashes
```

No optimizer step, fitted correction or architecture change occurred. The Monte Carlo evidence changes the next research question; it is not a new winning model.
