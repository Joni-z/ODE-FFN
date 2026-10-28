# Shared-state image risk

No new model is trained or nominated. The registered empirical-risk reversal fails.
The finite-bank reference fails its prespecified held-out-use guard.

| Domain | K1 / Swi86, MSE to bank mean |
|---|---:|
| paired_validation | 0.988757 |
| paired_training | 1.002248 |
| generated_K1 | 0.996893 |
| generated_Swi86 | 0.977018 |

These are equal-time clean-risk ratios on the same states; they are neither FID nor integrated native training risk. All100model rows and all20reference rows are retained.

| Time | Bank reference / K1, actual-image MSE | Bank reference / Swi86 |
|---|---:|---:|
| 13/128 | 0.99776 | 0.99384 |
| 38/128 | 1.56420 | 1.54915 |
| 64/128 | 3.11794 | 3.11013 |
| 90/128 | 6.20750 | 6.28760 |
| 115/128 | 11.32597 | 11.73769 |

## Interpretation

The exact conditional mean of a uniform finite training bank is not automatically a population denoiser. Its training/held-out behavior and posterior support must accompany any distance-to-reference ranking. This guard blocks using the measured bank-reference ranking to justify a population-quality architecture.
A cross-state ranking does not causally explain the earlier FID gap, identify a hidden P mechanism, or demonstrate neural memorization. No new loss, prior smoothing, bandwidth, sampling intervention or training rescue is selected.

## Claim candidates

- Claim: the exact bank conditional mean is unsuitable under the declared guard on these fixed held-out probes.
  - Evidence: reference-validity.csv, raw reference means, direct-distance audit.
  - Allowed wording: the finite empirical reference requires the reported scope.
  - Forbidden stronger wording: true population error, memorization of a neural model, or an explanation of FID.
  - Uncertainty: the population conditional mean is unknown; these paths and Gaussian starts are shared fixed probes.
  - Next check: use the result to choose an identifiable missing premise before another architectural experiment.
  - Decision: discard the proposed population-reference interpretation.
