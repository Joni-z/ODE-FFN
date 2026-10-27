# Cycle12 statistical appendix

## Units, cohort and estimands

This is a frozen-checkpoint diagnostic, not a new training comparison. It includes all six registered model families on both laws: twelve fitted endpoints, inherited model seed 0, no new optimizer updates. There are 2,048 saved common Gaussian sources per model. The 32 steps and three projected scopes are repeated computations on those same sources, not independent samples. Across families, each source is paired; a table with 2,359,296 signed source-step-scope values does not supply that many independent observations.

The primary diagnostic is signed temporal accounting of common-source model/oracle endpoint error. Every error energy is divided by its subspace dimension (128 full, 2 tangent, 126 normal). It compares to the existing capped Heun oracle endpoint, not the exact target draw. Smaller paired MSE means closer endpoints under this coupling; it does not imply smaller SW2. Existing field/SW2 columns are copied from frozen cycle11/cycle07 results, without new quality selection.

For each source, squared error is a scalar observation. Its finite-bank mean, source SD and standard error SD/sqrt(2048) are in [conditional-mc.csv](results/conditional-mc.csv) for all 36 model/scope combinations. The reported SD describes heterogeneity across source errors, not across trained models.

## Conditional source precision

Intervals are reporting-only additions after the registered identities passed. For fixed predictors and IID source draws, the displayed pointwise approximation for a mean is `mean +/- 1.959964*source_SD/sqrt(n)`. Forced fractions use a ratio-of-means delta approximation: for per-source forced energy A and total normal energy B, `r=mean(A)/mean(B)` and `SE=SD((A-r*B)/mean(B))/sqrt(n)`. They are not percentiles of source-wise fractions. Denominators are positive in all twelve cells; finite means/SDs/interval endpoints were checked. No normality of the individual squared errors is asserted. These are large-n approximations, not finite-sample coverage guarantees.

The source banks have been reused during research. Consequently these conditional precision indicators are not selection-adjusted confirmatory confidence statements about a chosen architecture. They also do not estimate optimization/training variation. No t-test, p-value, familywise significance claim or new model acceptance test is performed. The study's correctness gates are deterministic numerical equalities; no significance test is needed to verify them. Comparing overlapping/non-overlapping marginal intervals is not a paired model test. Multiple-comparison correction is therefore inapplicable to a rejected set of significance claims; all 48 summaries are retained, not filtered.

## Tangent endpoint error: mean, source SD and approximate interval

All rows use n=2,048. Full precision and the remaining scopes are in the CSV.

| Law | Family | Mean | Source SD | SE | Pointwise approximate 95% interval |
|---|---|---:|---:|---:|---:|
| thin | ode_k2 | 0.158101 | 0.296018 | 0.006541 | [0.145281, 0.170922] |
| thin | frame_k2 | 0.165753 | 0.520780 | 0.011508 | [0.143198, 0.188308] |
| thin | global_k2 | 0.159377 | 0.298122 | 0.006588 | [0.146466, 0.172289] |
| thin | frame_k1 | 0.157714 | 0.491441 | 0.010859 | [0.136430, 0.178998] |
| thin | swiglu192 | 0.166195 | 0.324967 | 0.007181 | [0.152121, 0.180269] |
| thin | swiglu257 | 0.167564 | 0.307721 | 0.006800 | [0.154236, 0.180891] |
| thick | ode_k2 | 0.216414 | 0.338457 | 0.007479 | [0.201756, 0.231073] |
| thick | frame_k2 | 0.265040 | 0.676095 | 0.014940 | [0.235758, 0.294321] |
| thick | global_k2 | 0.242139 | 0.319240 | 0.007054 | [0.228313, 0.255965] |
| thick | frame_k1 | 0.285160 | 0.724935 | 0.016019 | [0.253763, 0.316556] |
| thick | swiglu192 | 0.219922 | 0.443448 | 0.009799 | [0.200716, 0.239127] |
| thick | swiglu257 | 0.232655 | 0.546787 | 0.012082 | [0.208974, 0.256336] |

## Thick forced-normal energy fraction

These are approximate conditional ratio intervals with the same limitations. They describe the common structural component at each frozen readout, not an effect of changing P.

| Family | Fraction | SE | Pointwise delta approximate 95% interval |
|---|---:|---:|---:|
| ode_k2 | 0.918770 | 0.000948 | [0.916912, 0.920628] |
| frame_k2 | 0.920952 | 0.000844 | [0.919297, 0.922607] |
| global_k2 | 0.915265 | 0.000994 | [0.913316, 0.917213] |
| frame_k1 | 0.921792 | 0.000860 | [0.920107, 0.923478] |
| swiglu192 | 0.929477 | 0.000884 | [0.927745, 0.931210] |
| swiglu257 | 0.931543 | 0.000868 | [0.929842, 0.933245] |

## Descriptive effects and signed shares

- Thin input-frame K2 versus original K2: tangent paired MSE difference `0.007651863`, while existing tangent SW2 difference is `-0.045745581`. These are fixed-bank differences of different estimands, not a superiority test. Source-wise nearest-centroid disagreement is 11.71875% versus 8.44727%; it diagnoses assignment mismatch, not the cause of every error.
- Thin input-frame K2's registered four-bin signed shares are 20.3846%, 49.6867%, 28.9215% and 1.0071%. Thick shares are 22.8275%, 38.6117%, 22.6290% and 15.9318%. A signed share is a ratio of aggregate dot products to aggregate endpoint energy. It need not be a probability or a nonnegative causal attribution.
- The aggregate step table retains 12 negative signed rows out of 1,152. Bin shares sum to one within floating-point tolerance. Absolute-value sums or sums of squared transported defects would be different quantities and are not substituted.
- The prespecified last-bin threshold 0.5 is a descriptive label only. Tangent rows are all below this threshold; normal rows are above it. There is no threshold search or new acceptance decision.

## Numerical checks and coverage

All 12 native model endpoints, every learned one-step chain and 33 oracle-suffix branches per model replay bitwise. Independently recomputed scalar/bin values meet rtol 1e-7/atol 1e-9. Formal telescoping is exact on the recorded float32 bridges after conversion to float64. Normal scalar propagation deviates by at most 6.41e-7, below the registered 1e-4 fitted-data tolerance. The forced bridge prediction has max error below 2.89e-7; independent centered-basis QR and formal SVD projectors differ by at most 7.42e-12. Every fitted forced subspace has dimension 63. Nonlinear/affine float64 fixtures pass separately.

The audit shares the immutable nonlinear oracle and checkpoint-restoration ground truth. It independently computes native Heun arithmetic, projected/signed statistics, normal rates probed from the oracle, and the readout-null projector. It is not a second derivation of the complete mixture oracle. Formal output, registered source and prior seals are checked for changes.

## Interpretation boundaries

No interval rescues cycle11, nominates a hybrid sampler, or establishes an image-generation benefit. A path comparison can reveal differences in visited states and errors, but cannot separate their causal contributions without a registered intervention. A finite transported/local RMS ratio is not an operator-norm bound. A fixed-readout forced fraction cannot explain every independently fitted architecture difference. The theoretical restriction and the numerical observations are reported separately from untested design ideas.
