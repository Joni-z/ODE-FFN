# A train-only Gaussian reference does not match the image models

Zhizhe Zhang · October 28, 2026

## Question and decision

Does a fixed train-only Gaussian conditional-mean predictor reproduce the
neural image screen? PPCA59 gives FID **293.73**, versus **174.71** for cold
fixed-P training and **173.44** for native K1. The isotropic reference gives
**405.30**. Both fail every registered competitiveness gate. Stop these two
references; no rank, residual-noise, covariance-form, mixture or sampler rescue.
This is a classical reference result, not a successful ODELayer design.

## Complete primary comparison

The neural rows below are their registered 20k endpoints. PPCA/isotropic are
train-only density fits with no SGD updates; the table does not imply equal
training work. All 34 endpoints, including 5k/10k context, remain in
[comparison.csv](results/comparison.csv). Lower is better in each metric.

| Model | Weighted paired MSE | Clean paired MSE | FID |
|---|---:|---:|---:|
| K1 native | 0.160267 | 0.076153 | 173.44 |
| Fixed 0.6 P | 0.160392 | 0.076197 | 174.71 |
| Matched-init SiLU64 | 0.160598 | 0.076280 | 178.06 |
| PPCA59 | 0.185691 | 0.082118 | 293.73 |
| Isotropic Gaussian | 0.563128 | 0.219926 | 405.30 |
| Native posthoc 0.6 | 0.167715 | 0.078856 | 216.07 |
| Old K1 time | 0.160664 | 0.076303 | 180.86 |
| Old K1 pooled | 0.161006 | 0.076366 | 200.05 |
| Old K1 constant | 0.161452 | 0.076417 | 215.56 |
| K2 native | 0.160527 | 0.076292 | 180.78 |
| Old K2 constant | 0.163194 | 0.076846 | 226.23 |
| Swi86 | 0.159944 | 0.076124 | 195.31 |
| Swi129 | 0.157935 | 0.075416 | 183.57 |
| Default SiLU64 | 0.159842 | 0.075990 | 190.76 |
| SiLU131 | 0.159023 | 0.075724 | 185.89 |
| SiLU193 | 0.158280 | 0.075516 | 185.30 |

Relative to fixed P, PPCA FID is 119.03 points (68.13%)
higher. PPCA improves over the isotropic reference by 111.56 points
(27.53%), but these two fits have different scalar counts;
this is no parameter-matched estimate of a covariance mechanism. Their
[feature-mean and covariance terms](results/figures/reference-comparison.png)
are accounting terms, not identified causes. The fixed
[time panels](results/figures/time-panels.png) localize much of the PPCA fitting gap
to the middle times. It is not uniformly worse: at t=0.9 its paired error is
lower than native K1. No time or endpoint is selected from this pattern.

## What was fitted and checked

Use only the original 45,000 CIFAR-10 training indices; the other 5,000 indices
are validation, and the official test batch is unused. PPCA stores the leading
59 training covariance directions and a single average residual variance.
Rank 59 was fixed by a 185,744-scalar cap before outcomes: 184,380 float64
values (1,475,040 bytes); rank 60 would exceed that scalar cap. These are
storage values, not identical free dimensions or precision bytes to a neural
model. The [spectrum](results/figures/fitted-spectrum.png) accounts for 85.78%
of training variance in those 59 directions; it makes no semantic-quality
claim. Gaussian density maximum likelihood differs from neural FM regression.

For each fitted prior, use its exact Gaussian conditional mean, followed by
the unchanged CPU float32 Heun128 sampler with denominator cap 0.05. The
predictor uses float64 statistics and returns float32. Keep the same 1,024
paired validation states, five 64-pair panels, 1,024 common starts and 2,048
real Inception references. This is a finite exposed screening benchmark.
The [methods appendix](stats-appendix.md) records numerical and cost scopes.

An independent raw-moment covariance calculation checks the fit. Independent
Woodbury conditioning/manual Heun checks all new endpoints numerically;
PPCA's maximum sample difference is 1.19e-7, not bitwise equality. All 2,048
new feature rows and 2,048 reference rows replay bitwise; all 34 FIDs pass
the independent 1e-8 comparison. Formal source, input arrays and parent seals
are preserved. These checks establish calculation integrity, not model quality.

## Fixed cases and a reporting-only follow-up

The [fixed-example sheet](results/figures/fixed-examples.png) shows starts 0–7
for every primary row. At start 3, PPCA retains the broad dark region against
a blue background; at start 5 it retains a bright central region. Both have
strong fine speckling. Isotropic outputs lack those coarse layouts. The
neural outputs also remain blurred; the result is not high-quality generation.
These examples were fixed before quality outcomes, not selected for this account.

After viewing that sheet, a separate descriptive analysis projected all 16
primary sample banks and the original real reference into the already-fitted
PCA59 coordinates. It changes no fit, sample, metric or gate. It is explicitly
post-outcome reporting, not a preregistered mechanism test or a FID decomposition.
The global PCA basis is not a known image-manifold tangent.

| Original output | Retained RMS | Complement RMS |
|---|---:|---:|
| Real reference | 0.46111 | 0.18670 |
| PPCA59 | 0.44534 | 0.18537 |
| K1 native | 0.47437 | 0.15859 |
| Fixed 0.6 P | 0.47711 | 0.15691 |
| Matched-init SiLU64 | 0.46955 | 0.15699 |

RMS divides squared norm by the original 3,072 coordinates. PPCA complement
RMS is close to the real reference even though its FID is much worse. Matching
this aggregate energy is therefore insufficient for quality. Native/PPCA
retained coefficients have centered aggregate correlation 0.935 on common
starts; 55.77% of their paired squared output difference lies in the complement.
Neither number establishes that PPCA and the neural function are equivalent,
that complement noise causes the FID gap, or that all affine predictors fail.
See the [full decomposition](results/subspace-analysis/subspaces.csv) and
[figure](results/subspace-analysis/subspace-comparison.png).

The original reporting-producer terminal tool receipt was lost with a truncated
observation. Its process is absent and complete files are preserved; no exit
code is invented and the producer is not rerun. The closure verifier separately
recomputes every reported decomposition value from original arrays.

## Consequences for ODELayer

The successful cold fixed-P control already shows that the observed finite
margin does not require adaptive local clocks. For K1 and fixed positive c,
each core `h+c P h` is just `W h` with `W=I+cP`; this is a bijection onto the
ordinary bias-free square matrix. The full fixed-clock FFN has the same
function class as the corresponding SiLU block. Initialization and optimizer
coordinates remain separate explanations, not an ODE-specific mechanism.

The Gaussian reference now supplies a further limit: this particular global
second-order image model is insufficient. It does not identify what a neural
ODE block contributes. The projector analysis also gives no reason to tune
image residual variance or impose an internal P spectrum from output energies.
No new spectral penalty, extra Taylor term or candidate architecture is
selected by these results. A next design needs a nonredundant operation and
a test against its ordinary implementation, with fixed-P/matched-init/SwiGLU
controls retained. The broader research goal remains open.

## Claim candidates

- **Observed:** this fixed PPCA59 estimator fails the declared image screen.
  Evidence: all primary metrics, gate table and fixed cases. Keep this claim;
  do not promote it to universal Gaussian/affine impossibility.
- **Observed, exploratory:** comparable complement RMS can coexist with much
  worse FID. Evidence: original-array decomposition and FID table. Keep only
  the descriptive statement; no causal attribution to residual noise.
- **Not established:** a useful ODE-specific successor or a benefit from a P-only
  constraint. Neither the reference nor its mathematical exactness tests this.

[Protocol](protocol.md) · [Derivation](theory.md) ·
[Figures and interpretation](figure-catalog.md) ·
[Cold controls](../image-clock-training/analysis-report.md)
