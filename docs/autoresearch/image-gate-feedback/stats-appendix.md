# Statistical and verification appendix

The analysis follows the [registered protocol](protocol.md). Model seed 0,
checkpoints 5,000/20,000 and all native training settings are inherited. The
ordinary controls remain Swi86 and SiLU131; this analysis has no new training,
FID, significance test or architecture-acceptance criterion.

## Cohort and state identity

Families: `ode_k1`, `ode_k2`, `swiglu86`, `silu131`. Domains: `paired` and
`generated`. Times: 13/128,38/128,64/128,90/128,115/128. Images: original
indices 0–63. Jacobian tokens: 0,21,42,63 in that order within each image.

The paired diagnostic uses the first 64 clean validation targets and first 64
canonical Gaussian sampling starts, with float32 `tX+(1-t)e` at the exact
mesh times. These are additional diagnostic inputs, never replacement quality
banks. Generated states are captured before the first/predictor Heun forward
at each declared step; Euler trial states are excluded. Paired forwards occur
after the entire original generation path. All eight endpoints match their
published prefixes bitwise. Every parameter and buffer remains unchanged.

There are 240 complete-core groups and 240 local ODELayer groups, each with 256
states: 61,440 core and 61,440 local derivatives. Each group retains actual
float32 inputs/outputs and every singular value. Local groups also retain
all 4,096 native token clocks per time/domain. The 48 clock-layer/domain groups
contain 983,040 scalar clock observations across five times.

## Derivatives and precision

The [derivation package](../../../DERIVATION_PACKAGE.md) defines J, R, v and g.
For each float32 captured h, weights and h are promoted to float64. Native-core
forwards are checked against the literal extension. For a complete FFN, the
second local derivative is evaluated at its promoted intermediate SiLU output;
separate ODE2 local statistics use the actually captured float32 intermediate.
These are explicitly different evaluation points. We do not differentiate a
rounded floating-point function as a discrete mathematical map.

`full_gain` and `full_smallest` are the largest/smallest singular values of J.
`reference_gain`/`reference_smallest` use the native-value anchored reference R.
`full_to_reference` is the per-state ratio of the two largest singular values;
its median is not a ratio of pooled medians. `feedback_norm` is `||J-R||2`.
`relative_feedback` divides this by `||R||2`. A zero reference norm is excluded
and counted explicitly. There are no such exclusions. Ordinary references and
clock fields are not applicable and appear as empty cells, not zero effects.

For an individual ODELayer, `gradient_norm=||grad s||2`,
`feedback_norm=||v||2||g||2`, and `chi=g^T M^(-1)v` when the base is numerically
invertible. The criterion is
`sigma_min(M) > eps64 * 64 * sigma_max(M)`. Every local base passes; zero
inverse exclusions. Chi is not a log determinant. The determinant multiplier
is `1+chi`. Its sign or magnitude is not a denoising objective. P's spectral
norm is retained in each private local spectrum file. The full FFN feedback
has numerical rank at most 2; maximum third singular value is 2.96e-15.

Each group and pooled cohort reports minimum, 5th percentile, median,
95th percentile and maximum using NumPy's default linear quantiles.
`pooled.csv` weights every state equally across all five times and all three
cores or six local layers within one family/checkpoint/domain. The cohort has
3,840 core or 7,680 local probes per pooled row. `time-profiles.csv` pools only
three complete cores at one time, giving 768 states per row. No model/time is
selected by any statistic.

## Clock variance

For each separately centered clock array s[time,image,token], define time
means mu_t, image means mu_ti and grand mean mu. All means are population means.

```text
Vtotal = mean((s - mu)^2)
Vtime  = mean((mu_t - mu)^2)
Vimage = mean((mu_ti - mu_t)^2)
Vtoken = mean((s - mu_ti)^2)
Vtotal = Vtime + Vimage + Vtoken
```

`clock-variance.csv` retains all 48 layer/domain rows. `variance-pooled.csv`
sums each raw component over the six separately centered clocks, then divides
by their summed total. It does not introduce variance between layer means.
The decomposition is a finite-array identity, not a causal or mutual
information decomposition. Cases are the first clock in block 0, image 0/token 0,
at all five times in both domains for all four ODE checkpoints; all 40 are
published, without selecting an extreme.

## Independent verification and runtime

The fresh documented preflight ran verbatim once, exec session 21100, exit 0.
Eight endpoint banks and 48 promoted forward witnesses pass. Data-free fixtures
also check complete/held-clock derivatives, noncommuting P differentials,
a bounded-P counterexample, the rank identity and the Heun chain rule.

Controller 25342 exits 0. All eight analysis children and both controller phases
exit 0. The independent audit writes the Heun arithmetic explicitly, recaptures
all native states/outputs/clocks bitwise, and constructs separate elementary
forward expressions for automatic differentiation at every selected state.
All 122,880 spectral/clock comparisons pass the registered 1e-8 relative/
1e-10 absolute tolerances; the largest observed difference is 6.66e-15.
Native-to-promoted forward maximum error is 6.76e-6. This was reported rather
than changing any native precision, checkpoint or sampling arithmetic.

The full monitored analysis and audit take 241.13 seconds on Mac mini CPU.
At most two one-thread analysis workers run together; the independent audit
uses one. No GPU is reserved, no packages are installed and no remote host is
used. Raw captures, spectra, source snapshots and monitoring stay under
`/Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-gate-feedback`.
Curated outputs and reports are public; checkpoints remain under their
unchanged parent seals. The submitted Report1 is untouched.
