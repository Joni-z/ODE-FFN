# Product16: locate the density/SW2 disagreement

The complete Product16 density comparison rejects the proposed explanation
of the sharing SW2 lead. Stage-minus-shared KL estimates are -0.247 original
and -0.204 projected; both four-comparison-adjusted evaluation intervals are
negative. All eight Product16 cells pass the unchanged numerical gates.
This result does not justify selecting an oracle time window from a favorable
bin. Confirmation time profiles remain unopened.

## Frozen geometry diagnosis

Use the same 1024 qualified endpoints per model/field. Rotate into the native
known latent basis, where the target is a product of symmetric two-component
Gaussian mixtures. Keep the exact source/data/checkpoint hashes and both
ordinary controls. No training or new response is involved.

Inspect three distinct discrepancies:

1. Latent marginal displacement: one-dimensional W2 on each known latent axis.
   Use exact target-mixture midpoint quantiles, and calibrate the empirical
   sample floor using the prebound 1024 native target draws.
2. Component location and width: target posterior responsibilities define
   weighted residuals z_i-s*m_i, normalized by target standard deviation.
   Under the true target, each responsibility-weighted conditional residual
   mean is0and second moment is1. Report centroid energy and conditional
   variance separately; do not call these an exact decomposition of KL.
3. Cross-coordinate dependence: pairwise moments/covariances of the posterior
   sign means tanh(m_i*z_i/variance_i). These vanish between independent target
   coordinates. Calibrate finite-sample RMS against the same native target bank;
   pairwise statistics do not capture every higher-order dependence.

KL is sensitive to relative density and thin component widths; Euclidean W2
measures absolute displacement. Even one Gaussian illustrates the distinction:
for target standard deviation sigma and candidate a, W2 squared is
(a-sigma)^2, while reverse KL is[(a/sigma)^2-1-2log(a/sigma)]/2.
The learned models need not disagree because of dependence alone; test width,
centroid and dependence errors instead of choosing that explanation in advance.

This is a case study of the existing synthetic law, not a replacement headline
metric or a new architecture claim. Full Ring8 density results remain required.

## A population lower bound from the measured moments

Let pi_is=E_q[r_is(z_i)] and define q_is(z_i)=q_i(z_i)r_is(z_i)/pi_is.
The target posterior r_is is used under both distributions, so the joint
KL after adjoining the component label equals KL(q_i||p_i). The chain rule
and Gaussian maximum entropy at fixed mean/variance give:

```text
KL(q || product_i p_i) >= sum_i KL(q_i || p_i)
  = sum_i [ KL(pi_i || [1/2,1/2])
            + sum_s pi_is KL(q_is || Normal(s*m_i, sigma_i^2)) ]
 >= sum_i,s pi_is * [log(2*pi_is)
                    + (v_is + mu_is^2 - 1 - log(v_is))/2].
```

Here mu is the conditional residual mean divided by sigma_i and v is its
variance divided by sigma_i squared. This is a lower bound on the population
joint KL. It separates mode-mass, centroid and width contributions to that
bound; it is not an additive explanation of the entire observed KL. Finite
sample plug-in values have estimation bias, calibrated using native target
draws, and must not be advertised as exact finite-sample lower bounds.

The same Gaussian sources also supply an endpoint-law calibration without
training: rotate to independent Gaussian coordinates, apply their Gaussian
CDF, and invert each exact target-mixture CDF. This pushforward has the target
law, hence population KL0, but finite empirical SW2 against the separate
target bank remains positive. It calibrates the1024-sample floor; it is not
the native flow-matching oracle trajectory or a learned model.
