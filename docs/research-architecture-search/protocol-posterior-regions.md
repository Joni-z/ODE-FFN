# Saved-field posterior-region diagnostic

Registered before computing the entropy-region outcomes on 25 October 2026. This is a read-only diagnostic, not a new architecture experiment.

## Question and fixed inputs

Does the Product16 ODE/SwiGLU velocity-error gap concentrate in coordinates whose binary mixture-component posterior is ambiguous, or in coordinates whose component is already confidently identified?

Use only the Stage17 final `vector_parent`, `swiglu_budget` (SwiGLU96), and `swiglu` (SwiGLU128) arrays for `rotated_product_bimodal_16d`, at times `.1, .3, .5, .7, .9`: 15 existing NPZ files, 512 anchors and 16 coordinates each. Inputs and analytic oracle fields must agree exactly across the three arms at each time. Verify each array against its native summary hash and the distribution against the saved data manifest. No additional inputs, labels, model evaluations, checkpoints, training, or Ring8 extension are included.

The earlier `scripts/diagnose_posterior_regions.py` partitions exposed 20,000-step operator-study **Jacobian row error** by derivative sign and half-log-odds; its two velocity examples do not answer this aggregate velocity-error question for the paired Stage17 models. Preserve that earlier analysis unchanged.

## Coordinates and partitions

Read the orthogonal rotation `Q`, component locations `mu`, and within-component variances `var` from the Product16 manifest. Set `u=x Q`, `a=1-t`, `c=a^2+t^2 var`, and `q=t mu u/c`, coordinatewise. The positive-component posterior is `p=(1+tanh(q))/2`; normalized binary entropy is `h=H(p)/log(2)`. Compute entropy stably from the absolute logit `r=2 abs(q)` as `[log1p(exp(-r)) + r exp(-r)/(1+exp(-r))]/log(2)`.

Fix all four entropy bins now: `[0,.1)`, `[.1,.5)`, `[.5,.9)`, and `[.9,1]`. Low entropy indicates confident component identity; high entropy indicates ambiguous identity. Use the same masks for all arms, retain all bins including empty bins, and do not change cutoffs after reading outcomes. The transformed residual is `(prediction-oracle) Q`. Its squared entries are coordinatewise velocity errors; orthogonality preserves total squared error.

## Recorded quantities and checks

Write one CSV containing every arm/time/bin plus `all` rows and five-time aggregates. For each bin record coordinate count/fraction, conditional mean squared error, contribution to total componentwise MSE (sum of squared errors in the bin divided by all coordinates in the panel), and its fraction of total error. Empty conditional means are blank, while empty contributions are zero. Five-time aggregates pool all coordinates with equal weight; their contributions must equal the average of the five panel contributions. Also report each ODE-minus-baseline contribution and its share of the total gap, separately for both fixed SwiGLU references. Signed gap shares are descriptive and can be negative.

Reconstruct the already-saved oracle algebraically from the manifest only to verify the coordinate convention; do not replace any target. Require the latent and ambient MSEs to agree, all bins to partition every coordinate, and the five-panel averages to reproduce the native field summaries. Store input, protocol and analysis-code hashes with the CSV and a short readout.

## Interpretation fixed before outcomes

Dominant excess error in confident coordinates weakens an **ambiguous-boundary-only** account of the ODE gap. Dominant excess in ambiguous coordinates would motivate examining the relevant functional features; it would not establish that sigmoid features, a particular architecture, or an ODE construction repairs them. Report per-time and conditional errors alongside pooled contributions so bin prevalence cannot be mistaken for difficulty.

These networks do not predict component probabilities explicitly. Posterior-region membership locates velocity error but cannot decompose it causally into component-assignment error versus within-component slope, offset, or cross-coordinate error. The calculation uses exposed saved evaluations, not a fresh confirmation set; coordinates are repeated measurements, not independent training replicates. It provides neither an architecture-wide capacity bound nor a training recommendation by itself.
