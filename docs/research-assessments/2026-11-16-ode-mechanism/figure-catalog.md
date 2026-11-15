# Completed figures and interpretation

## final-comparison.png

[Figure](figures/final-comparison.png). Seven model families on two tasks;
rows are field MSE/pooled SW2, gray selection versus teal confirmation at20k.
Each score uses its bound evaluation arrays. There are no error bars; dots
are not retraining replicates. State flow improves within-family sampling,
while free coefficients/stages stay close to their fixed references.
Product16 ordinary controls remain stronger. Do not describe this as a
general ODE replacement advantage.

## composition-diagnosis.png

[Figure](figures/composition-diagnosis.png). Per physical time, median ratios
over six local layers and eight states compare truncation, restarted-generator
composition and state feedback. The logarithmic axis separates small matrix-
exponential approximation error from a substantial operator difference.
Ratios concern hidden increments, not global Lipschitz constants or FID.

## transport-calibration.png

[Figure](figures/transport-calibration.png). Sixty-four common sources compare
the first-order variational prediction with actual mixed-field endpoints at
four perturbation weights. The large residual at full weight rejects the
proposed extrapolation. The smallest displayed weight is already too large
for a good Product16 linear approximation; the separately archived0.001
calibration verifies the genuinely local regime. No curve is a sampling-
distribution score or training-variation estimate.

## trained-convergence.png

[Figure](figures/trained-convergence.png). Learned-state local-map error versus
internal substeps4/8/16/32/64, against a DOP853 reference verified at tighter
tolerance. Lines are medians over six layers; shading is the layer IQR, not
a confidence interval. Fixed midpoint retains approximately second order;
free stages approach first order. This demonstrates numerical structure,
not an established generation-quality gain or a new numerical method.

## symmetry-full.png and internal-resolution-full.png

The completed25-case cohort is plotted separately. The symmetry plot compares
seven original/projected Ring8 fields on identical C8-stratified sources and
targets. Its4,096base points are independent; their rotations are not. State
flow's original lead disappears after projection, motivating the targeted
sharing-symmetry follow-up. Equal projected sector counts are constructed by
the group action. The internal-resolution plot uses the original iid pooled
arrays, showing4->16steps at fixed weights. Scores from these two protocols
must not be mixed. Both figures were rendered and visually checked.

All figures use archived measured JSON/CSV. The additional four sharing figures,
their purposes and limits are in [the sharing catalog](sharing-results/figure-catalog.md).
