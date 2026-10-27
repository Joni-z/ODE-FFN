# Figures

## loss-versus-update.png / .pdf

Purpose: distinguish a large error component from its local update effect.
Source: audited checkpoints.csv and gradient-groups.csv; final-summary.csv
contains the final columns. Left: forced paired-loss fraction at step5000;
middle/right: relative hidden Adam update difference at1000/5000, percent.
Rows are thin/thick laws, all11families shown. No error bars. Values below
1e-12are plotted as zero in the final-update panel; exact values remain saved.

Observation: thick forced loss is about73%, but its direct hidden gradient
is zero. A clipping-mediated effect appears at1000and vanishes for the
particular final next batch. Thin panels are zero on the displayed scales.

Check: percentages use total paired loss; one-step effects use saved moments,
not retraining; each panel's checkpoint is labeled. Do not interpret empty
zero-height bars as missing experiments or infer whole-history causation.

## clip-and-P-directions.png / .pdf

Purpose: locate the measured clipping coupling and test gradient conflict.
Source: audited gradient-groups.csv at steps1000/2500/5000. Left: ratio of
forced-removed to full global clip factors. Right: P tangent/remaining-normal
gradient cosine for the eight families containing P. No error bars.

Observation: all thin clip ratios equal1; thick ratios approach1on these
three next batches. Final thick P cosines are positive, ruling out severe
cancellation on those probes. Different models are repeated comparisons,
not independent observations of a population trend.

Check: no cold near-zero cosine is shown; SwiGLU is absent only from the P
panel because it has no P; the remaining-normal component differs from the
forced component. No new optimizer or candidate is justified by the plot.
