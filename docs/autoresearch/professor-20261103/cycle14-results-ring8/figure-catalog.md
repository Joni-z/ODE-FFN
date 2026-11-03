# Figure interpretation

## gmm_ring8_2d-sampling.png

Purpose: check checkpoint and solver dependence of the fixed-model comparison.
Data: sampling_summary.csv, all five models, checkpoints 10k / 15k/20k. Lines are
means of eight common banks; shading is plus/minus one bank SD, not a training
confidence interval. Left Heun 32; right Heun 128. Notice the shared 15k-to-20k
increase in sampling discrepancy and close ISRU-D/selected-Swi64 lines.
Decision: no checkpoint-independent or solver-specific frontier win follows.
Inspection: all labels readable, all five models shown; overlapping SD bands
are intentionally supplemented by the paired-difference figure.

## precision/paired-banks-and-pooling.png

Purpose: show which reference comparisons survive changes in finite sampling.
Left: all eight 4,096-sample primary bank differences. Right: all eight dependent
28,672-sample omission-pool differences. Dotted horizontal lines are the full
32,768-sample pool differences; black zero separates wins/losses. All panels
use20k / Heun 128. Below zero favors ISRU-D. There are no uncertainty intervals.
Notice the negative pooled Free-ISRU/same-AdamW/larger-Swi192 contrasts and
the near-zero selected-Swi64 contrast with one omission reversal.
Decision: retain a modest fixed-model lead, avoid a strong ordinary-frontier
claim, and finish the pending transfer control before new architecture.
Inspection: signs, legends and full case membership checked against CSVs.

Both figures were rendered and visually inspected. They describe different
estimators; do not compare absolute4,096 versus32,768 SW2 magnitudes as a change
in model quality. No image-quality or FID claim is supported.
