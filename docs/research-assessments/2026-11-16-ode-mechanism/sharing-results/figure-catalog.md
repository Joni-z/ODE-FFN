# Figures

## sharing-comparison.png

Purpose: compare the complete fresh sharing cohort without mixing evaluation
streams. Gray crosses/circles are selection at15k/20k; teal dots are independent
confirmation at20k. Field MSE and pooled SW2 are separate columns. There are
no retraining error bars. Notice the shared/released gap and the continued
strength of ordinary Product16 controls. The next decision retains sharing
within this family but does not launch an image experiment.

## sharing-attribution.png

Purpose: verify that released stage computations are active. Top: complete
field MSE under native, averaged, reversed and base-only stages, at frozen
weights; the scale is logarithmic. Bottom: each layer's median feature-weighted
drift variation over five physical times. Shared null interventions are exactly
unchanged in the raw probe. These interventions measure reliance on the learned
schedule, not its benefit over a separately trained shared model.

## sharing-training.png

Purpose: distinguish training-fit failure from worse generated distributions.
Top: native validation oracle MSE versus training step. Bottom: pair-loss
difference, released minus shared, over the same native batch stream. Product16
released stages fit better but sample worse. Ring8 late losses are close.
This supports a distribution-matching question, while leaving optimizer
geometry part of the interpretation.

## sharing-cost-frontier.png

Purpose: place the structural result against ordinary computational references.
Horizontal: measured process CPU seconds for512samples with Heun128. Vertical:
confirmation pooled SW2. Labels include parameter counts. Sharing beats released
stages at equal dense work; Free ISRU remains much cheaper, and both ordinary
controls beat the ODE family on Product16. CPU timings are not GPU throughput.

All four images were rendered and visually inspected for labels, legends,
cropping and correspondence with the numeric tables.

## full-resolution/refinement.png

Purpose: separate outer-sampling resolution from hidden-map refinement on
the completed sharing cohort. Top: Heun128/256/512, inner4 fixed. Bottom:
inner4/16, Heun128 fixed. Every point uses all32768common samples. The outer
solver leaves the gap; hidden refinement changes its magnitude and Ring8bank
support. Those are different interventions: one integrates the same field,
the other changes the evaluated network at frozen weights. The figure was
rendered and visually checked. [Scores and interpretation](full-resolution/analysis.md).
