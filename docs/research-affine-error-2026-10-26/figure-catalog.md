# Figure catalog

## all-model-error-components.png / .pdf

Purpose: compare affine and orthogonal errors across all21 predictors, both readouts and all three tasks. Source: the verified five-panel means in aggregates.csv. Blue/orange stacks sum to native five-panel MSE. Each dataset uses the same y range in both rows, so the large Product16 readout-associated change is visible. S384 remains the direct-readout cost control in the gradient row and is hatched explicitly. No error bars or confidence intervals.

Observation: the large direct Product16 deficit is mainly orange, whereas the Gaussian error is mostly blue. The pattern depends on the readout and is not a global ODE impossibility statement. All controls are shown, no fitted correction is deployed, and initialization/training quality is not inferred from the projection alone.

## Product16-affine-falsifier.png / .pdf

Purpose: show the declared primary falsifier and its cross-fit control. Left: signed aggregate ODE excess is positive versus all three references; stacks separate affine and nonlinear contributions, with affine percentages labelled. Right: native-minus-out-of-fold-corrected diagnostic error, including the negative ODE value. Sources: verified gaps.csv and aggregates.csv. No time/ref/degree selection and no clipped negative values.

Observation: affine excess remains near4% across references, and out-of-fold affine subtraction fails to help ODE. These two observations reject the majority-linear explanation on the fixed bank without pretending that diagnostic subtraction is an achieved architecture. Both PNGs were visually inspected with legible labels and all comparisons present.
