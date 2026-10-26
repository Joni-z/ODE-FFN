# Figure catalog

- partial-field-errors.png/pdf: all18 saved states, all four partial sums. Annotations are MSE ratios to that state's full A evaluation. Color encodes log10 of the ratio, centered at1; rows distinguish A-trained and B-trained weights. A B-trained row is not evaluated with its native B output. No partial sum is a candidate.
- skew-cancellation.png/pdf: Product16 at t=.5, A-trained states. Bars show each component's mean squared Frobenius skew norm and the complete sum's norm. The last bar is numerically near zero; it is not a missing observation. Individual components' skew parts cancel.

Every field bank and five512-input panels are retained in the CSV/NPZ records. Plots illustrate fixed-weight computation and make no retraining, uncertainty or speed claim. The report-only renderer was inspected at saved resolution and preserved separately from the frozen evaluator.
