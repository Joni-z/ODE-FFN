# Figure interpretation

## crossed-field-risks
Purpose: test whether each learned weight set prefers its training solver.
Data: independently qualified full field predictions, 4,096 points/case.
Rows are fixed weights; columns are solver substitutions; all values are
per-coordinate MSE×1000. Ring8 both rows prefer midpoint. This contradicts the
strong diagonal compensation prediction; it does not establish sample quality.

## compensation-and-risk-accounting
Purpose: show the registered paired comparisons and the exact direction of bias.
Left intervals: four-comparison-adjusted conditional paired field-point bootstrap;
negative means the model prefers its training solver. Right: exact 2C/R and S/R
contributions to relative native-to-continuous risk change. Positive alignment
can make a more accurate hidden solve a worse field predictor. Next decision:
test crossed sampling with the existing ordinary controls.
