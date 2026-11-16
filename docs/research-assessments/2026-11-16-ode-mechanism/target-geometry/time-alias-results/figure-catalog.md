# Figure catalogue

- same-sampler-different-response: every original/alias Jacobian pair on the
  same128-pair/five-time grid. No error bars or new model selection. Notice that
  the metric can change while native sampling is preserved. This limits causal
  interpretation, without reversing the observed focal ordering.
- rounding-level-sample-differences: maximum endpoint change across sixteen
  fixed sources per checkpoint, actual FP32andFP64results. The logarithmic axis
  separates rounding scales. This verifies the construction rather than ranking
  model performance or extrapolating a sixteen-sampleSW2estimate.
