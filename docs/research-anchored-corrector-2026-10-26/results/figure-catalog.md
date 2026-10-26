# Figure catalog

No figure has error bars; raw pointwise paired evaluation intervals are in paired-evaluation.csv.

- comparison.pdf/png: all six new endpoints in field, Jacobian and generation metrics. The dashed line in each panel is the best combined generic value, potentially a different arm for each metric. Notice that anchoring worsens every primary family. This stops the setting, without claiming an impossibility result.
- iteration-behavior.pdf/png: left, full validation error versus internal update count; right, anchored residual on the256-input probe. Notice the ODE residual decrease alongside worsening validation prediction. The panels use different banks; the separate fixed-point-results files establish a paired same-input comparison. This blocks interpreting numerical convergence as task correctness.
