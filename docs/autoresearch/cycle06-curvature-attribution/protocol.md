# Read-only curvature attribution after cycle05

Registered after all cycle05 outcomes, before these counterfactual outputs. This is a descriptive mechanism study, not a cycle05 rescue or a new quality candidate. No training, coefficient optimization, changed initialization or new data is permitted.

Use all five final cycle05 quadrature2 checkpoints and every point of their existing full field banks and five512-anchor Jacobian panels. At each checkpoint replace all two-node gates by their single-midpoint expression, retaining every learned tensor. Compare the actual fitted function with that fixed counterfactual, not with a retrained model. Both the candidate and the independently trained midpoint remain inferior to matched-budget Swi143 on the three16D laws; that decision cannot change.

For predictions f_Q and f_M and oracle v, set `delta=f_Q-f_M`, `e_M=f_M-v`. Verify in float64 on the full arrays

```text
MSE(f_Q)-MSE(f_M)=2*mean(e_M*delta)+mean(delta^2).
```

Report every law's signed alignment, displacement energy, prediction-RMS ratio, field risks and all time-panel differences. Record three cases per law with the largest absolute squared-error change, using row index to break ties; this is descriptive case selection, not an unexposed holdout or model selection. Keep raw arrays private.

Separately set C=0 in copies of all five fitted models, leaving their nonzero output heads and other tensors fixed. At the first32 fixed probe anchors verify the **complete network** C-JVP identity in float64. For each block individually and for all three jointly, map deterministic `dC=.01*sin(arange(C.numel())+block_index)` to ordinary tangents `dWg_left=.5*dC*Wg_right`, `dbg_left=.5*dC*bg_right`; every other tangent is zero. Compare JVPs within1e-9relative/1e-10absolute and retain their absolute magnitudes to rule out vacuous zero-head agreement. This does not measure the tangent rank at the actual nonzero trained C.

Freeze this source/protocol and the entire sealed cycle05. Replay actual candidate field predictions and Jacobians against saved arrays. Verify the endpoint-error decomposition, all recorded bank lengths, finite values and prior hashes. The fixed midpoint counterfactual is ordinary SwiGLU after absorbing the matrix into gate projections. It is not a separately trained or nominated ordinary endpoint. Report functional attribution and the proven initialization identity; do not claim a causal explanation of optimizer failure.

Use one single-thread Mac CPU process, record PID/time/RSS and terminal status; timeout600seconds. No GPU, Lenovo assignment or MBP/torch dependency. Publish code, small tables and figures; checkpoint and prediction arrays stay private. This study selects no new architecture or training extension.
