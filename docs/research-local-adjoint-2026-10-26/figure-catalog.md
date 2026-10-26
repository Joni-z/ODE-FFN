# Figure catalog

## endpoint-comparison.png / .pdf

Purpose: assess practical quality against all six fixed controls, rather than credit a gain only over original ODE. Sources: audited comparison.csv and every raw terminal result. Columns retain uniform-time primary MSE, five-panel Jacobian error and common-source SW2 separately. All seven rows use the predeclared order. No uncertainty bars or independent-repeat interpretation. Field MSE has a logarithmic axis; the other two axes use readable numeric ticks.

Observation: the candidate improves original and omitted-clock field error but loses to the cheaper near-parameter tied baseline and both SwiGLUs. Derivative and sampling guards also fail. The figure supports stopping this setting, not choosing a favorable metric/endpoint or claiming ODE impossibility.

## learning-and-P-trajectories.png / .pdf

Purpose: show fitting dynamics, actual learned P and an active local clock response. Sources: all recorded primary validation checkpoints,48 audited raw P spectra and the saved local activity probes. Left: all seven checkpoint curves. Middle: raw-P largest singular-value mean and layer range for the three ODE settings. Right: full arm's executed clock-adjoint RMS/input-RMS and omitted arm's explicitly labelled hypothetical term.

Observation: the generic field advantage persists through the final checkpoint; P and clock paths learn nontrivial values. Activity and larger singular values do not establish usefulness or a complete-field stability statement. Layer ranges are descriptive, not confidence intervals; original has six P matrices and each local setting has three. Both figures were visually inspected; only tick-label layout was corrected.
