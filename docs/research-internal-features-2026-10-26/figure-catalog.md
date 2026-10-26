# Figure interpretation

`figures/ODE-internal-sites.png` and `.pdf`

- Purpose: inspect every original-ODE internal residual probe without selecting a favorable depth or activation.
- Source: all three blocks, nine internal site types, both calibration labels; input/final endpoints are shown below each panel.
- Encoding: pooled two-fold five-time paired-redraw MSE, exact cell annotations, common linear color scale across the two panels. Orange outlines identify the predeclared last-block controller sites. No confidence interval is implied.
- Observation: internal mean-label probes stay near the final-feature result and far above generic references. The primary controller sites are slightly worse than final features.
- Decision: no evidence for the registered late-controller reservoir; no new controller bypass or selected earlier site. A linear single-site failure does not exclude nonlinear or distributed information.

`figures/common-sites.png` and `.pdf`

- Purpose: retain all corresponding64-channel sites in all four backbones, with the same residual-probe format.
- Source: input embedding, conditioned FFN inputs/outputs and residual states at each block, and final normalized feature, for both label choices.
- Encoding: exact risks with a logarithmic color scale. Each native predictor remains a fixed offset; these are not standalone input-only predictors. Neither panel is an architecture training result.
- Observation: ODE/generic separation persists throughout the common sites. Pair-label probes are not retrospectively selected over the primary mean-label comparison.
- Decision: changing the final linear readout or reading a controller feature is not supported as a useful architecture direction by this study.
