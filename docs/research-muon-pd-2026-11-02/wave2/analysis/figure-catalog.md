# Figure interpretation

All figures are reproduced by `studies/muon_pd_20261102/wave2/render_wave1.py`.
PNG previews and standalone PDF versions are provided. No error bars are implied.

## 01-all-endpoints

Purpose: show every first-wave configuration on both quality metrics, including
Gaussian, and avoid a selected scalar-ODE-only comparison. Source: parent
`endpoints/all-configurations.csv`, independently checked against replay.
Caption must explain lower-is-better, log field axis, family markers, optimizer
colors,20k budget and the reference-versus-reference floor. Observe that a lower
field error does not always mean a lower sample distance. Decision: the ordinary
controls must remain in all follow-ups; no universal optimizer win is supported.

## 02-p-gates-jacobians

Purpose: distinguish improved field fitting from controlled hidden dynamics.
Source: curves.csv; only scalar ODE is plotted so the matrices have a common
shape/meaning. Caption must specify six-matrix averages, saturation definition,
and the12-point local-layer Jacobian maximum. Observe growing P and sensitivity
at the larger Muon rate. Decision: test explicit growth feedback, without calling
these local measurements a whole-network stability certificate.

## 03-fitting-and-updates

Purpose: expose the direction-versus-magnitude confound. Source: curves.csv and
updates.csv. Caption must distinguish dense500-step validation logging from the
sparse registered optimizer diagnostics; step1 zero updates are omitted from
log-scale update plots. Dashed curves in the left panel are ordinary SwiGLU192
controls; right panels describe scalar-ODE P updates only. Observe the stable
large Muon magnitudes alongside increased update rank. Decision: include the
same-gradient norm-matched momentum control before claiming a causal spectral
flattening explanation.
