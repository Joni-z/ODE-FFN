# Figure catalog and interpretation audit

All figures use actual seed0 artifacts in `../stage-summary.json`; no confidence/error bars are shown. Axes differ across datasets, so visual slopes are not cross-dataset effect sizes. PNGs inspected for readable labels/legends and clipping; PDF companions provided.

|Figure|Purpose and variables|Observation|Decision implication / caveat|
|---|---|---|---|
|`../figures/gate-source.png`|Compare input/time gates at K1/K2: SW2 and gate Jacobian correction norm|Correction vanishes by design; SW2 response is mixed|Do not assume removing this term improves the trained full Jacobian or generation. Gate-input distribution also changes.|
|`../figures/order-and-budget.png`|Compare input-gated K1/2/4/8 with baseline endpoints matched by forward-linear MAC|ODE does not have consistent matched-budget superiority|Preserve declared endpoints, read alongside training curve; endpoints are correlated points from baseline long runs, not independent models.|
|`../figures/operator-mechanisms.png`|Plot exp-action relative error and local layer Jacobian versus K|Approximation error falls sharply; Jacobian generally rises then saturates|More accurate frozen linear flow is not automatically a more contractive learned layer. Each point uses a separately trained P.|
|`../figures/p-training-dynamics.png`|Plot mean sigma_max and stable rank over training for original input gates|Leading singular value rises, stable rank falls; K4/K8 overlap|Study learned directional selectivity rather than assume exploding series. Six-layer means describe one model; full spectra remain in JSON.|
|`../figures/training-and-nfe.png`|Top: all saved baseline samples from4000 updates onward; bottom: fixed ODE checkpoint acrossNFE|Baseline endpoint fluctuations; main K ordering persists at higher NFE|Budget claims require endpoint-sensitivity context. Top dotted ODE levels are fixed step5000 references, not ODE learning curves. No posthoc best-checkpoint selection.|

Captions must retain: seed0, data names, primary metric direction, fixed-vs-changing checkpoint distinction, six-layer/16-probe aggregation where used, MAC proxy limitation. No figure should be described as demonstrating significance, a global Lipschitz bound, or proof of mechanism.
