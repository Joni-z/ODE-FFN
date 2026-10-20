# Descriptive statistics appendix

Primary endpoint: SW2 at the predeclared checkpoint, Heun32, 2048 fixed generated and target points, 128 fixed directions; lower is better.
Unit: one trained trajectory per dataset/configuration; seed0 only. Baseline prefix duplicates are technical reruns. Checkpoints and layers are correlated measurements, not independent repetitions.
No across-seed SD, confidence intervals, p-values or multiple-comparison significance tests are estimable here. Effect sizes below are observed absolute/relative differences, not population estimates. No cross-dataset mean score is used.
Baseline budget matching is a full-network forward-linear-MAC proxy, not measured end-to-end cost. Endpoint fluctuation is reported without replacing the prespecified endpoint with the best checkpoint.

|Dataset|Contrast|Reference|Alternative|Difference|Relative change|
|---|---|---:|---:|---:|---:|
|Gaussian|time-minus-input-K1|0.061061|0.056772|-0.004288|-7.02%|
|Gaussian|time-minus-input-K2|0.067437|0.064254|-0.003182|-4.72%|
|Gaussian|K8-minus-K4|0.070309|0.070221|-0.000088|-0.12%|
|Gaussian|MAC-SwiGLU-minus-ODE-K1|0.061061|0.057765|-0.003296|-5.40%|
|Gaussian|MAC-SwiGLU-minus-ODE-K2|0.067437|0.065757|-0.001680|-2.49%|
|Gaussian|MAC-SwiGLU-minus-ODE-K4|0.070309|0.056350|-0.013959|-19.85%|
|Gaussian|MAC-SwiGLU-minus-ODE-K8|0.070221|0.054928|-0.015293|-21.78%|
|Ring8|time-minus-input-K1|0.200826|0.190919|-0.009907|-4.93%|
|Ring8|time-minus-input-K2|0.184820|0.188372|+0.003552|+1.92%|
|Ring8|K8-minus-K4|0.175176|0.174829|-0.000347|-0.20%|
|Ring8|MAC-SwiGLU-minus-ODE-K1|0.200826|0.208023|+0.007197|+3.58%|
|Ring8|MAC-SwiGLU-minus-ODE-K2|0.184820|0.150846|-0.033974|-18.38%|
|Ring8|MAC-SwiGLU-minus-ODE-K4|0.175176|0.149529|-0.025647|-14.64%|
|Ring8|MAC-SwiGLU-minus-ODE-K8|0.174829|0.198176|+0.023347|+13.35%|
|Moons|time-minus-input-K1|0.031507|0.034079|+0.002572|+8.16%|
|Moons|time-minus-input-K2|0.027083|0.026628|-0.000456|-1.68%|
|Moons|K8-minus-K4|0.027682|0.027664|-0.000017|-0.06%|
|Moons|MAC-SwiGLU-minus-ODE-K1|0.031507|0.024115|-0.007392|-23.46%|
|Moons|MAC-SwiGLU-minus-ODE-K2|0.027083|0.062414|+0.035331|+130.45%|
|Moons|MAC-SwiGLU-minus-ODE-K4|0.027682|0.035437|+0.007755|+28.02%|
|Moons|MAC-SwiGLU-minus-ODE-K8|0.027664|0.046784|+0.019119|+69.11%|

## P at initial and final checkpoint

Layer means are descriptive; stable rank is ||P||_F^2 / sigma_max(P)^2, not algebraic rank.
|Dataset|Model|Initial sigma|max sigma final|Initial stable rank|Final stable rank|Final mean s|Exp action difference|
|---|---|---:|---:|---:|---:|---:|---:|
|Gaussian|gate_k1_input|1.1088|1.8616|17.372|8.313|0.732|1.492e-01|
|Gaussian|gate_k2_input|1.1088|1.7411|17.372|9.159|0.684|3.159e-02|
|Gaussian|order_k4_input|1.1088|1.7152|17.372|9.301|0.672|8.123e-04|
|Gaussian|order_k8_input|1.1088|1.7150|17.372|9.305|0.672|1.330e-07|
|Ring8|gate_k1_input|1.1088|1.8083|17.372|9.190|0.796|2.007e-01|
|Ring8|gate_k2_input|1.1088|1.8364|17.372|8.860|0.731|5.310e-02|
|Ring8|order_k4_input|1.1088|1.8370|17.372|8.752|0.712|2.553e-03|
|Ring8|order_k8_input|1.1088|1.8395|17.372|8.719|0.710|1.822e-06|
|Moons|gate_k1_input|1.1088|1.8172|17.372|9.400|0.652|1.236e-01|
|Moons|gate_k2_input|1.1088|1.7868|17.372|9.139|0.607|2.385e-02|
|Moons|order_k4_input|1.1088|1.7529|17.372|9.301|0.596|5.077e-04|
|Moons|order_k8_input|1.1088|1.7526|17.372|9.300|0.596|6.163e-08|

Source summary SHA256: e5063a7d25a0a87a8bc51c85f92295ffd426f1e1fad6e61312de081ad210bdb5
All54 checkpoint hashes,29 frozen source/config hashes and training config/source identities audited remotely; all30 runs complete. Six baseline prefix weight comparisons are exactly zero.
