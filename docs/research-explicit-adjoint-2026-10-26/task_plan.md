# Explicit adjoint and clock contribution

Objective: use the verified A/B orientation disparity to identify what the original ODE's transposed computation contains, before choosing another architecture or cold-training setting.

Previous goal turn: progress. Primary literature and existing negative recurrence controls were rechecked; the review now separates internal depth, numerical updates and local FFN parameterization. The orientation experiment is also terminal and independently audited. This continuation first closes its report, then evaluates a mathematically exact implementation and bounded frozen-weight mechanism cases.

- [x] Inspect actual sources, completed runs, audit outputs and host/process state. Lenovo offline; no training/evaluation process live.
- [x] Close the orientation report and preserve its full saved fitting/P trajectories.
- [x] Derive and implement a complete primal/adjoint sweep for the unchanged ODE and SwiGLU networks.
- [x] Validate complete values, input/time derivatives, parameter gradients and actual optimizer updates against the original scalar-potential construction.
- [x] Evaluate bound saved checkpoints, record actual work and latency, and isolate the ODE gate-derivative contribution as a descriptive fixed-weight case study.
- [x] Inspect all outputs, write evidence and decide whether a distinct training hypothesis is justified.

Status: completed PASS on18 complete fields,90 panels and18 actual gradient/update cases. The observed ODE clock omission penalty supports a separately registered six-cell cold mechanism test; it does not select a practical architecture. Explicit inference is only about1.06x faster for ODE here, with no training-speed improvement.

No changed potential, interpolation coefficient, training seed, objective, clock, spectral bound, norm or optimizer. No fresh training is selected in this protocol. Explicit formulas must remain differentiable; no detached gate/input tangent. Preserve all previous frozen sources, runs and submitted Report1; no commit/push.
