# Current project status

## Target-loss response is predictable; individual trajectories can be highly sensitive

The complete fresh-bank study retains four frozen20kmodels, both splits and all
64model-bank evaluations. The unfitted target-loss derivative predicts all8pooled
effect directions and beats zero-change prediction in each. Its particle endpoint
prediction is worse than zero displacement in7/8cases. A small tail of very large
sensitivities drives that difference. No new model was trained.

[Complete findings and six figures](docs/research-assessments/2026-11-16-ode-mechanism/target-transport-results/analysis-report.md),
[original numbers](docs/research-assessments/2026-11-16-ode-mechanism/target-transport-results/summary.json),
[execution/source/protection audit](docs/research-assessments/2026-11-16-ode-mechanism/target-transport-results/audit.json).

| Product16RK4weights, midpoint inference | Native SW2 | Crossed SW2 | Scalar score | Endpoint score |
|---|---:|---:|---:|---:|
| Fresh selection |0.053|0.048|0.994|-2.481|
| Fresh confirmation |0.047|0.044|0.870|-49.172|

These are within-model effects on new banks, not a comparison with ordinary
models measured on the older bank streams. Scalar score compares squared
prediction error with zero-change prediction; it is not explained variance.

All8native/eight independent evaluators, primary analysis and cached checks
terminate0. The original primary Boolean remainsfalse: four finite-difference
flags fail at epsilon1e-4. Complete72case/bank derivative checks agree with
independent Torch ADwithin1.09e-18; sorting changes explain those finite-step
discrepancies. The flags and original thresholds remain unchanged.

The exact remainder accounting reconstructs all288metric rows within4.45e-16.
For ProductRK4confirmation, one of32768sources contributes82.385%of tangent
squared norm: predicted displacement381.359versus actual1.801. All sources
remain in the analysis. Accurate aggregate response does not imply accurate
finite particle displacements.

## Next bounded check

Verify the maximum-, median- and original-probe-sensitive sources on all8cases.
Use four small perturbations and outer Heun127/128/129/256/512to distinguish
genuine local sensitivity from a time-resolution effect. The8cases/24roles
are already bound from original arrays; no new draw or model observation.
[Protocol](docs/research-assessments/2026-11-16-ode-mechanism/extreme-transport-protocol.md),
[fixed source bindings](docs/research-assessments/2026-11-16-ode-mechanism/extreme-transport-cases.json).
Then test whether the analytic midpoint/RK4defects predict the useful field/
distributional change. This ODE-specific link remains untested; no new image
architecture is promoted. Do not add particle accuracy to the old scalar gate.

## Execution and resources

The original Mac queue98117/15465, collector98379/82357, postprocessor332/64468
and cached-rank actor3315/10554are terminal. Full raw evidence remains under
/Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/.
Warm CPU04f11f75unchanged; no new formal trajectory study is running yet.

The complete-network ODE-defect observer passes independent algebra/JVP/FD
checks. CUDArecovery14772has actual direct exit0,5050source hashes unchanged
and GPU4context released; the prior receipt-wrapper failure/unknown child exit
is retained. CPU/CUDAparity2.23e-16primal/2.17e-19tangent. A4096-input reference
probe takes23.814sCPU/0.485sCUDAon145,48.694xincluding transfers. Use CUDAfor
large batches only after fresh availability checks.
[Observer derivation](docs/research-assessments/2026-11-16-ode-mechanism/hidden-defect-transport.md),
[complete CUDAreceipts](docs/research-assessments/2026-11-16-ode-mechanism/hidden-defect-cuda-readiness.json).

145physical0-3remain prohibited;4-7only<500MiBand no compute PID. Never
stop/share foreign jobs. No packages changed. Reports1/2,97protected files/
11final bindings and five native image files remain unchanged. Image benchmark
at160epochs stays native24.304/K1 27.789/K8 28.083FID50k.
