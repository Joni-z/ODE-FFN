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

## Rare-source check complete; full-bank resolution is next

All8cases/24prebound roles complete in140.223s, native/independent exits0/0.
All24pass endpoint derivative checks at epsilon1e-5and1e-6. ProductRK4
confirmation maximum tangent changes381.359at128to1.390at129and2.025at512;
its median control stays0.028. The exact sources were selected at128, so this
is source-local mesh dependence, not a population quality/sensitivity ranking.
[Complete findings and two figures](docs/research-assessments/2026-11-16-ode-mechanism/extreme-transport-results/analysis-report.md),
[source/protection audit](docs/research-assessments/2026-11-16-ode-mechanism/extreme-transport-results/audit.json).

Next compare full-bank native/crossed generation at Heun127/129/512/2048,
with cached128as replay anchor, on the same four frozen weights/two streams.
Export actual parameters and nonpersistent buffers before CUDAportability checks.
The analytic ODE-defect-to-target prediction remains untested. No new image
architecture is promoted; the original target-derivative gate stays unchanged.

## Execution and resources

The original Mac queue98117/15465, collector98379/82357, postprocessor332/64468
and cached-rank actor3315/10554are terminal. Full raw evidence remains under
/Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/.
Warm CPU04f11f75unchanged; the24-role probe queue34920is terminal, actual0.
Source146c158/5090hashes and all original inputs remain unchanged.

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
