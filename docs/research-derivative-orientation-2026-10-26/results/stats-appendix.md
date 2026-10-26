# Scope of the orientation evidence

Primary task-quality selection uses final full uniform-time validation oracle MSE. Jacobian panels use all512 inputs at each of .1,.3,.5,.7,.9. Sampling uses Heun32 with2048 common starts and targets; SW2 is recomputed independently in NumPy float64. Saved native float32 values/Jacobians/samples replay bitwise. Alternative complete-Jacobian constructions use float64 and do not depend on the new forward-AD path. Fixed-weight crossovers and their finite-change decompositions are descriptive, not cold-trained candidate results.

The independent audit checked36 new checkpoints,72 new P matrices,18 native field replays,90 native panel replays,36 complete crossover fields and180 crossover panels. Three actual training RNG/hash streams were regenerated. All291 previously bound parent inputs were reverified; parent reuse is not independent replication. Source and analyzer hashes are recorded in [audit.json](audit.json), with file bindings in [input-bindings.json](input-bindings.json).

All45 final B finite-difference probes passed, maximum relative error1.19e-9. Before launch,45 old A finite-difference probes passed, maximum relative error2.03e-9. Alternative formulas match complete values within9.22e-15 and input Jacobians within4.27e-14; the orientation identity error is at most1.19e-14. These checks cover the declared CPU environment and probes, not every PyTorch version/device. The LayerNorm failure was resolved before formal training; no affected formal B run was retained as a result.

Training uses seed0,5000updates,batch512,AdamW3e-4,weight decay0,gradient clipping1, common cached target banks and actual uniform-time pair-target streams. Model initialization, buffers and RNG match declared parent references. Two one-thread Mac mini CPU workers completed nine cells;39 ten-second telemetry samples recorded at least4.72GiB available memory and peak worker RSS375.1MiB. Lenovo was offline. No GPU was allocated to idle work.

[decision.json](decision.json) applies the preregistered practical criteria to native B only:10% nonlinear advantage over every generic control,5% own-parent gain, at most5% primary regression elsewhere, and nonlinear derivative/sampling guards. No arm qualifies. The old A setting remains stopped under its own report. Neither coefficient tuning nor longer training is authorized by a failed decision.

The36-layer adjoint identity check uses learned weights with deterministic algebra probes, not actual hidden task activations or new task-quality measurements. Its maximum adjoint error is1.78e-15. [adjoint-identity-checks.json](adjoint-identity-checks.json) binds those checkpoints and the verifier source.

Saved curves are taken directly from raw metrics; all30 final endpoints are checked against comparison.csv. P bands show min/max across six different matrices, not uncertainty intervals. The plotting source is independent of the audited model analyzer.
