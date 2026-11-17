# Preserve the analytic predictor while reducing its execution cost

The selected exact algebraic implementation passes every full-field numerical
comparison and runs4.475x faster than its paired eager control. It retains the
original equations, FP64,256reference steps and every input. This reduces the
primary observer-only projection to28.008GPUhours; the actual complete sampler
cost remains to be measured.

## Original compilation: purpose and observation

Compile one coupled RK4step while retaining256reference steps, FP64, both
solver defects, the complete model and all eight4,096-input field cases.
Compare every output with the saved original eager arrays. All cases pass;
the largest primal difference is6.661e-15and tangent difference1.249e-16.
Independent NumPy reconstruction reproduces every saved-array comparison.
The whole-network derivative, zero/sign, native replay and tensor/RNGcontrols
also pass. Six actual graphs are compiled without graph breaks or fallback.

| Paired repetition | Eager seconds | Compiled seconds |
|---|---:|---:|
|0|7.103|5.722|
|1|7.107|5.726|
|2|7.101|5.731|

These are synchronized resident timings on the fixed Product16 RK4
confirmation field, alternating execution order. The median ratio is1.240.
First-call timing includes tracing/compilation and is recorded separately;
it is not used to calculate the resident speedup. The frozenba0c426runtime
retains all5,200source hashes. Raw arrays and compiler diagnostics remain
outside Git; this folder contains their receipts and independent checks.

## Algebraic factorization: purpose and observation

Linearity permits grouping shared Papplications in the same coupled drift,
reducing19matrix actions to6. A second variant also reuses half-integer powers.
Both variants pass768prebound learned states, all six layers of each model
and both solver/zero-forcing directions. Maximum difference from the original
drift is8.327e-17; the independent polynomial-jet construction agrees within
1.339e-15. Actual isolated linear calls confirm19versus6, or2versus2when the
forcing is zero. This establishes the local algebra, not GPUthroughput.

Both variants now pass all8x4096original field comparisons, the independent
array reconstruction and fresh documented whole-network controls. All5,213
frozen source hashes remain unchanged. The fresh reader recorded39expected
bytecode files outside the archive manifest; those are runtime artifacts,
not changed frozen source. Both exact commands exited0without divergence.

| Execution | Paired eager median seconds | Execution median seconds | Speedup |
|---|---:|---:|---:|
| Original compilation |7.103|5.726|1.240|
| Linear factorization |7.013|3.651|1.921|
| Factorization and power reuse |6.886|1.539|4.475|

Each comparison uses its own three alternating, synchronized repetitions.
Select power reuse, whose largest full-field primal/tangent differences are
6.661e-15/1.457e-16. The selection concerns execution of the same analytic
predictor. It is not an FFNtraining or generation-quality gain. All timings,
independent checks and compact fresh receipts are retained beside this report.
