# Preserve the analytic predictor while reducing its execution cost

The original compiled observer passes every full-field numerical comparison,
but its resident speedup is1.240x. This is insufficient to make the remaining
analytic target-transport calculation inexpensive. The profile identifies
matrix products as the main remaining cost, motivating an exact linear
factorization before a longer run.

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

The next execution checks both variants against all original full-field
outputs, with the same derivative controls and three paired timings. Only a
numerically qualified measured speed gain can select an execution backend.
No training, new evaluation draw or architecture-quality result is involved.
