# Comparison units and conditional bank effects

Each architecture/task has one fixed20k training trajectory, seed0. Selection
contains28rows (15k/20k); confirmation contains14rows (20k). Confirmation uses
different prebound evaluation arrays. No best-checkpoint selection is made.

Sampling uses eight common banks of4096points,256directions and native
Heun128. Pooled SW2 uses concatenated32768samples and differs from averaging
the eight bank scores. Compare model effects within a task; do not average
raw scores across Ring8/Product16.

For each common bank compute candidate minus reference SW2. The following
values are multiplied by1000 for readability. Negative favors the candidate.
Intervals are pointwise Student-t95%intervals (df7) for the conditional bank
mean, not intervals for pooled SW2 or variability of retraining. No multiple-
comparison correction or architecture-wide significance claim is made.

| Task | Candidate / reference | Mean delta x1000 | Conditional interval x1000 | Banks improved |
|---|---|---:|---|---:|
| Ring8 | Free Taylor / fixed K8 |-0.118|[-2.418,2.183]|4/8|
| Product16 | Free Taylor / fixed K8 |-1.252|[-1.857,-0.647]|8/8|
| Ring8 | State flow / fixed K8 |-10.523|[-14.987,-6.058]|8/8|
| Product16 | State flow / fixed K8 |-3.289|[-5.865,-0.714]|6/8|
| Ring8 | Free stages / fixed midpoint |0.559|[-0.468,1.586]|2/8|
| Product16 | Free stages / fixed midpoint |0.126|[-0.195,0.447]|3/8|

[All bank differences and exact values](results/paired-effects.json).
Each stage includes checkpoint/configuration/asset binding and independent
NumPy recomputation of both field MSE and generated-sample SW2. Those checks
validate the numbers, not a causal explanation of their ranking.

The C8-stratified follow-up uses4096independent base points and32768rotated
points per side. Its rows are dependent and its sampling protocol differs
from the iid common-bank comparison. Report its own paired intervention;
do not merge its absolute scores or infer an iid sample size32768.
