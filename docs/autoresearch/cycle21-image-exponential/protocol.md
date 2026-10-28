# Frozen image polynomial versus exponential

## Question and boundary

The old synthetic frozen-order study found that exact exponentiation worsens
all six low-order sampling comparisons. That evidence stops a generic
higher-order rescue. It does not measure the new image checkpoints, whose
constant-clock K2 has the lowest recorded FID. Check whether those learned
image maps are close to, or benefit from, the linear flow they approximate.
This is a finite frozen-checkpoint attribution, not a new architecture,
retraining proposal, new acceptance gate or reopening of stopped families.

## Exactly four new endpoints

Reuse both final native K1/K2 checkpoints and all twelve cycle19 calibrated
constant durations, without recalibration. At each of six solves, set A=cP.
Replace its local action by one cached matrix multiplication, using either:

```text
folded: T_K(A) = I + A                    (K1)
                 I + A + A*A/2            (K2)
exp:    E(A)   = matrix_exp(A)
```

Construct both matrices in float64 and cast once to float32. Both arms use
the same single-linear-expression implementation. Four endpoints are
K1-folded/K1-exp/K2-folded/K2-exp. The two folded arms measure changes from
the old literal float32 recurrence; the exp-versus-folded contrast changes
only the cached matrix function. No dynamic-clock exponentiation, polynomial
order grid, matrix rescaling, new constraints or trained exact-flow rescue.
All learned parameters, runtime buffers, SiLU, attention, external time and
readout remain unchanged. Cached matrices are separate untrained data.

Retain all nine cycle20 endpoints as explicitly reused references. Their
existing costs stay labeled as such. This study is not an optimization or
cost-matched training comparison. Do not infer runtime savings from folding.

## Fixed inputs and computation

Use existing official CIFAR split and unchanged prediction banks, capped
velocity, Gaussian starts, Heun128, 1,024 generated and 2,048 reference images,
Inception weights/MPS backend and uint8 conversion. Test data stay unused.
No optimizer, new seed or additional training. Read all parents under seals:

| Parent | Runtime | Seal SHA256 |
|---|---|---|
| Native images | oct28-cycle17-v2 | 62d178c4e2b88780495c9ea45c6aeef96453ce7098a8aeee84076cf58f081172 |
| Resolution | oct28-cycle18 | 0a6fb24d1aba029be90e95f7c74919bbbdead973c101819852f04cb55cc64108 |
| Constant clocks | oct28-cycle19-v2 | c8ecd01c4c24ad6be6f3a8d1bf3aa7b5b3222f03712e4879e2cd5ecf15759d96 |
| Ordinary controls | oct28-cycle20 | d7fe17e92c7bd8468b6c048aa9925a4350d95168ed611682bb9c7d787634e973 |

All runtimes are under /Users/mr.z/ode-workspace-prep/autoresearch/. New root:
oct28-cycle21. At most two one-thread CPU sampling children; 900s each and
1,800s for the queue. Save five-second PID/create-time/CPU/RSS telemetry and
terminal exits; never restart because an observation expires. Expected extra
disk below0.3GB. Feature extraction uses the existing metrics environment;
no package installation or GPU reservation. Mac mini/Lenovo only.

## Operator history and preflight

At steps0/1000/2500/5000, inspect all six P matrices for both native models:
48records. Use final calibrated c for every historical record, explicitly a
fixed retrospective slice, not a recalibrated training trajectory. Save full
P singular/eigenvalue distributions, numerical abscissa, nonnormality, c||P||,
map norms, polynomial/exponential relative error and semigroup defects.
No scalar spectral statistic is a generation-performance certificate.

Before formal predictions: verify every binding/seal/data/weight; compare
matrix_exp with independent SciPy expm (rtol/atol1e-11); verify zero, negative
diagonal and nonnormal nilpotent fixtures and analytic semigroup defects;
replay original native/constant validation predictions bitwise. A promoted
whole-network polynomial identity checks the matched expression at all five
fixed time inputs (maxabs1e-9). Save cached float64 and float32 matrices and
hash-bind them. Run a fresh documented-command preflight before quality use.

## Outputs and independent checks

Save all four validation predictions, twenty time panels and4,096generated
images. Record exp-versus-folded and folded-versus-literal prediction/sample
RMS changes, actual fitting/FIDs and raw out-of-range fractions. Preserve all
thirteen endpoints in the final table, without selecting the best variant.

Independent sampling audit checks SciPy-versus-Torch exponentials and all48
operator records, then replays predictions/panels/samples using the registered
cached matrices and a separate Heun expression. Require bitwise arrays. This
is independently checked matrix construction plus cached-map replay, not a
claim that SciPy and Torch exponentials round bitwise identically. Verify all
parameters/buffers unchanged. Audit ceiling1,800s.

Fresh feature audit re-extracts4,096new generated and2,048reference feature
rows bitwise. Prior nine banks retain their seals/audits. Independently check
all thirteen FIDs with SciPy gesvd; absolute difference<1e-8. Metrics and
feature-audit ceilings900s each. Keep all failures and tolerances. Curation
requires every phase to pass. No new ODE architecture is nominated regardless
of outcome. If exp helps, that is frozen intervention evidence; if it hurts,
that concerns these weights, not models trained with exact flows.
