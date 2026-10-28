# Statistical and verification appendix

The [protocol](protocol.md) and source 86afd38 fix the 20k K1 checkpoint, data,
two replacements, calibration and criteria. Model seed 0 and the original
training trajectories are inherited. No training is performed in this study.

## Calibration and operator

Use target/source indices 0–63 from the original 1024-pair validation bank,
not Gaussian sampling starts. At j=0,...,128, form the original float32
interpolant at j/128 and capture every native token duration in each of six
local ODELayers. Each raw array is 129×64×64; all 3,170,304 durations are kept
privately. Average images/tokens in float64 then store 774 float32 table means.
Average each complete raw array in float64 then store its one float32 pooled
constant. Table and pooled control use the same native states and calendar.
The table is not calibrated on the replacement's generated path.

At off-mesh t, u=128t, a=floor(u), b=min(a+1,128), r=u-a. Use the unchanged
float32 expression `mu[a]+(mu[b]-mu[a])*r`. All 129 mesh nodes reproduce their
stored entries exactly, including t=1. Both replacements recompute downstream
states using the native K1 literal `h+q(t)Ph`. The six original local clocks
are bypassed; all parameters and runtime buffers remain byte-identical.

At fixed t, the local derivative is `I+q(t)P`. For the complete two-solve FFN,
`J=M2 diag(SiLU'(M1h)) M1`. At non-mesh times the explicit time derivative is
`q2'P2 z + M2 diag(SiLU'(M1h)) q1'P1h`, with z=SiLU(M1h).
The data-free local and complete state/time AD fixtures agree within 2.23e-16.
Ten actual promoted whole-network literal/dense witnesses have zero recorded
forward difference. This is algebraic verification, not a whole-sampler
stability or quality theorem.

## Evaluation and decision estimands

All 1024 original validation pairs contribute equally to weighted/clean MSE in
float64. The original weighted residual divides prediction-minus-target by
`delta=max(1-t,.05)` before squaring. Keep all five native 64-pair time panels;
no new coupling or panel replaces these quality banks. Sample every original
Gaussian start with unchanged float32 CPU Heun128 and its denominator cap.
There is no state clipping. FID uses the pinned Inception 2048 features,
original uint8 conversion, 1024 generated images and 2048 reference images.
It is not a FID50K result. Reused native/old-constant rows retain their prior
full-bank audits and seals. Mean and covariance FID terms and raw tails are
in [comparison.csv](results/comparison.csv); these are descriptive accounting
terms, not causal contributions.

Primary ordinary references remain Swi86/SiLU131. Strict ordering requires
FID below both. The allocation margin requires FID<=.95 times each reference
and weighted/clean MSE each <=1.05 times the best corresponding reference.
The added SiLU64 context cannot replace a harder reference. The full time-only
criterion also requires strict ordering over SiLU64 and a time/pooled FID
ratio <=.95 with both fitting ratios <=1.05. Strict ranking and the matched
pooled contrast pass; the allocation margin fails. The runtime TSV therefore
marks the time-table screen `discard`; the pooled row is retained only as a
control, never nominated. No threshold is selected from the observed scores.

## Counts and resource interpretation

Actual native constructor names distinguish 211094 learned parameters from
4096 fixed position values. Both source checkpoints still contain all those
values; 25350 learned clock parameters become inactive under each replacement.
The 185744 active learned count excludes 774/6 calibration constants and the
common fixed positions. Actual forward hooks measure 8192 FFN matrix MACs per
token/block for each replacement; pointwise lookup/activation/normalization,
attention and backbone work are not included. No backward saving is measured
or claimed. SiLU64 did not receive the native clock MLP training scaffold.

Both sampling children run concurrently, each one CPU thread. Their full 1024
sampling times are 139.4460 and 139.0018 seconds; recorded peak RSS is 489046016
and 481312768 bytes. Matrix work is lower, but no paired isolated native timing
benchmark is added. All four monitored controller phases complete with exit 0;
actual controller session 61529, curator session 8411. Sampling/audit-sampling/
metrics/audit-features phase times are 155.67/246.01/40.27/65.49 seconds.

## Readiness, independent checks and preservation

Two pre-quality attempts stopped before new generation: v1 included fixed
positions in a learned-parameter assertion; v2 imported a historical `study`
module after a parent changed its import path. Their sources, calibration,
terminal tool receipts and absence of formal output remain preserved under
binding hashes. V3 separates learned/fixed values and pins module identities;
no operator, data, calibration, precision, tolerance or decision rule changes.

Fresh documented preflights 69731 and 72055 exit 0. Native 1024 predictions and a
64-image Heun bank reproduce bitwise. All six native calendar arrays, 774/6
reductions and constant injection identities pass independent checks. The
calendar recomputes bitwise identically to the earlier pre-quality attempt.
The metric witness replays 32 historical MPS rows and native FID exactly.

The independent sampling audit explicitly recomputes native gate affine/
SiLU operations, clock lookup and Heun updates; both new 1024 prediction banks,
all ten time panels and 2048 new generated images reproduce bitwise.
Feature audit reproduces 2048 new and 2048 reference feature rows bitwise.
Independent SciPy gesvd recomputes all eleven FIDs within 5.69e-13 of the
formal values, below the unchanged 1e-8 criterion. Formal output hashes remain
unchanged. No failed score is replaced by a later selected result.

All raw arrays and exact process/resource/terminal receipts stay under
`/Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-time-clocks-v3`.
Parent runs, learned P histories, environments, source banks and submitted
Report1 remain unchanged. Public evidence consists of complete numeric tables,
four inspected figures and every first-eight-start example.
