# Can time-only clocks retain the native K1 image lead?

Zhizhe Zhang · October 28, 2026. Register before new calibration/outcomes.

## Evidence and question

The native 20k K1 has FID 173.43691, versus 195.31010 for Swi86 and 185.88760 for
near-parameter SiLU131. The old simultaneous global collapse raises its FID
to 215.56371. The completed native feedback study finds 62.21% of K1 clock
variance between five external time points, with stronger 20k gate derivatives
but small typical full/reference norm increases. These are descriptive
observations, not proof that time dependence explains the quality lead.

Test one hypothesis: a fixed table of native per-time mean durations preserves
K1's ordinary-control lead after removing within-time image/token variation.
Include a pooled-clock control derived from exactly the same calibration
inputs/times, so changing the calibration cohort alone is not mistaken for
time preservation. This new control is never nominated as a rescued method.
The prior failed global intervention and all seven native endpoints remain
visible. No K2 intervention, retraining, new loss, temperature, order, spectral
constraint, alternative calibration, table resolution or solver is selected.

## Frozen sources and calibration

Restore the actual K1 `step_0020000.pt`, all learned tensors and nonpersistent
buffers. Bind the budget, 20k clock, native-feedback, original native17,
resolution18 and ordinary20 seals. Keep all sealed sources/runs read-only.

Use the first 64 original validation clean targets and their saved independent
Gaussian `source` arrays. These are not the Gaussian sampling/FID starts.
At each j=0,...,128, form float32 `x_j=(j/128)X+(1-j/128)e` and forward the
unchanged native model with external time j/128. This uses the clean images
only to construct the fixed noisy inputs; no prediction error, class label,
FID or generated output is used to fit or select anything. Record every
native duration in all six clocks:129×64×64 values per clock.

For each clock l:

```text
mu_l[j] = float32(mean_float64(s_l at time j/128 over images and tokens))
c_l     = float32(mean_float64(all 129×64×64 native durations))
```

The table has 129 time points because those are all evaluation nodes of the
unchanged Heun128 sampler, not a searched approximation resolution. For an
off-mesh time t, set u=128t, a=floor(u), b=min(a+1,128), r=u-a and use the
float32 expression `mu[a]+(mu[b]-mu[a])*r`. Require 0<=t<=1. At sampling nodes
the exact stored table entry is used, including t=1.

Two new fixed arms only:
- `ode_k1_time`: replace all six local durations by their external-time table;
- `ode_k1_pooled`: replace all six by the corresponding c_l from this same
  calibration. Its role is an attribution control, irrespective of quality.

Calibration is exposed validation data. There is no fresh-holdout claim and
no replacement of any native evaluation bank. Average native clocks along
the paired calibration inputs only, not along a new model's generated path.
Both interventions recompute every downstream state; calibration mismatch
remains a limitation if time preservation fails.

## Exact operator, derivative and cost

At fixed external t, every K1 local map is

```text
F_l(h,t) = h + q_l(t)P_l h
J_h F_l = I + q_l(t)P_l
```

The complete FFN is `M2(t) SiLU(M1(t)h)`. Its state derivative is
`M2 diag(SiLU'(M1h)) M1`. At off-mesh differentiable times its explicit time
derivative includes both q1' and q2' terms. These are identities, not FID
predictions. External AdaLN/time conditioning, attention and readout remain
native. Execute the literal float32 K1 recurrence, not a reordered dense fold.

Each intervention executes 8192 FFN matrix MACs per token/block versus 16512
for native K1. Stored learned weights remain 211094, plus the unchanged 4096 fixed position values. The six unused clock
MLPs contain 25350 parameters, leaving 185744 active learned values, plus 774
table constants or 6 pooled constants. SiLU64 provides the 8192-MAC/185744-
parameter ordinary context; also retain Swi86, SiLU131 and all other native
controls. Report measured counts, active versus stored values and sampling
time/RSS. Do not claim a measured training-cost reduction: no training occurs.

## Unchanged evaluation and decisions

Use every original 1024 validation pair, all five native 64-pair time panels,
all 1024 common Gaussian starts, the original literal CPU Heun128 update and
denominator cap .05. Keep the first eight starts for examples. Use unchanged
uint8 conversion, pinned Inception weights/MPS features and 2048 references.
Reuse the nine audited 20k endpoints/features from the completed clock study;
generate/extract only these two new arms. Report all eleven endpoints,
weighted/clean MSE, FID, mean/covariance terms, raw tails and native deltas.

For each new arm retain strict FID ordering and the same allocation margin:
FID below every declared reference, or at most .95 of each reference with both
fitting errors at most 1.05 of the best corresponding reference. The primary
ordinary reference set is Swi86/SiLU131, unchanged from native K1. Separately
report the added equal-inference/active-parameter SiLU64 comparison; it may not
replace a harder reference. Record continuous native/time/pooled differences.

Call the time-specific explanation supported in this finite screen only if:
1. the time-table arm retains the declared allocation margin and the SiLU64
   context does not reverse its strict FID ordering;
2. its FID is at most .95 of the same-calibration pooled control, with weighted
   and clean MSE no worse than 1.05 times the pooled control.

If both replacements preserve quality without this extra 5% time/pooled gap,
the data do not establish a time-specific advantage. If only the pooled arm
passes, record it as a control outcome, not a new constant-model acceptance.
If time-only fails, stop this one table/calibration setting; do not claim all
time-only or retrained clocks must fail. A passing frozen substitution still
does not establish a retrained architecture or a unique ODE mechanism.

## Readiness, monitoring and stopping

Use Mac mini. Lenovo reachability may be checked, but no migration, new
packages or GPU reservation is needed for these small frozen CPU models.
Warm specs remain image-training@475428fb and image-metrics@dc729a50.
At most two one-thread sampling children; each ceiling 1200 seconds, sampling
phase 1300, independent sampling audit 1300, metrics/feature audit 600 each.
Monitor actual PID/create-time and primed five-second CPU/RSS, including
children. Keep all new/private sources, raw arrays and failed attempts under
`/Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-time-clocks-v2`.
Estimated additional storage below 300 MB. No automatic retry or outcome-based
change of calibration, tolerance or acceptance criteria.

Before quality generation, a fresh agent executes the documented preflight
once: full native 1024 prediction and64-image Heun replay; unchanged buffers;
all native calendar arrays and both calibration reductions; constant-table
injection identity; state/time derivatives; literal/dense promoted witnesses;
and actual 8192 FFN MACs. A separate metric witness replays 32 previously audited
MPS feature rows bitwise and recomputes a historical FID before formal metrics.

Independent audit recaptures calibration with explicit affine/SiLU operations,
checks every table/pooled scalar and every random-time lookup, replays both
new 1024 predictions, ten time panels and 2048 generated images bitwise using
separate schedule/Heun code. Re-extract all new/reference feature rows
bitwise; recompute all eleven FIDs using independent SciPy gesvd within 1e-8.
Preserve formal output/source hashes. Stop after this cohort and its complete
audit, figures and archive. No layer selection or new training follows from
the outcomes without a new task-relevant protocol.
