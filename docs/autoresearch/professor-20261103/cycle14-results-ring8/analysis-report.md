# The Ring8 lead narrows under paired sampling checks

ISRU-D retains a conditional sampling lead over ordinary Free ISRU, but the
original 15.8% gap overstates its size on fresh banks. Against the stronger
validation-selected Swi64, the primary mean is nearly equal and the ordering
changes with checkpoint and sampling bank. This supports retaining ISRU-D for
a targeted explanation, not declaring an ODE-specific advantage or starting
image transfer. Product16 is still pending for a free GPU on 145.

## What the fixed comparison says

The primary endpoint was fixed at 20k updates and Heun 128 before this audit.
Every row below averages the same eight fresh source/target banks, with 4,096
samples and 256 unchanged directions per bank. AdamW 3e-4 Swi64 is the same-
optimizer/rate control; the Muon rows retain the previously selected ordinary
settings. They answer different fairness questions and are both retained.

| Model and FFN optimizer | SW2 mean | Bank SD | Native field MSE |
|---|---:|---:|---:|
| ISRU-D / adamw 0.0003 | 0.171104 | 0.047210 | 0.009773 |
| Free ISRU / adamw 0.0003 | 0.179305 | 0.047281 | 0.009998 |
| Swi64 / adamw 0.0003 | 0.194535 | 0.065031 | 0.013717 |
| Swi64 / muon 0.02 | 0.171453 | 0.053104 | 0.011279 |
| Swi192 / muon 0.02 | 0.183991 | 0.057236 | 0.011231 |

Across all three checkpoints, ISRU-D has lower native validation oracle MSE
than each of the four ordinary controls. This is a consistent fitting result
within these saved trajectories. Sampling does not preserve that ordering.

The candidate-minus-Free-ISRU paired difference is -0.008201, SD 0.009612,
range [-0.019372,+0.003736]; ISRU-D wins five of eight banks. The relative
mean reduction is 4.57%, versus 15.85% on the original archived test bank.
Against Muon Swi64 the paired mean is only -0.000349 (0.20%); six bank wins
coexist with a difference range [-0.011731,+0.016912]. These are fixed-model
evaluation observations, not a test of repeatability across training runs.

Increasing Heun 32 to 128 changes the ISRU-D mean from 0.170167 to 0.171104.
Its paired lead over Free ISRU remains about 0.0081–0.0082 and its mean
difference against selected Swi64 remains about 0.0003. The larger solver
resolution does not uncover a clear frontier gain. This checks the registered
resolutions; it is not proof of numerical convergence at all resolutions.

## Checkpoint and finite-sample dependence

| Model | SW2 at 10k | At 15k | At 20k |
|---|---:|---:|---:|
| ISRU-D / adamw 0.0003 | 0.184132 | 0.160535 | 0.171104 |
| Free ISRU / adamw 0.0003 | 0.185094 | 0.176845 | 0.179305 |
| Swi64 / adamw 0.0003 | 0.189830 | 0.166032 | 0.194535 |
| Swi64 / muon 0.02 | 0.190487 | 0.156681 | 0.171453 |
| Swi192 / muon 0.02 | 0.189339 | 0.161852 | 0.183991 |

At 15k, selected Swi64 has lower mean SW2 than ISRU-D (0.156681 versus
0.160535). ISRU-D field MSE improves from 0.010256 at 15k to 0.009773 at 20k,
while mean sampling discrepancy increases. The same 15k-to-20k sampling
increase occurs in every ordinary control. This is a shared fitting/generation
mismatch in these saved checkpoints; it is not evidence for an ODE-specific
failure. We do not choose 15k after looking at these scores.

Because the primary bank differences change sign, the registered decision
called for greater evaluation precision before new architecture. A separately
recorded post-outcome diagnostic concatenates the saved primary samples; it
adds no model calls. It uses 32,768 samples per model, the same formula and
directions, plus all eight leave-one-bank-out pools of 28,672 samples. This is
a different sample-size estimator and does not replace the primary endpoint.

| Model | Pooled SW2,32,768 samples |
|---|---:|
| ISRU-D / adamw 0.0003 | 0.132610 |
| Free ISRU / adamw 0.0003 | 0.144674 |
| Swi64 / adamw 0.0003 | 0.177708 |
| Swi64 / muon 0.02 | 0.136426 |
| Swi192 / muon 0.02 | 0.157686 |

The pooled gap versus Free ISRU is 8.34%; all eight omission pools retain
its sign, with candidate-minus-reference differences from -0.017949 to
-0.007284. The pooled gap versus selected Swi64 is 2.80%; seven omission
pools retain its sign and one reverses it (range-0.006445 to +0.000497).
Thus the retained positive result is stronger against Free ISRU and same-rate
AdamW Swi64 than against the selected ordinary frontier. The pooled target-
target discrepancy is0.033385, versus mean 0.152845 for the 4,096-sample banks.
This illustrates finite-sample error; neither floor is subtracted from scores.

The omission cases overlap and are dependent. Their signs describe sensitivity;
they are not eight independent confirmations or a confidence interval.

## Fairness, mechanism and measured work

The copied native definitions were reused throughout. All five final anchors
reproduce the archived generated/target arrays bitwise. Independent NumPy
recomputation matches all 240 saved SW2 values within 5.56e-17, and every sample
is finite. The complete pair table and figure retain all banks and checkpoints.
No target law, loss, direction bank, initialization or model weight was changed.

| Ring8 model | Trainable parameters | Executed dense MACs / vector-field call |
|---|---:|---:|
| ISRU-D | 56,706 | 55,616 |
| Free ISRU | 56,706 | 55,616 |
| Swi64 | 56,766 | 55,232 |
| Swi192 | 106,818 | 104,768 |

The matrix counts are verified by native Linear forward hooks plus the direct
P/ordinary-parameter multiplications, and agree with earlier Product16 counts.
They exclude pointwise/scalar arithmetic. Queue sampler timings overlap two
workers; they are not isolated latency, full FLOPs or fair training-time
benchmarks. [Cost receipt](cost.json) and [counter](../../../../studies/muon_pd_20261102/autoresearch_20261103/cycle14/cost.py).

Free ISRU differs in initialization and factorization. Its closeness limits
ODE attribution rather than proving the ODE design has no value. In this K1
setting the layer is exactly `F(h)=h+P q(h)`, with
`q(h)=h/sqrt(1+1.2h^2)`. This also describes an ordinary residual ISRU layer.
Its Jacobian `I+P diag(q'(h))` gives no standalone claim that P's spectrum
controls the full learned velocity field or that a Taylor mechanism caused the
observed gain. No new ODE-specific mechanism has been established.

## Research decision

1. Keep the modest Ring8 fixed-model sampling signal, including the weaker
   comparison against selected Swi64. Drop the 15.8% figure as a general effect
   size; preserve it only as the original-bank observation.
2. Complete the registered Product16 CUDA control when145 is free. The current
   result is explicitly partial; no transfer conclusion follows yet.
3. Before another layer, identify whether the remaining signal concerns error
   along transport states and its amplification. Use the analytic oracle and
   common states, with both Free ISRU and selected Swi64. A hypothesis that also
   explains the ordinary models is a feature/optimization hypothesis, not ODE
   attribution. This diagnostic is the next proposed mechanism check; it has
   not run and is not claimed as evidence.
4. No architecture, coefficient/rate rescue, longer training or image run is
   selected. Report1 and the submitted Report2 PDF/TeX remain unchanged.

The research question remains: which part of the tied flow construction makes
a prediction that a cost-matched ordinary feature map fails to explain? The
present audit improves the measurement of the candidate effect; it does not
yet answer that mechanism question. See the existing
[transport error derivation](../transport-bound.md): sampling depends on the
complete field error along transport states and its propagation; the bound
neither orders these empirical SW2 scores nor turns sampled Jacobians into a
certified global Lipschitz constant.
