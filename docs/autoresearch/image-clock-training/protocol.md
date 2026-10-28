# Training-time clocks versus ordinary parameterization

Zhizhe Zhang · October 28, 2026. Register before new training or frozen scores.

## Question

The 20k K1 time-only replacement keeps strict ordinary ordering but misses the
predeclared 5% margin: FID 180.86 versus 173.44 native and 200.05 for matched pooled
clocks. A frozen replacement inherits weights trained with adaptive clocks.
It cannot establish that a constant core trained from the beginning must fail.
The [derivation](theory.md) also identifies ordinary-function equivalence and
Adam-coordinate differences. Test these missing image controls before a new
ODE method; a result on this exposed image screen is not broad acceptance.

## Exactly two cold training controls and one frozen reference

Use c=float32(.2+.8*.5)=float32(.6), the midpoint of the professor's original
clock, fixed before outcomes. No calibration, learned constant or clock sweep.

1. `fixed06`: retain the native first-order maps `h+cPh` in both solves and
   middle SiLU. Start every P and shared learned tensor/buffer at the actual
   native K1 step0 checkpoint. Clock MLPs are unused and frozen unchanged.
2. `matched_silu64`: replace each FFN by ordinary bias-free 64→64→64 SiLU with
   `W1=I+cP1_initial` and `W2=I+cP2_initial`, rounded once to float32. Shared
   backbone weights/buffers are identical to `fixed06`. Initial functions agree
   in real arithmetic and within recorded float32 roundoff. Both use ordinary
   native AdamW settings; no coordinate-specific learning-rate adjustment.
3. `native_posthoc06`: at 20k only, replace the already-trained native K1 clocks by
   this same c. It is a frozen attribution reference, never a rescued candidate.
   Compare its inference map with `fixed06` trained under the same constant.

These controls do not introduce a novel function class. At fixed positive c,
`W=I+cP` is a bijection onto a free square linear map. `fixed06` versus
`matched_silu64` isolates their parameter coordinates at the same initialized
function; the comparison with the old default SiLU64 also changes initialization.
The native/cold/posthoc comparison changes the entire training history, not P
alone, and cannot identify an exclusive P-mediated causal effect.

The older failed frozen calendars, synthetic identity-location candidates and
submitted Report1 remain unchanged. No constant, width, identity-location,
loss, order, spectral penalty, optimizer or initialization grid is selected.
The matched SiLU model is an ordinary baseline, not an ODE rescue.

## Fixed data, initialization and training

Restore the original step0 model/data-generator/Torch RNG state. The historical
model seed is 0; data stream starts at its saved seed 100000 state. Retain actual
CIFAR-10 split, target/noise/t sampling, clean-image objective, denominator cap,
batch 64, AdamW 1e-4/(.9,.99)/eps1e-8, weight decay 0, clip norm 1, constant learning
rate, no EMA/AMP/augmentation and CPU one-thread arithmetic. Keep conditional
label 0, native time/AdaLN, attention, readout and all shared dimensions.

Both controls train exactly 20000 updates. Primary endpoint is 20k; 5k and 10k
generation are fixed context, not stopping/selection gates. There is no
outcome-based early stop. Save full model/optimizer/RNG/buffer checkpoints and
validation predictions at 0,1000,2500,5000,10000,15000,20000. Retain all five
native 64-image time panels and P/linear-matrix histories at these checkpoints.
For `matched_silu64`, inferred `(W-I)/c` is explicitly a coordinate conversion,
not a directly trained P. Raw P or singular-value uniformity is not a quality
objective or a representation-invariant stability statistic.

Every 1000 updates, compare the entire input-stream digest with the corresponding
sealed native K1trajectory. At each nonzero saved checkpoint, also save its
step-minus-one model/optimizer/RNG/digest as an independent next-update witness.
The original loss/optimizer/update functions are reused unchanged.

## Costs and ordinary controls

Verify actual FFN forward/backward matrix work with dispatch hooks. Both new
controls should execute 8192 forward/24576 forward-plus-backward matrix MACs per
token/block, excluding pointwise/backbone/optimizer work. Native K1 forward is
16512; its training cost is remeasured in the same fixture. `fixed06` retains
211094 native learned values but freezes 25350 unused gate values; 185744 are active.
`matched_silu64` directly stores 185744 learned values. Common 4096 fixed position
values are separate. Neither new cold model inherits fitted gate training.

Reuse all seven native families at 5k/10k/20k, including Swi86, Swi129 and
SiLU64/131/193, under their budget/native/resolution/ordinary seals. SiLU64 is
the exact active-parameter/matrix-work context. Keep the four old 20k global/
time/pooled clock interventions as frozen context. Do not drop a harder control.

## Quality banks and decisions

Generate 1024 images with unchanged native CPU Heun128 at 5k,10k,20k for both new
cold controls and at 20k for `native_posthoc06`: seven new banks. Keep the same
Gaussian starts, uint8 conversion, pinned Inception/MPS features and 2048 real
reference images. Evaluation uses all original 1024 pairs and five time panels.
First eight common starts provide fixed examples. No solver/state clipping or
alternative checkpoint. Report all 21 native budget rows, four previous frozen
20k rows and seven new rows: 32 FID endpoints, with costs and provenance.

At 20k for each cold control, retain strict ordinary ordering and the unchanged
allocation criterion: FID<=.95 of every Swi86/SiLU131 reference, weighted/clean MSE
each <=1.05 of the better corresponding reference. Require strict SiLU64 FID
ordering as additional matched-cost context. Record continuous ratios to all
controls and native K1; no new native-parity tolerance or significance test.

A passing cold constant control shows adaptive clocks are not required for
this finite allocation margin under that training setup. Failure stops only
this fixed constant/initialization/coordinate setting. `fixed06` versus
`native_posthoc06` measures different learned parameters at a common constant
inference architecture. `fixed06` versus `matched_silu64` concerns coordinates,
while `matched_silu64` versus default SiLU64 concerns initialization. Do not
attribute these differences to a larger function class or unique ODE physics.
No outcome nominates a new ODE-specific architecture or reopens the old failures.

## Verification and finite resources

Warm Mac mini image-training@475428fb/image-metrics@dc729a50. No installation,
GPU reservation for training, remote data transfer or MBP/torch dependency.
At most two one-thread training workers. Each training ceiling 7200 seconds,
cohort 7500; training audit 1200; seven-bank sampling 3600, independent sampling
audit 3600; metric and feature-audit 1200each. Expected new storage below 1 GiB.
Monitor actual PID/create-time plus primed five-second CPU/RSS. Stop on a
numerical failure or wall ceiling; preserve it and do not retry automatically.

Fresh documented preflight verifies baseline predictions, cold/shared tensors,
active promoted whole-model output/gradient transformation, local/FFN state
and parameter identities, the Adam/clipping equivalence witness, nonzero active
gradients, measured costs and three-step physical checkpoint/resume without
optimizer-state aliasing. Replay 32 cached metric features and a historical FID.

Training audit checks every saved checkpoint's identity/predictions/panels,
all native stream digests, frozen unused gates and every saved boundary update
using a separate operator/update implementation. This is a boundary/stream
audit, not a complete second execution of all 40000 optimization steps. The
unchanged native update kernel is source-bound and verified by the witnesses.
Sampling and feature audits independently replay every new full bank bitwise
and recompute all 32 FIDs with SciPy gesvd within 1e-8. State this verification
scope accurately; do not label the training audit a full trajectory replay.

Preserve source/input/parent hashes, all failed sources, checkpoints, P histories
and actual terminal exits. Runtime:
`/Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-clock-training`.
Stop after this registered cohort and complete report/archive. No learning-rate,
constant, schedule, width or training extension is selected after outcomes.
