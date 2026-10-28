# Methods and verification appendix

Zhizhe Zhang · October 28, 2026

**FINAL QUALITY PENDING.** Both training trajectories and the checkpoint audit
and all seven prescribed sample banks are complete. Independent sample replay
and feature/FID checks are still running. This appendix records the fixed methods; it is not a result report.

## Data and objective

The original CIFAR-10 training archive supplies 50,000 RGB images of size 32×32.
The saved split contains 45,000 training and 5,000 validation indices, with no
overlap. The official test batch is not used. Class labels are discarded;
the model receives label 0 for every example. No augmentation is applied.
Images are converted by `float32(image)/127.5-1`.

Each batch contains 64 images. A saved generator draws target indices, standard
Gaussian noise and `t=sigmoid(.8*z-.8)`, where z is standard normal. The input
is `x=t*target+(1-t)*noise`. The model predicts the clean target. Training
minimizes the squared residual after division by `max(1-t,.05)`. Weighted MSE
and clean MSE are separately evaluated on the same 1024 saved validation pairs
in float64. Five fixed 64-pair panels at t=.1,.3,.5,.7,.9 retain time detail.
These paired diagnostics do not decompose generation FID.

Model seed 0, native step 0 parameters/buffers and the saved seed 100000 data RNG
state are reused. Both controls run exactly 20,000 updates with AdamW rate 1e-4,
betas (.9,.99), epsilon 1e-8, weight decay 0 and global norm clipping at 1. No EMA,
AMP or learning-rate schedule is added. CPU arithmetic and one thread per arm
match the historical native controls. This is an unconditional image screen,
not the project's earlier synthetic-data experiment.

## What each comparison changes

The [protocol](protocol.md) fixes c=float32(.6) before outcomes. `fixed06` trains
the literal maps `h+cPh`; its unused clock MLPs stay frozen. `matched_silu64`
trains ordinary matrices initialized to `W=I+cP_initial`, rounded once to
float32. Every shared backbone parameter and buffer is identical initially.
The active promoted whole-model witness checks their function and gradient
map; matching zero-readout initial outputs alone would be insufficient.

The [derivation](theory.md) gives `grad_P=c*grad_W`. A constant-P core has the
same function class as ordinary square SiLU, but native Adam and joint clipping
depend on the chosen coordinates. With the same mapped clipping sequence,
equivalent W-coordinate Adam would use rate eta*c and epsilon/c on the six
core matrices. That transformed optimizer is only a verification fixture.
Neither formal arm changes the native optimizer. Their comparison therefore
concerns coordinates and finite-precision training, not a new function class.

`native_posthoc06` fixes the already-trained native K1 clocks to the same c at
20k. It is a frozen reference. Comparing it with `fixed06` changes all learned
parameters through training history, not P alone. Comparing matched-init SiLU
with default SiLU64 concerns initialization as well as the resulting training
history. These contrasts do not identify an exclusive causal pathway.

## Parameter and matrix costs

Counts exclude 4096 common fixed position values. MACs are per token/block,
measured by actual dispatch of matrix operations; they exclude pointwise
operations, attention, the remaining backbone and optimizer work.

| Training arm | Stored learned values | Active learned values | FFN forward MACs | FFN forward + backward MACs |
|---|---:|---:|---:|---:|
| Native K1 | 211094 | 211094 | 16512 | 49536 |
| Fixed 0.6 P | 211094 | 185744 | 8192 | 24576 |
| Matched-init SiLU64 | 185744 | 185744 | 8192 | 24576 |

The 25,350 unused gate values remain stored in `fixed06`, but receive no gradient
or update. Both new controls have the active count and FFN matrix work of
default SiLU64. Neither inherits adaptive-clock training. The posthoc native
reference does inherit that larger training cost. Lower FFN matrix work is
not a measured whole-model speedup. Training wall times are 1732.34s and 1632.28s
while the two arms share the Mac mini; no isolated timing claim follows.

## Generation and decision

The registered evaluator generates 1024 images per endpoint from the same
Gaussian starts using native float32 CPU Heun128 and its denominator cap.
There is no state clipping. Original uint8 conversion, pinned Inception/MPS
features and 2048 real reference images define FID. These are finite screen
FIDs, not FID50K or a fresh held-out benchmark. All seven native families at
5k/10k/20k and four earlier 20k frozen interventions remain in the comparison.

Each cold control has a 20k primary endpoint. It must beat both Swi86/SiLU131
strictly; the allocation margin additionally requires FID<=.95 of each and
both MSEs<=1.05 of the better corresponding reference. Strict ordering over
equal-cost default SiLU64 is also retained. The fixed 5k/10k results provide
context and cannot replace a failed 20k endpoint. No outcome establishes a
new ODE-specific architecture or selects another constant/optimizer setting.

## Verification completed so far

Fresh documented preflights 47808/32097 exit0, with physical model/Adam/RNG/
buffer resumes, nonzero active gradient transformations, measured costs and
historical metric replay. All 43 shared gradient tensors include the four
timestep-MLP tensors. The earlier passed preflight is preserved; v2 strengthens
that coverage without changing model, data, optimizer or budget.

Both formal training children and their parent phase exit0. The separate
operator/update audit also exits0: all 14 saved checkpoints reproduce validation
and time-panel predictions, matrix arrays and 42 native input-stream checks.
All 12 saved step-minus-one boundary updates reproduce full checkpoint state
bitwise. This is not a second execution of all 40,000 optimization steps.

Actual external pixel bytes, split and manifest match the native bindings;
their filesystem metadata predates the controller launch. Recursive native
external/metric checks also pass. The unused download tarball is not read by
the data loader. The final archive must recheck actual input bytes, source and
parent hashes, final numerical tables, fixed examples and actual tool exits.

Raw P, inferred `(W-I)/c` and effective W histories are distinct quantities.
The inferred P was not directly trained. Their singular values do not bound
the complete adaptive FFN, denoiser or sampler Jacobian, and they do not order
FID. All raw checkpoints and arrays stay outside Git; Report1 is unchanged.
