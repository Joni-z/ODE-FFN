# Cycle17: a fixed unconditional CIFAR-10 baseline bridge

Registered before CIFAR quality outcomes. This tests whether the literal
professor block is competitive in a token/attention image backbone. It adds
no new architecture and does not rescue covariance, frame, independent-C,
head, solver, initialization or optimizer settings stopped on synthetic data.
All earlier negative results remain negative and mandatory context.

## Models and data

Four arms: ode_k1, ode_k2, swiglu86, swiglu129. Reuse cycle16's immutable
image_model.py and JiT snapshot:32x32RGB,4x4patches,64tokens,width64,depth3,
four attention heads,32Dpatch bottleneck, no dropout or in-context tokens.
All surrounding initial tensors/buffers match; both ODE orders have the same
initial parameters. The local ODE clock reads its input feature, not c.
The complete model retains its time/AdaLN routes and a constant class index.

Official CIFAR-10 archive MD5c58f30108f718f92721af3b95e74349a, verified
cycle16images and default_rng(0) split:45000train/5000validation. Class labels
are discarded; official test data are not decoded or used for selection.
Normalize uint8 pixels by image/127.5-1, no augmentation. No oracle field,
analytic tangent plane, teacher, extra loss or data filter is introduced.

## Training and compute

Model seed0 only. Use5000updates,batch64,AdamWlr1e-4,betas(.9,.99),wd0,
clip1,constant learning rate, no EMA or mixed precision. CPU float32 on
Mac mini; at most two one-thread children, with OMP/BLAS threads1.
CPU is a common backend with already verified bitwise model/moment resume;
two workers use CPU parallelism; their cohort speedup will be measured rather than assumed.
Record actual throughput rather than claim a full-model latency advantage.

Each arm has an independent CPU data Generator initialized100000, a fixed
stream offset, not an extra model seed. Each draw orders:64training indices,
standard-Gaussian source e, then64standard-Gaussian time logits z. Set
`t=sigmoid(.8*z-.8)`, `x_t=t*x+(1-t)*e`, `delta=max(1-t,.05)`.
The JiT output predicts clean pixels D(x_t,t); minimize mean((D-x)/delta)^2.
No unweighted-x-loss or direct-v-prediction arm is mixed into this comparison.

Save model/optimizer/data RNG/global RNG/all runtime buffers and input-stream
digest every500updates, including0/1000/2500/5000 (44checkpoints total). Save train windows and validation every500updates;
status every100updates; parent PID/CPU/RSS telemetry every5seconds. No
best-checkpoint selection: final5000is primary. Ceiling1800seconds per child;
if exceeded, stop/report the incomplete cell, no shortened endpoint claim.

Actual cycle16costs per token/block, dense FFN only:

| Arm | Whole trainable params | FFN params | FFN forward MACs |
|---|---:|---:|---:|
| ODE K1 |211094|16642|16512|
| ODE K2 |211094|16642|24704|
| Swi86 |211412|16748|16512|
| Swi129 |236438|25090|24768|

Swi86 covers ODE parameters and K1matrix work; Swi129covers K2matrix work.
Attention/backbone are identical. Pointwise work and optimizer overhead are
excluded; record full run time and isolated cycle16latency separately.

## Fixed validation and generation

Use the first1024validation indices with a separate fixed CPU Generator
initialized17101: Gaussian source first, then time logits with the same
logit-normal law. Save raw target/source/t/x_t before training. Native inference
chunks64. Primary metric is paired velocity-weighted validation MSE, plus
clean-pixel MSE. These are noisy-image regression metrics, not oracle field MSE.

For all five times .1/.3/.5/.7/.9, use the first64of these images and their
same sources. Save final predictions, paired losses and denoising examples.
Record each P's full spectrum/eigenvalues at every snapshot, as well as clock,
linear/quadratic term and local input/output RMS on these fixed panels. No
spectral constraint or post-outcome feature selection.

Generate1024images at step5000using CPU Generator17102standard-Gaussian
starts, no class input/CFG. Unchanged explicit Heun32on[0,1], velocity
`(D(x_t,t)-x_t)/max(1-t,.05)`,64-rowchunks. Save raw unclipped outputs.
Only display/feature conversion clamps to[-1,1] and rounds to uint8. Record
out-of-range fraction and finite checks; never clip sampler states.

## Inception FID: a small-sample diagnostic

Use torch-fidelity0.3.0's unmodified inception-v3-compat2048features and its
own pretrained weights, with versions/hash/cache fixed before quality runs.
One common MPS extraction backend,32-imagechunks, compared against CPU on a
fixed8-image fixture before launch (maxabs1e-3,RMS1e-4). The independent
metrics venv cannot modify the training/synthetic environments.

Reference: first2048validation images, raw uint8. Generated: all1024fixed
starts, identical count/order across arms. Report explicitly as **FID with
1024generated/2048validation references**, not FID-50K or a published CIFAR
score. Its bias and exposed validation reference prevent a final-test claim.

Compute the exact empirical-covariance FID in float64 through the nuclear
norm identity in theory.md. Verify against the upstream SciPy covariance
formula on well-conditioned small matrices, singular cases and identical
features. No fitted epsilon/feature change. Preserve raw features for audit.

## Preflight, audit and interpretation

Before launch: verify all source/data/prior seals; exact three-update native
trainer/data/buffer/optimizer resume for each arm; shared tensors; active P
and controller gradients; metric fixtures/weight identity; documented fresh-
agent environment witness required by the compute skill. Commit runnable
source/protocol before observing fitted CIFAR outcomes.

After training: independent CPU replay of all44checkpoint predictions and
buffers, all P/clock records, full5000-batch stream and saved RNG states;
manual Heun sampling replay; independent reference/generated features and
FID recomputation. Formal outputs are hashed before/after audit. No training
restart on an observation timeout and no sealed-source mutation.

A descriptive original-K2lead requires final weighted MSE<=.9times BOTH
SwiGLUs, clean MSE/FID<=1.10times the better ordinary reference, finite
training/samples. Higher-order support additionally needs MSE<=.95times K1
with no greater than10%FID/clean-MSE regression against K1. A passing
baseline is not a new method, synthetic transfer win or paper-ready claim.
If these gates fail, stop this fixed image setting without budget, width,
patch, loss, gate-range, seed, optimizer or solver rescue. No new architecture
or full project completion follows from this bridge alone.
