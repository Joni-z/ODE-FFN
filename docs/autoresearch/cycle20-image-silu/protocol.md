# Cold ordinary SiLU controls for the image ODE comparison

## Question and evidence boundary

Cycle19's image FID lead survives fixing all local durations. The resulting
local function is two matrices around SiLU. Test cold ordinary FFNs of that
form before claiming that the lead needs the ODE parameterization. This is
missing baseline coverage, not a new ODE candidate or an optimization rescue.
The earlier synthetic expanded-SiLU comparison remains task-dependent and
negative versus strong SwiGLUs. No failed ODE setting is reopened.

## Exactly three ordinary arms

Use `W2 SiLU(W1 h)`, bias-free, no local controller/dropout/extra conditioning.
The unchanged JiT retains attention and external time/AdaLN conditioning.
Three widths are derived from existing comparisons, not swept:

| Arm | Width | FFN parameters/MACs per token | Role |
|---|---:|---:|---|
| silu64 |64|8192|Same two-matrix shape as the frozen constant-clock core|
| silu131 |131|16768|Minimum integer covering original16642FFNparameters|
| silu193 |193|24704|Exactly originalK2denseFFNforward work|

Each block has128mweight parameters and128mforward denseMACs. Expected whole
trainable counts are185744/211472/235280; measure rather than trust them.
Measure actual isolated FFN forward+backward matrix work with TorchDispatch,
including input/weight gradients. Nonlinear/pointwise/optimizer work and the
shared attention/backbone are separate; no equal wall-time claim follows.
Silu131is a near parameter-covering control, with slightly greater K1matrix
work. Silu193matches K2matrix work but has more parameters. Retain both
existing SwiGLU controls rather than claim one comparison matches all costs.

## Cold initialization and immutable inputs

Image parent cycle17 seal
62d178c4e2b88780495c9ea45c6aeef96453ce7098a8aeee84076cf58f081172.
Resolution parent cycle18 seal
0a6fb24d1aba029be90e95f7c74919bbbdead973c101819852f04cb55cc64108.
Clock parent cycle19-v2 seal
c8ecd01c4c24ad6be6f3a8d1bf3aa7b5b3222f03712e4879e2cd5ecf15759d96.
All roots remain under `/Users/mr.z/ode-workspace-prep/autoresearch/`.
New runtime: `/Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle20`.
Reuse verified official CIFAR45000/5000split, all parent banks, features and
weights; official test data remain unused. No new dataset or metric.

All new models use modelseed0, the identical native cold non-FFN tensors
and runtime buffers. Reset torchseed0 before drawing new bias-free Linear
weights with default PyTorch initialization. Their widths/functions and FFN
initialization differ from the ODEs; a zero output head does not establish
nonzero-head functional equivalence. No fitted-weight or folded-ODE transfer.

## Training and recorded trajectories

Exactly5000updates, batch64, AdamW1e-4,betas(.9,.99),wd0,clip1,constantLR,
CPUfloat32, no EMA/AMP/augmentation/new loss. The separate data generator
starts100000; each batch draws trainindices, Gaussian source, then time logits
in native order. Keep `t=sigmoid(.8z-.8)`, image normalization/scale and the
weighted clean loss divided by max(1-t,.05). Global/model RNG and data RNG
are preserved separately. The data stream must match every native checkpoint.

Save model/optimizer/data/global RNG/buffers/digest, validation predictions
and metrics every500updates including0:33checkpoints. Keep all5timepanels
and both weight matrices' full singular spectra at0/1000/2500/5000:72matrix
records. Square64matrices also retain eigenvalues; rectangular matrices do
not have eigenvalues. Native P histories stay sealed and linked, not replaced.
Save clipping/loss windows and actual5sPID/CPU/RSS telemetry.

At most two one-thread CPU children; third queues.1800seconds per child,
3600seconds queue ceiling. Expected extra disk<1.5GB. Sample all1,024common
starts using the unchanged uniform Heun128,64-imagechunks and cappedvelocity.
No clipping of states; common uint8 conversion only. Preserve failures and
record actual exits. Never restart due to an observation timeout.

## Evaluation, audit and interpretation

Primary paired validation weighted/clean MSE uses the same1,024pairs; all5
64-image time panels remain visible. FID is1,024generated/2,048validation
references with the unchanged Inception/MPSbackend/weights/conversion. Reuse
four native and two constant-clock references explicitly under their seals.
Compute every new endpoint at5000, no validation-best selection.

Before training: verify sources/seals/common initialization, analytic isolated
SiLUvalues/derivatives, actual nonzero-head backbone gradients, measured costs,
physical three-update resume of model/moments/RNG/buffers and cold-Heun replay.
The new trainer entry point gets a fresh agent-follows-document preflight,
executing its documented invocation verbatim before formal training.
Warm existing environment specs475428fb/dc729a50and their seeded kernel/doc
receipts, with installed versions checked. No rebuild or package installation.

After training: independent replay of all33checkpoint predictions, all60new
time panels,72matrix histories, full5000batchstream and checkpoint RNG/moments;
manualHeun replay of all3,072generated outputs. Independently recompute all
new feature rows/reference and all9FIDvalues with a separate SVD driver.
Require bitwise predictions/samples/features, FIDdifference<1e-8. Audit phase
ceiling1800s, metrics/feature-audit600s each; no tolerance change after outcomes.

Report exact losses/FIDs/costs and contrasts for each declared comparison.
An ordinary arm is better on both quality axes only when both final paired
weighted MSE and FID are lower; report mixed rankings without a winner.
Cost scope still applies: no resource dominance claim across unmatched costs.
An ordinary success is an ordinary-control result, not a new ODE method. Its
failure does not isolate P optimization: initialization and training-time
conditioning differ. No pooled score, new nomination or altered cycle17gate.
Stop after these three arms. No width, bias, initialization, optimizer,
training-duration, seed, loss or solver rescue. Earlier negative studies and
the useful-ODE-successor objective remain unchanged.

After training, CPU checkpoint/sampling audit and MPS feature production can
run concurrently: the former protects only the immutable training-output
files. Feature audit starts after both terminal exits and then protects all
training and metric outputs. At most one CPU audit and one MPS extractor
overlap. This changes no scientific comparison or acceptance rule.
