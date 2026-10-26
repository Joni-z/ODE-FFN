# Share computed projection features inside the original solve

Registered before this study's training. This tests one use of computation,
not a new loss or an FM-specific approximation theorem. The
[mathematical review](shared-feature-design-review.md) gives the exact family,
costs, local derivatives and limitations.

## Question and controlled operators

Keep the original K1 positive-clock map, both independent solves and middle
SiLU. Compare the existing vector32 controller with two width64 controllers:

```text
shared64: z=Ph; a=SiLU(z+b);   s=.2+.8sigmoid((Wg*a+bg)/10); F=h+s*z
untied64: z=Ph; a=SiLU(Rh+b);  s=.2+.8sigmoid((Wg*a+bg)/10); F=h+s*z.
```

Shared64 caches z once and uses the actual same P in both paths. Untied64
initializes its separate R from P, with equal b=0, Wg=0, bg=0. They have the
same complete initial functions/input Jacobians and initial FFN tangent kernel
when probed with a common nonzero velocity head. Untied R is initially
inactive, whereas shared P already supplies the value feature. Actual formal
zero-head initialization masks all backbone gradients initially; the probe
does not imply a first-update training benefit.

Shared64 costs two square matrix products per solve, untied64 three. Existing
vector32 also costs two squares' worth through its two narrower controller
matrices plus P. Shared64 therefore obtains64 gate features at the existing
matrix budget, with192 extra biases in the complete model. It doubles gate
hidden SiLU work; measure wall time. The untied family contains the shared
family. Sharing cannot add unconstrained tangent directions against untied64;
its two P-gradient routes introduce signed cross terms that can help or hurt.

This is feature sharing *within* a solve, distinct from sharing a clock
network or its output between solves. It retains the conditional operator DP.
Its K1 formula is also an ordinary residual feature-gated map. No renamed
copy of that formula is counted as an independent non-ODE control, and no
continuous-flow novelty or global stability claim follows from the notation.

## Six arms and common initialization

- vector_parent: existing hidden32 independent controller, constant-gate init.
- vector_shared: hidden64, shared_projection feature mode.
- vector_untied64: hidden64, paired_untied feature mode.
- swiglu_budget: nominal96, actual64 hidden units.
- swiglu: nominal128, actual85 hidden units.
- swiglu_large: nominal192, actual128 hidden units.

All six copy the exact non-FFN initial state and actual runtime buffers from
the hash-bound Stage15 vector_mul step0 reference. Their FFNs remain as
declared; no trained weights, labels or optimizer state are copied. ODE P
draws agree across the three ODE arms. Shared/untied feature biases are zero
and their final controller affines zero. All three ODE complete initial
functions agree, tested with nonzero heads. Shared versus untied has identical
feature inputs initially; the independent32 parent has its recorded narrower
feature basis, so no tangent-equivalence claim is made against that arm.

Preserve Gaussian2D, Ring8 and rotated product16 data versions; cached-bank
pair labels,seed0,5000updates,batch512,AdamW3e-4,wd0,clip1,width64/depth3.
Use original validation streams, five-time complete field/Jacobian diagnostics,
SW2 and Heun32. All18 cells are rerun together on the Mac CPU, two one-thread
workers with10s actual telemetry. No GPU is reserved. No additive receiver,
identity change, scale/temperature/range adjustment, order change or spectral
constraint is combined.

## Fixed decision

Shared64 is nominated only if it improves primary oracle MSE at least10%
against both established SwiGLU sizes on a nonlinear task and improves its
vector_parent there; stays within5% of the better established baseline on
every other primary task; and keeps full Jacobian error and SW2 within10%
of the better established baseline on both nonlinear tasks. This retains
the practical engineering screen, not a significance test. Show the larger
SwiGLU too; untied64's cost matches it and any usefulness claim for that
arm additionally requires the same comparisons against SwiGLU192.

Compare shared/untied directly to separate feature reuse from a wider private
controller. An improvement over the old parent alone is insufficient. Retain
every endpoint and full curve. No setting is extended if it fails; do not
rescue the axis by partial sharing, another width, temperature or Taylor-order
sweep. A genuinely comparable-quality speed lead must be confirmed in an
isolated benchmark before being called a practical Pareto gain; concurrent
training wall times alone do not establish it.

A passing screen only nominates a subsequent matched-budget confirmation,
fresh independent holdout and explicit mechanism analysis. All original goal
requirements remain. These calibration tasks and existing validation data
are not that independent holdout.

## Preflight and audits

Verify the cached formula, full input Jacobians, true Parameter sharing and
deduplication, P-gradient addition, initial kernel identity and signed
cross-term identity under nonzero test Wg. Confirm actual F.linear operations
agree with reported dense MACs, aliases survive checkpoint/dtype restoration,
and unsupported combinations fail explicitly. Unchanged defaults retain
the frozen predecessor's three-update trajectories. Record full-width float32
and float64 nonzero-head initial functions and surrounding tensor equality.

Freeze source/protocol/plans and verify hashes before launch. Require an
actual psutil monitor write and a complete trainer/checkpoint/resume check.
After training, audit every snapshot, initialization reference, data-generator
state and common-field array, then replay final predictions/Jacobians. The
earlier independent-controller tangent script must reject shared P/controller
layers rather than omit their parameter cross terms. Historical runtime
sources and submitted Report1 remain unchanged.
