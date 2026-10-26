# A known sigmoid-GLU basis control

Registered before any GLU training. This is a generic baseline comparison,
not an ODE architecture, architecture nomination or completion of the project.
The mathematical/prior-art reviews permit one fixed control to resolve a
specific basis confound before any further probability-shaped ODE design.

## Question and its limits

For the independent Gaussian-source path, a fixed-time balanced two-component
Gaussian mixture has velocity `L*x+C*tanh(k*x)`. A rotated product population
has one such response per coordinate. An ordinary biased sigmoid GLU can
represent this raw-input fixed-time map constructively with2d channels. That
does not prove economy through the full normalized/time-conditioned backbone,
joint-time learnability, or a wide/deep SwiGLU limitation. The cached training
bank also has a different joint-atom posterior from this population oracle.

The saved-region analysis rejects an ambiguous-boundary-only motivation:
78.26% of the Product16 parent/Swi128 field-error gap is in confident
coordinates, and93.71% at t=.7. The narrow remaining question is whether the
generic sigmoid basis improves the learned affine-plus-signed-offset response,
including confident regions. It does not presume that learned gates are
posterior probabilities or that gate shape caused the old ODE gap.

## One change, three arms

```text
a,value = the unchanged fused affine input projection of h
SwiGLU: output_projection(SiLU(a)*value)
GLU:    output_projection(sigmoid(a)*value).
```

Keep all biases, independent projections and the nominal2/3 hidden convention.
Use SwiGLU96(actual64), SwiGLU128(actual85), and GLU128(actual85). GLU128 has
exactly the same parameters and dense MACs as SwiGLU128; pointwise work differs.
The new constructor must preserve every native tensor and RNG draw. Native
hidden functions differ, so neither identical input/output-head gradients nor
identical initial NTKs are claimed. A common nonzero-head probe must expose
that difference; a formal zero velocity head alone would mask it.

Nine cells: all three arms on Gaussian2D, Ring8 and Product16. Seed0,
5000updates,batch512,AdamW3e-4,weight decay0,clip1 and the original cached-bank
pair-velocity objective remain fixed. Keep uniform-time training, data
versions, five-time native field/Jacobian panels and Heun32 generation. Copy
the same hash-bound Stage15 step0 non-FFN state/buffers. Do not supply component
labels, analytic weights, rotation, posterior targets or teacher initialization.
The exact-representation weights belong only in a unit test.

Run two one-thread Mac CPU workers with actual10s monitoring and idle-sleep
prevention. Freeze a fresh source/config/protocol runtime. Both SwiGLU reruns
must reproduce their historical saved states; they are compatibility reruns,
not new independent repetitions. Preserve larger SwiGLU256's existing results
as a clearly higher-cost context, without relabeling it a matched comparator.

## Decision fixed before outcomes

A **generic baseline lead** requires at least10% lower primary validation
oracle MSE than both SwiGLUs on a nonlinear task, at most5% regression against
the better SwiGLU on every other primary task, and nonlinear Jacobian error
and validation SW2 each within10% of the better baseline. These are engineering
criteria. Gaussian-only or a selected panel/bin improvement is insufficient.

Report all endpoints/curves and every fixed time. On Product16 also report all
four previously fixed posterior-entropy bins and confident-mode errors; do
not select a favorable region after outcomes. A lead would need independent
evaluation before any broad generalization claim. It would strengthen the
generic baseline and raise the bar for subsequent ODE architectures, not
establish an ODE mechanism or meet the architecture goal by itself.

If this fixed gate substitution fails, stop it without centering, temperature,
slope, width, added-response, analytic initialization or output-head rescue.
Neither outcome reopens the stopped phi1, rotation or state-clock settings.

## Required verification

Before launch: exact tensor/RNG/parameter/dense-MAC pairing with native SwiGLU;
the fixed-time raw-input representation test and its scope; actual different
initial functions/head tangents; unchanged predecessor trajectories; full
trainer/checkpoint/resume/sampling; config propagation and a real monitor
heartbeat. After completion: source/data/config/reference hashes, every
checkpoint/diagnostic, snapshot data RNG, historical baseline states, and
native final field/Jacobian replay. No new architecture is accepted from a
unit test, this theorem, a successful generic control or a differential-equation
rewriting of it.
