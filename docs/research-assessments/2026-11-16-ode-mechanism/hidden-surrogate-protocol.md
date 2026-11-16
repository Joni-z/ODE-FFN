# Native reference qualification before sampling interventions

Registered after the completed hidden-energy and one-layer output studies,
before observing this qualification's results. The question is numerical:
can a fixed native midpoint resolution reproduce the qualified DOP853-based
interventions closely enough to use during generation? This is not training
or a performance threshold for selecting a better FFN.

## Fixed cohort and intervention

Retain all six shared/PAL/ALT20kseed0checkpoints on Ring8/Product16, every
registered80input and all six separate one-layer interventions. Reuse the
original layer entrances, four-stage matrices, native outputs and two qualified
continuous references. Bind both prior studies' raw completion and array
hashes. Keep the existing network, data, stage boundaries and time embedding.

The five candidate resolutions are16/32/64/128/256midpoint substeps across
the same internal interval. At each level form the same reference/normal/
tangential/exact-potential-ray candidates as the previous study, substituting
the refined native endpoint for its continuous reference. Identity remains
native4. Do not select levels or layers according to generation scores.

## Numerical gates fixed before observation

For each individual point and every nonidentity mode, require

```text
||candidate_L - candidate_DOP|| <= 1e-9 + 1e-3*||native4 - DOP||
||field_L - field_DOP|| <= 1e-10 + 1e-3*||field_native - field_DOP_reference||.
```

The relative scale is the intervention being approximated, not the much
larger raw state or velocity. The absolute floors retain near-zero response
points instead of dividing by tiny values. These tolerances limit the
surrogate error to one part in1000of the relevant reference response, subject
to the stated floor; they are not significance or architecture-gain margins.

Also require the absolute change in each normal/tangential/ray reproduction
fraction F to be <=0.0005for every layer, each of eight input banks pooled
over layers, and the complete model. This is half the displayed three-decimal
unit. Retain zero-denominator cases as unresolved rather than dropping them.
Compute F using each intervention's own full-reference displacement, and
compare it with the completed DOP-based F on exactly the same inputs.

Replay the native identity prediction bit-for-bit. Independently reconstruct
every refined endpoint using NumPy and require each component to agree with
native Torch within1e-11absolute/relative. Ray energies must agree with their
respective reference within1e-11absolute/relative. Check unchanged parameters,
buffers, exact upstream entrances, hook removal and native roundtrip.

## Decision and cost

Report every level and failing point/layer/mode. Select the smallest single
resolution satisfying every gate across all six models and six layers.
No case-specific resolution, removed inputs or post-result tolerance change.
If none passes through256, no surrogate is qualified under this protocol.
Measure execution time and record the exact matrix-application multiplier
for replacing one fixed layer: (2*L+5*8)/(6*8). This is local computation work,
not measured end-to-end throughput. A qualified local surrogate does not
guarantee resolution on model-generated trajectories; a later sampling
protocol must separately check that.

Use two one-thread MacCPUworkers, one dataset each, in the unchanged warm
response-screen-cpu@04f11f75environment. A fresh documented witness executes
both datasets with all80inputs and first layer only, followed by independent
verification. Full execution retains all six layers. No packages, GPU,
training, image-code changes or finalized-report changes.
