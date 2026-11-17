# Predict the target effect from the analytic hidden-ODE correction

Registered after the learned-field and outer-resolution results, before any
analytic-forcing target trajectory. These are retrospective mechanistic
predictions of a known intervention, not a new prospective quality comparison.

## Fixed scientific question

Does the unfitted analytic full-field correction b(x,t), derived from both
hidden solver defects, predict the target-distribution response to replacing
the native solver? Its field prediction already qualifies. The remaining
chain is b -> actual learned sampler -> empirical target loss.

Keep all four original20kseed0models, both existing evaluation streams,
eight original4,096-source/target banks and256directions per stream. Reuse
actual exported parameters and nonpersistent buffers. No new training/draw,
activation, selected time window, fitted scale or checkpoint selection.
Use FP64and the same256-step hidden reference. Execution factorization must
qualify against every original saved field before selection.

## Propagate analytic and measured forcing together

Let v0be the actual native field, v1the crossed field and delta=v1-v0.
Differentiate the actual native Heun computation, using an augmented state:

    x'  = v0(x,t)
    ua' = J_x v0(x,t) ua + b(x,t)
    um' = J_x v0(x,t) um + delta(x,t)
    ua(0)=um(0)=0.

The original sampler operates on this augmented state without modification.
It therefore propagates the discrete-Heun tangent, not an assumed exact
continuous outer solution. Two JVPs apply the SAME native Jacobian. The
analytic observer is evaluated separately, outside those JVPs; b is a direct
forcing at lambda0, so a derivative of b is not part of this first variation.
The native component must replay the cached original trajectory endpoint.

The measured-forcing tangent um is a simultaneous control. It diagnoses
whether a failure comes from the analytic field approximation or from
first-order propagation of a finite intervention. Do not replace b by delta
or fit either amplitude. Preserve all three paths and every input.

## Original banks and meshes

Compute both128and2048outer Heun steps for all eight model/stream cases.
The primary endpoint is Product16RK4weights at2048on BOTHstreams. Schedule
those16banks first, then all other registered controls. Each bank is a
separate monitored job; pool only after all eight original banks complete.
Cached native/crossed endpoints from population-resolution provide actual
intervention outcomes at each mesh, bound by file and array hashes.

## Numerical qualification

1. Before formal work, check the augmented sampler against independently
   differentiated deterministic analytic fields and central differences of
   the actual complete learned-model computation. Full-network prefix
   controls use the first2bank0sources of each original stream, Heun8,
   reference16for derivative-composition readiness only, both eps1e-4/1e-5
   and both forcings; formal observer reference remains256. Component
   tolerance2e-7+2e-4*abs(FD). Keep every failure. The short check validates
   the computational derivative, not final-mesh stability or target quality.
   Direct whole-sampler JVPis also checked on the analytic linear fixtures
   and measured learned forcing. The analytic learned observer itself uses
   forward AD; nesting that under a whole-sampler JVPis unsupported. Preserve
   the first failed attempt. Its independent end-to-end check uses the two
   prescribed finite differences; neither threshold nor forcing is changed.
2. For each formal bank, native endpoints must match cached native endpoints
   componentwise within1e-10+1e-9*abs(reference). Track the primal returned
   by each native-field JVP against direct evaluation at every call, within
   1e-12*(1+abs(reference)). Preserve tensors, methods and RNG.
3. Prebind eight equally spaced outer steps, including the initial step,
   and both Heun stages at each. Retain the first8sources at these16calls.
   Compare their analytic observer at128/256hidden-reference steps, using
   primal1e-7*(1+abs(reference)) and tangent1e-8+1e-5*abs(reference).
   These controls probe trajectory inputs; a failure is unresolved numerical
   evidence and must not be hidden by the earlier field-set qualification.
4. Reconstruct target derivatives independently from saved endpoints and
   tangents, including ties. Use the fixed squared SW2functional and original
   direction vectors. Preserve finite-difference metric checks and compare
   Torch/NumPy arithmetic with the original numerical contracts.

## Falsifiable primary prediction

For each mesh/case/stream, compute the exact first variation of the same
empirical squared sliced distance L at the native endpoints, using ua.
The primary score is

    score_a = 1 - (Lprime_a - (L_cross-L_native))^2
                    / (L_cross-L_native)^2.

Pass only if every associated numerical gate qualifies AND score_a>0on
BOTHprimary2048streams. Zero actual change gives undefined score, not a
pass. This compares with predicting zero target change, without fitted scale.
Report signed error, sign agreement and measured-forcing score_m separately.
Do not substitute SW2prime for the squared-loss change.

Secondary accounting: evaluate the exact target metric on x+uaand x+um,
with their separate endpoint-prediction errors and sorting/functional
remainders. Keep bank results and pooled results distinct. All controls and
both meshes remain visible regardless of the primary outcome. No fresh
ordinary-FFN ranking is claimed from this within-weight solver experiment.

If analytic target prediction qualifies, the next design question is which
derived correction is useful for generation at fixed compute. If only the
measured forcing qualifies, the analytic field approximation is insufficient
for target-sensitive directions. If neither qualifies, retain the measured
generation effect and reject this first-order target explanation.

## Resources and protected source

Warm Mac04f11f75for CPUcontrols/analysis. Warm14541450454for formal GPUwork;
physical0-3prohibited,4-7only if below500MiBand no compute PID. One worker
per eligible card, one CPUthread per worker, external compiler cache/raw
outputs. Re-poll original live handles; no restart on an observation timeout.
Publish/freeze code and obtain the required fresh documented execution before
formal launch. Final Reports1/2and the five native image files stay fixed.
