# Cycle15: frozen gradient routes

Question: can the large fixed-readout normal residual directly drive hidden
P, and how much does it change the next hidden Adam update through clipping?
This is a diagnostic, not a candidate screen or training rescue.

## Fixed cohort and data

Use every unique clean family from cycles07/11/14: ode_k2, swiglu192,
frame_k2, global_k2, frame_k1, swiglu257, jet_c, affine_c, free_c, direct_b,
swiglu258. Two registered laws and steps0/1000/2500/5000 give88checkpoints.
No new seed, training, solver, data, head or evaluation change.

For each checkpoint restore the exact model and buffers. First replay all
20000 held-out field rows in native1024-row chunks, requiring bitwise
agreement with the saved prediction. Clone the saved data RNG and draw the
next native512-row training batch and capped paired label in float32. At
5000 this is batch5001, a defined unexecuted probe, not a historical update.
At other checkpoints it is the next historical batch. Promote these native
inputs, labels, parameters and buffers to float64 for the algebraic test.
All frames and clocks remain live in autograd.

## Measurements fixed before fitted outcomes

Construct B=W diag(gamma) C and T=ker([B,Q]^T) by SVD with relative cutoff
1e-10*max(sigma_max,1). Retain adaptive ranks, including zero-head rank126.
Compute actual squared-loss gradients and the three frozen-cotangent VJPs.
Save all flat gradient routes, parameter layout, projectors and batch inputs
privately. Publish groups: all, hidden, readout, P-only, FFN. Readout means
raw.output and final raw.norm only; block norms remain hidden.

Report loss energies; all three gradient norms and cross inner products;
closure errors; forced-hidden absolute norms; tangent/remaining-normal
cosine and cancellation ratio. Zero denominators are undefined, not zero.
Never compare absolute gradient size across different parameterizations as
an architecture capacity ranking.

Report full versus forced-removed global clip factors and the relative norm
and cosine of the promoted one-step Adam prediction using actual saved
moments. Verify weight_decay=0, parameter order, step counts and defaults.
Also save native float32 total gradient norm/clip as a numerical context.
No fitted optimizer.step is permitted.

## Verification, budget and decision

Preflight: independent null-space/zero-head/dense VJP fixtures, signed cross
terms, clipping equivalence, and float64 one-step Adam against PyTorch.
Commit source/protocol before fitted gradient outcomes. Prepare immutable
source/settings/input bindings; verify all prior runtime seals. One monitored
Mac CPU thread,600seconds for each formal/audit phase; no large GPU claim.
Lenovo SSH is currently unavailable. No MBP/torch execution or publishing relay.

Independent audit reconstructs the output-null intersection through QR of
a centered64D basis and Q, then differentiates projected squared losses
rather than the formal frozen-cotangent scalar. Replay all88native fields,
all gradient routes, cross-term accounting, saved moments and predicted
updates. Absolute/relative gradient tolerances1e-10/1e-8; projector1e-9.
No tolerance changes after fitted outcomes. Hash every formal artifact
before/after audit and all immutable inputs before/after execution.

The theorem is accepted only if the hidden forced-gradient norm is below
1e-10*max(1,||g_hidden||) at every probe. Otherwise stop and debug. There is
no architecture keep gate or significance test. Publish descriptive local
effects for every model/checkpoint, including null results. Do not select
best checkpoint/batch, diagnose full training causality, or reopen any stop.
