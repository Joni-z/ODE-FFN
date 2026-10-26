# Shared-parameter response diagnosis after the generator screen

This is a read-only frozen-checkpoint diagnostic, declared before its kernel
outcomes. It does not train an architecture or reopen failed Stage15 arms.
The question is whether the trained controllers' *shared parameter responses*
can address the residual velocity field efficiently across inputs. Free gate
values at every point answer a different and much weaker question.

## Models, inputs and labels

Use Stage15's four scalar/vector multiplicative/additive arms and both SwiGLU
budgets, Ring8/product16, at1000 and5000updates. Restore actual runtime buffers
from the immutable checkpoints and verify frozen source, config, checkpoint,
dataset and diagnostic-array identities. Stage15's initial ODE/SwiGLU FiLM
tensors differ; record that control caveat rather than attributing every
cross-family difference to FFN type. A separate pairing control addresses it.

For each of the five existing diagnostic times, use saved common inputs with
target/source path indices0:16 for diagnostic calibration and256:272 for
diagnostic evaluation. The same16 paths recur across times; time panels are
not independent experimental repetitions. Inputs and population-oracle labels
come from the already exposed validation bank. This is neither a fresh
holdout nor ordinary pair-label training. No label is used to update a model.

## Exact parameter kernels

Perturb each solve's output separately. Its downstream Jacobian G includes
native recomputation of all later gates; freezing all gates would omit paths.
For each independent controller, with hidden input h, preactivation u,
a=SiLU(u), final matrix V, gate-output response R and slope C=diag(ds/dg),
set A=RC, T=AV diag(SiLU'(u)). Its exact cross-input parameter kernel is

```text
K_ij = (1+a_i dot a_j) A_i A_j^T
     + (1+h_i dot h_j) T_i T_j^T.
```

Sum six independent controllers, retaining all cross-time/cross-sample blocks.
The P response is G diag(s) for multiplication and .6G for addition; multiply
its response Gram by h_i dot h_j. Report controller, P and complete-FFN
kernels separately. Exact SwiGLU kernels include both w12 halves, their biases,
w3 and its bias. Their complete-FFN kernel is the generic reference; it is
not called a controller-only comparison.

Also compute a controller feature counterfactual dividing each solve's response
by the RMS of its receiving vector (Ph or h) at that input. This removes one
sample/layer amplitude factor and changes the shared fitting problem. It is
not a fixed parameter-coordinate change, pure direction control or causal
test of why training improved. Channel magnitudes, gate slopes and downstream
anisotropy remain. Report native response traces and pointwise free-gate
residual floors alongside this counterfactual.

## Fixed analysis, no selection of a winning lambda

For each kernel normalize by its calibration trace divided by calibration
output-coordinate count. This removes only one overall scale. Report all
ridge values1e-4,1e-3,1e-2,.1,1,10. Compute the calibration linear correction
and its response on the disjoint exposed evaluation anchors. Save every
residual curve and calibration spectrum; do not select one lambda to claim
model superiority. Native-kernel traces and component contributions preserve
amplitude information removed by trace normalization. Zero-trace kernels
are reported as zero responses without undefined ratios.

Report feature-update norms as well. They have a parameter-update meaning for
the native parameter kernels; the amplitude-equalized counterfactual has no
such direct nonlinear-network parameter interpretation. Check the ridge dual
against a direct primal shared-feature solution on a small independent case.

These corrections are privileged-label tangent-model predictions. They are
not finite nonlinear updates, optimizer convergence claims or achieved model
errors. A high-dimensional tangent may interpolate calibration labels yet
generalize poorly; that contrast is part of the diagnostic. A pointwise
response floor is conditional on weights/inputs, not a lower bound after
retraining. Fixed-response full-rank vector frames cannot demonstrate new
pointwise directions merely because Ph was replaced by h.

## Verification and decision

Before the real diagnostic, compare exact kernels against direct complete
parameter autograd on two-block scalar/vector mul/add and SwiGLU networks,
including downstream input-dependent gates. Test context restoration and the
zero-head limit. Preserve a frozen diagnostic source plus source/test hashes.
Run one CPU worker with progress and memory records; no GPU is reserved.

Use the outcome to reject explanations, not automatically choose a new design.
A poor final linear readout has already been excluded as the sole product16
explanation by an independent frozen-feature OLS check. If normalized shared
controller responses remain ineffective, another amplitude/receiver tweak
has no support. If the full FFN tangent is adequate but controller-only is
not, distinguish projection features and optimization before a new operator.
No training follows from this protocol without a separate evidence-based
design argument and matched control registration.
