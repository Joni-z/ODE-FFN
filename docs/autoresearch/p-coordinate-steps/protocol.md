# Nonzero-state coordinate identity on real image checkpoints

Register before execution. Zhizhe Zhang, October 29, 2026.

## Question
Does the exact direct-W optimizer transported from the fixed-clock P model
reproduce its next effective update at actual trained image states? How does
the unchanged direct-W Adam rule differ on that same transported state?
This checks implementation and scope of an existing identity. It does not
train a new model, explain a final FID difference or propose an optimizer.

## Fixed cohort
Use fixed06's saved immediately preceding states for updates 1000, 2500,
5000, 10000, 15000 and 20000. All six must be retained. Use each saved data RNG
to draw its exact next original 64-image training batch. Keep the checkpoint,
optimizer moments, backbone, buffers and data identity fixed. Read the stored
post checkpoint only for the native prediction bridge and immutable bindings.

## Three representations of each state
A: constant-clock professor core, c=float32(0.6), native AdamW settings.
B: ordinary bias-free SiLU with W=I+cP, shared weights unchanged; transport
the optimizer's first/second moments by mW=mP/c and vW=vP/c^2. Preserve every
parameter's step count. Use FFN rate c*eta, epsilon epsilon/c and the same
P-coordinate joint clipping factor; shared parameters keep native settings.
C: same direct-W function and transported starting moments, but the unchanged
native learning rate, epsilon and ordinary W-coordinate clipping norm.
C is a one-step counterfactual, not the historical ordinary training trajectory.

Use the already validated cast-free float64 RMSNorm/attention extension for
the coordinate identity. Keep native float32 time features and the original
float32 loss denominators, promoted as constants. First reproduce each native
post-state 64-image validation prefix bitwise, reproduce its literal float32
path bitwise and compare its promoted path within rtol1e-4/atol1e-5. This
separates the real-arithmetic identity from native reassociation roundoff.

## Recorded quantities and acceptance
For all six states, compare initial predictions, all active gradients, the
P and W joint gradient norms, clipping factors, all mapped next parameters,
Adam moments, six effective-matrix increments and post-update predictions.
Keep all A/B/C outcomes. B must match A within rtol1e-8/atol1e-10 for
parameters, gradients and predictions; record absolute errors as well.
Zero increments are reported explicitly, without dividing by zero.

Do not require C to differ by a preset performance margin. It uses another
optimizer-coordinate rule; record measured differences and clipping status.
The familiar 1/c Adam update ratio is only the negligible-epsilon, same-gradient
and same-clipping approximation, not a universal formula with differing clips.

## Limits and resources
No optimizer sweep, training continuation, new checkpoint selection, FID,
sampling, width change or initializer rescue. No claim of bitwise native
float32 coordinate equivalence or a causal generative mechanism. A failed
identity check preserves source/output and is investigated as a mismatch,
not corrected by changing the measured cohort or tolerance after the fact.

One one-thread CPU process, 300-second ceiling, expected output below 1 MB.
Warm image-training@475428fb; no package installation or GPU. The documented
check is executed exactly once by a fresh agent under the compute contract.
Retain actual tool exit and discrepancies before interpreting its output.
