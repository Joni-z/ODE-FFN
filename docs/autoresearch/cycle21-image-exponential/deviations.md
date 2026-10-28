# Preserved preflight failure

The first prepared runtime was oct28-cycle21, source registration f56509b.
A fresh agent executed the documented preflight once in the existing check
environment. Session 87896 exited 1 while importing the image model because
that environment lacks einops. No image-model predictions, operator history
or formal sampling outputs were produced. Analytic fixtures had run before
the failed import; they were not treated as environment readiness.

The failed source snapshot, exact log, agent receipt and terminal-failure
record remain under that original runtime. They were not overwritten.

V2 registration a2ad4d2 separates the work across existing environments:
the image interpreter performs actual model replay; the NumPy/SciPy check
interpreter reads checkpoint tensors and checks matrix algebra without
constructing image models. The sampling audit uses the same separation.
No package is installed and no model, matrix function, data, score, control,
sample count or numerical tolerance changes.

A different fresh agent executed the revised documented invocation exactly
once. Session 82275 exited 0, all nine runtime source files matched a2ad4d2,
and no documentation mismatch was found. The preflight checks all 48 operator
records, native and constant-clock predictions for both models, and ten
promoted whole-network polynomial witnesses. Readiness 90d20a5 was committed
before the formal sampling controller was launched.

V2 runtime: /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle21-v2.
Original failure: /Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle21.
Both use the same unchanged, sealed parent studies. The final analysis
distinguishes this environment correction from the scientific intervention.
