# Uniformly refine the learned observer's numerical reference

Registered after learned-defect-field-author-cpu.json, before new128/256-step
observer results. Original16/32/64arrays and failed gates remain unchanged.
All eight fixed16-input prefix cases completed actual0. Their whole-model
JVP, finite differences, two-tolerance DOP853primal reference and tensor/method/
RNGchecks pass. Four Ring8cells qualify. All four Product16cells fail only
the32->64tangent component gate: maximum differences1.099e-7to1.643e-7.
The differences decrease strongly under refinement; the method's learned-field
prediction has not been assessed on all4096inputs.

Keep the exact four weights/two streams, BOTH-solver forcing, complete network,
all inputs and all prediction criteria. Add referenceRK4counts128AND256for
EVERY case, including Ring8. Retain16/32/64.256is the prescribed final observer
reference; do not choose the best count using the measured prediction score.

Use the SAME component tolerances on128->256: primal1e-7*(1+abs(primal256)),
tangent1e-8+1e-5*abs(tangent256). Require aggregate differences to decrease
from64->128to128->256unless both<=1e-12. The final primal256must agree with
the original tight DOP853prefix within1e-7*(1+abs(reference)). Preserve the
original independent16-step derivative checks and every old failure flag.
If this refinement fails, retain the unresolved status and diagnose it.

For full4096-input diagnostics, compute all five reference counts, report both
the original32->64qualification and refined128->256qualification, and use
the fixed256tangent for the refined field-prediction test. The original test
does not become a pass retrospectively. The empirical hypothesis is unchanged:
an unfitted analytic correction must predict the actual coarse-solver field
change better than zero on BOTH ProductRK4streams. Better numerical integration
of the observer is a measurement change, not a new learned architecture.

Implement this as an explicit --refined-reference option with this document's
hash bound in the output. The old CLIdefault and old outputs remain intact.
Repeat the same16-input CPUwitness, then the exact independent documented
CUDA/NumPywitness before full-field execution. No new training or random draw.
