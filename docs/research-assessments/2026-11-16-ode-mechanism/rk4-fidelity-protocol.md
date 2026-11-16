# Measuring the trained hidden solver faithfully

This implements the hidden-fidelity outcome already fixed in the matched RK4
training protocol, before either new 20k endpoint is observed. It changes no
training, sampling or model code. Four cases: midpoint4 and RK4two-step at
20k on Ring8 and Product16, seed 0. Keep all six local solves and the same
80 interpolation inputs (two source/target pairs per original bank, eight
banks, physical times .1/.3/.5/.7/.9). Capture each model at its own entrances.

## Primary quantity and reference qualification

For every local solve retain native output y_num and continuous y_ref of the
same learned f(y)=P ISRU(y). Primary rho is the sum of squared native/reference
errors divided by the sum of squared continuous displacements y_ref-h, pooled
across all six layers and 80 inputs. Also report absolute numerator/denominator,
per-layer results and all eight bank values. Zero-displacement and unresolved
points remain explicit, never silently filtered.

Use the original piecewise DOP853 reference and unchanged tolerance pairs
(1e-10,1e-11)/(1e-12,1e-13), including its energy-work quadrature. Every point
must meet the existing componentwise state agreement 1e-9+1e-9*abs(reference),
work agreement 1e-8*(1+abs(work)), and energy closure 1e-8*(1+abs(initial energy)
+abs(final energy)+abs(work)). A separate verifier uses state-only RK45 for
EVERY individual input, rtol1e-11/atol1e-13, requiring agreement within
2e-8+2e-8*abs(reference). These are numerical checks, not effect thresholds.
All records and gate failures are retained. An unresolved case cannot support
H_fidelity until resolved under a separately recorded numerical extension.

Before reference integration, reconstruct actual native outputs with independent
NumPy algebra at every captured state. Require 1e-11 absolute/relative agreement.
Recreate the saved FP32 field on128inputs before converting to double, using
the earlier fixed 2e-5/2e-5 replay and3e-5/3e-5 conversion checks. Verify source,
checkpoint/config/data/evaluation hashes and unchanged parameters/buffers.

The primary solve uses native4midpoint or native2RK4steps. Retain doubled and
quadrupled step counts only as refinement diagnostics; do not choose the best
resolution for a quality comparison or replace the registered native endpoint.

## Evaluation and decision

Use the registered paired-bank differences and four-comparison-adjusted intervals
for two tasks by quality/fidelity. Resample original banks jointly, keeping
layers/times within each bank together. Report pooled rho and bank effects
separately. Check targetSW2using the unchanged common-bank evaluator; hidden
fidelity is not a target-quality metric. Keep ordinary SwiGLU/FreeISRUcontrols.

Witness mode uses eight inputs at time .5 and allows the existing16-step RK4
preflight checkpoints. It verifies restoration/algebra/references only and must
never be mixed with formal20kresults. Both solver families and both datasets
must be exercised before freezing the four formal observer invocations.
