# Statistical and numerical scope

Six fixed endpoints: four sealed native controls and two interventions of
existing models. No training or new seed is added. The intervention unit is
an entire frozen model, with all six clocks changed together. Layers, pixels,
images and fixed-time panels are not independent trained repetitions. No
p-values or confidence intervals over model reruns are claimed.

Constants use the first 64 exposed validation image/noise/time inputs, with
4,096 native scalar durations per clock. Calibration is not target/FID fitting,
but it is not a new holdout. All original 1,024 paired validation examples
remain in evaluation. Thirty 64-image/time model panels are retained; only
ten belong to the two new interventions, while twenty are reused native
predictions. Primary weighted MSE uses the original random-time bank.

K1 changes: weighted MSE +0.060378%, clean MSE +0.03039%, FID -1.216512.
K2 changes: weighted MSE -0.039084%, clean MSE +0.01752%, FID -3.980203.
The retained FID gap to the better ordinary control is 1.1830 times the native
gap for K1 and 1.4784 for K2. These ratios describe the fixed bank, not an
estimator of a population effect or a newly selected acceptance threshold.
Cycle17's failed fitting/order gates are untouched.

All FIDs use the same 1,024 generated / 2,048 reference counts, common Heun128
starts, stock compatibility Inception weights and uint8 conversion. Small-
sample bias remains. Native control raw fractions in the original producer
JSON were copied from the wrong mesh; the separate correction changes no
score or prediction. Published fractions are independently recomputed from
each actual 128-step raw bank.

The registered scalar/matrix identity is tested in float64, including AD;
maximum data-free discrepancy is 2.22e-16. Native time features are lifted
only in the promoted witness to match its MLP dtype. Ten complete-model
folding outputs agree exactly against a maxabs1e-9 gate. Formal float32
quality arrays reproduce bitwise, not just within that witness tolerance.

New feature rows and reference rows reproduce bitwise. Four native banks
are reused under their existing full audit and immutable hashes. All six
scores independently agree within 5.41e-13 against the unchanged 1e-8 limit.
No metric, feature, tolerance, calibration subset or mesh is tuned after
intervention outcomes. Better FID alongside larger out-of-range tails is
retained; it is not replaced by a stronger claim about diversity or fidelity.
