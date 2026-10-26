# Measurement scope

This is a deterministic intervention on nine saved checkpoints: three tasks and three model arms. Source/data/weights are bound in the raw provenance. The full-validation uniform-time field MSE is primary; the five 512-input panels are a separate fixed-time diagnostic. Jacobian error is mean squared Frobenius distance to the oracle input Jacobian, averaged over all five panels. It is not the same normalization as elementwise field MSE.

The three arms are original scalar K2 ODE native-final, SwiGLU192 native-final and SwiGLU128 native-deep. The first two are nearly matched in matrix compute and share supervision; the latter is a strong practical comparator with a different supervision history. No comparison to the latter is called a one-factor causal estimate.

Uniform, radial and tangential fractions divide summed component energies by summed branch energy over all 2560 panel inputs per block. They are not a mean of individual fractions. RMS is computed from mean energy. The case was fixed by index, task, time and block; it was not selected for a large effect.

All raw intervention predictions and complete derivatives were replayed independently. Risk differences obey the per-input squared-error polarization identity; finite output change and error alignment are reported separately. No hypothesis significance claim is made from repeated times, blocks or derivative coordinates. The registered quality rule is in ../protocol.md; decision.json records its failure. No sampling, fresh test, training or tuning follows from this setting.
