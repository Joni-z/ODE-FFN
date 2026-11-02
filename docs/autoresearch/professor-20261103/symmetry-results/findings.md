# Frozen parity diagnostic: a concrete feature-direction lead

The registered12 checkpoints were read without retraining or changing primary
evaluation. Native predictions on the original validation bank reproduce the
recorded field errors within4.8e-10. Both analytic oracles pass central symmetry
within1e-10; the paired error decomposition closes within1.12e-16. Full hashes
and time profiles are retained. This is a diagnostic, not an odd-projected model.

The paired score averages error on x and -x. It is exactly the sum of the odd
prediction's error against the odd oracle and the even prediction's energy.
It differs from the original unpaired validation score and never replaces it.

| Task/model | Paired error | Odd-part error | Even energy | Even fraction |
|---|---:|---:|---:|---:|
|Product16 rawK1|0.023569|0.012049|0.011520|48.9%|
|Product16 scalarK1|0.019871|0.011929|0.007942|40.0%|
|Product16 Swi64|0.018750|0.011025|0.007725|41.2%|
|Product16 Swi128|0.016780|0.010300|0.006480|38.6%|
|Ring8 rawK1|0.011237|0.006029|0.005208|46.3%|
|Ring8 scalarK1|0.011863|0.006967|0.004896|41.3%|
|Ring8 Swi64|0.011784|0.007300|0.004484|38.1%|
|Ring8 Swi128|0.011970|0.007255|0.004715|39.4%|

On Product16, rawK1's excess even energy accounts algebraically for about74%
of its paired-error gap to Swi128, and79% of its gap to Swi64. Against scalarK1,
its odd error is nearly the same while its even energy is substantially larger.
Thus the deficit is not merely a less accurate desired odd response. This
identifies a feature-direction issue worth testing, after precision and coarse
integration failed to explain the earlier observations.

On Ring8, rawK1's odd-part error is lower than the shown ordinary controls,
while its even energy is larger. This is compatible with useful features mixed
with unwanted asymmetry. It does not prove that the raw diagonal causes the
asymmetry, or that it causes worse sampled distance. Other model components and
finite-bank training can also contribute. Even energy is appreciable in ordinary
controls too; it is not an ODE-specific failure mode.

A raw-versus-abs input diagonal is now a justified next controlled contrast:
h*h versus h*abs(h), with identical conditional generator singular values at
matched P,h. The latter is an odd feature, but the whole network is not forced
odd because SiLU, biases and time modulation remain. Retain an ordinary bilinear
FFN reference to distinguish a useful gated feature from an ODE-specific effect.
No symmetry projection, augmentation or extra loss is added. Real images do not
share this exact central-symmetry premise, so these results do not justify image
transfer or a general FM-specific claim.

[All12 rows](endpoints.csv), [time profiles](time-profiles.csv),
[145 receipt](145.json), [Lenovo receipt](lenovo.json), [figure](comparison.png),
[registered protocol](../symmetry-protocol.md).
