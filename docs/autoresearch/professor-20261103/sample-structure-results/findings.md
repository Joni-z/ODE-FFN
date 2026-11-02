# Low aggregate distance can coexist with a blurred mixture coordinate

The registered CPU-only audit read all74 frozen Product16 cells. Checkpoint,
config and manifest bindings pass; shared targets and original replayed scores
match. Three technical anchors duplicate earlier generated arrays, leaving71
distinct arrays. No sample was regenerated and no model/checkpoint was selected
using the new readouts.

## The concrete discrepancy

| Fixed case | Primary random SW2 | Latent-axis SW2 | Central-gap mass, mean over16 axes |
|---|---:|---:|---:|
|Scalar K2 Muon.02|0.08810|0.33536|1.63%|
|Raw K1 Muon.006|0.11114|0.39552|1.92%|
|Abs K1 Muon.02|0.08981|0.34883|2.15%|
|Swi128 AdamW.0003|0.08185|0.30472|2.02%|
|Swi192 AdamW.0003|0.07921|0.31262|1.98%|
|Constant ordinary Muon.02|0.07552|0.30530|1.50%|
|Bilinear64 AdamW.0003|0.07054|0.24702|9.06%|
|Saved target reference|—|0.23615|0.51%|

The population central-gap mass is0.52%, with |z_j|<a_j/2 fixed before evaluation.
The bilinear case's favorable aggregate distances do not mean that every
two-mode marginal is accurate. On the preselected smallest-separation axis0,
it puts32.45% of samples in the gap where the population puts3.04%; the two
narrow peaks become a broad bridge. Swi192 has8.13% and rawK1 has6.67% there.
On axis15 all shown models reproduce the widely separated modes much better.
The [ordinary-case plots](ordinary-cases.png) and [ODE-case plots](ode-cases.png)
show the complete registered0/7/15 panels, including those favorable cases.

The bilinear mean near-mode squared distance is0.11515 versus saved targets'
0.04054 and Swi192's0.04773. This is a shape diagnostic, not a new universal
metric or a selection rule. Raw and ordinary models also retain excess gap mass;
the audit does not establish perfect convergence for any model.

## The latent-axis average is not a replacement ground truth

The bilinear case is also lowest on the new mean axis SW2. Thus merely changing
projection directions does not resolve the ambiguity. Averaging can hide an
individual coordinate defect, and separated mixtures have substantial finite-
sample Wasserstein effects. Even independent target samples have axis SW2
0.23615 here; this is a reference value, not a lower bound for every model.

A limiting example makes the count sensitivity precise. For two point-mass
mixtures at +/-a with positive-mode probabilities p and q,

```
W2^2 = 4*a^2*abs(p-q).
```

Only the unmatched mode mass must move the distance2a. Two independent finite
samples can therefore have positive distance even with exact atom locations.
For balanced atoms, the variance of their probability difference is1/(2N).
The example explains why mode counts matter; it is not an asserted exact
decomposition or asymptotic rate for our continuous Gaussian mixture.

## Consequence for the research loop

Keep original field MSE and random SW2 unchanged, and show actual marginal
examples beside them. A low aggregate score alone is insufficient to nominate
the bilinear case as a strong generator, while a large field error alone is
insufficient to diagnose its terminal samples. These are complementary checks,
not an outcome-driven replacement metric.

The next controlled question is budget/convergence of the existing rawK1 lead
against matched Swi64 and strongest tested Swi192. Their rates are chosen by
the already frozen validation results, not these new sample metrics. No new D
shape is selected. The earlier20k ordinary curves still improve: Product16
Swi192 drops0.01659->0.01567 from15k to20k, and Ring8 Swi64 drops0.01366->0.01128.
A longer matched-budget comparison can test whether raw's small Ring8 lead is
transient. It must retain ordinary controls and preserve the stopped abs setting.

[All74 endpoints](endpoints.csv), [axis readouts](axis-readouts.csv),
[fixed cases](case-studies.csv), [complete receipt](receipt.json),
[input/provenance manifest](cases.json), [direction readouts](direction-readouts.png),
[preregistered protocol](../sample-structure-protocol.md).
