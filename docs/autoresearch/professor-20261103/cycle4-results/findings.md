# Cycle4: deterministic diagonal features do not establish an ODE advantage

All18 registered20k trajectories completed and replayed from frozen c3dbe27.
Stored sample arrays reproduce exactly; maximum field and NumPy SW2 discrepancies
are8.19e-10 and1.35e-8. No failures or retries. This extends the K1 comparison
without new parameters, losses, constraints, schedules or packages.

## Purpose and observations

Constant d=1 removes input dependence; bounded d=.2+.8sigmoid(h/10) tests direct
input compression without K2; shifted d=h+1 adds a tied linear feature to the
raw quadratic branch. P/common tensors, data, optimizer assignments and budget
are inherited exactly. Initial functions differ, so amplitude, parity and
derivative effects are not independently identified.

| Dataset/family | Lowest validation MSE in family | Associated test SW2 | Optimizer/rate |
|---|---:|---:|---|
|Ring8 raw K1, previous|0.010913|0.177924|Muon.02|
|Ring8 constant K1|0.012337|0.171335|Muon.02|
|Ring8 bounded K1|0.011360|0.177053|Muon.006|
|Ring8 shifted K1|0.011272|0.180856|Muon.02|
|Ring8 native Swi64|0.011279|0.150384|Muon.02|
|Ring8 native Swi192|0.011231|0.165805|Muon.02|
|Product16 raw K1, previous|0.023711|0.111139|Muon.006|
|Product16 constant K1|0.018963|0.075520|Muon.02|
|Product16 bounded K1|0.023349|0.086112|Muon.006|
|Product16 shifted K1|0.023696|0.105222|Muon.006|
|Product16 native Swi64|0.018781|0.081061|AdamW.001|
|Product16 native Swi192|0.015668|0.079206|AdamW.0003|

These are validation-selected summaries, not a selection of the best test number
per column. Full rates remain in endpoints.csv. No new arm beats the strongest
ordinary field result. Ring8 raw K1 remains the small local field lead from
cycle3, with worse sampled distance than Swi64. Adding1 does not improve its
best field result and worsens its associated sampled distance.

On Product16 the constant control improves both raw-K1 field and sampling.
Its full FFN is exactly a bias-free ordinary SiLU FFN under W=I+P. Against the
matched Swi64 it has about1% worse field MSE but6.8% lower measured SW2. Against
the larger Swi192 it has21% worse field MSE but4.7% lower SW2. This ordinary
control is worth retaining; it is not evidence for input-dependent ODE dynamics.
Its low sampled distance also shows why field MSE alone cannot rank generators.

Removing K2 from the bounded direct design improves field error in all three
Ring8 settings and both Product16 Muon settings, but worsens Product16 AdamW.
It does not consistently surpass rawK1 or the ordinary controls. Some provisional
keeps only improve a weak matched parent; the keep ledger is not a method claim.

## Decision

Do not add spectral penalties or image transfer to these outcomes. Retain the
constant arm as an ordinary control, close the current shift1 setting, and use
the frozen symmetry diagnostic to decide whether a sign/orientation contrast
is justified. A later raw-versus-abs diagonal would match conditional singular
values at the same P,h, making it more informative than another arbitrary range
or shift. It still needs ordinary feature controls and actual quality evidence.

[Endpoints](endpoints.csv), [all decisions](decisions.csv), [curves](curves.csv),
[update spectra](updates.csv), [replay checks](verification.json), [figure](comparison.png).
