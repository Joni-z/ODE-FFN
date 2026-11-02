# What grew under Muon?

A read-only follow-up to the professor's spectrum question. This uses336
archived matrices (two tasks, four optimizer/rate settings, seven checkpoints,
six P matrices), not new independent training runs. All source checkpoint/config
hashes are retained in row-column.csv.

![Magnitude and coherence trajectories](row-decomposition.png)

All entries below average six matrices. AdamW uses the stronger tested LR.001;
Muon uses.02. Every rate is retained in the CSV and figure.

| Task/method | Initial P sigma | Final P sigma | Initial mean row norm | Final mean row norm | Initial row-normalized sigma | Final row-normalized sigma |
|---|---:|---:|---:|---:|---:|---:|
|Product16 AdamW|1.124|3.370|0.578|1.308|1.944|2.490|
|Product16 Muon|1.124|19.173|0.578|10.141|1.944|1.881|
|Ring8 AdamW|1.109|4.269|0.576|1.116|1.915|3.303|
|Ring8 Muon|1.109|22.235|0.576|9.208|1.915|2.155|

Write P=diag(r)R, with r the row lengths and R unit-row normalized. Then
||P||2<=max(r)||R||2. Both row and column bounds were checked for every tensor.
This is an inequality, not an exact factorization of spectral-norm changes.

On Product16 Muon's P sigma grows17.06x and mean row magnitude17.54x, while
row-normalized sigma changes0.968x. On Ring8 the corresponding ratios are
20.05x,15.98x,1.125x. The large raw norms coexist with much less growth in
normalized directional coherence. AdamW's row-normalized sigma increases more.
Column-normalized results show the same qualitative distinction in these runs.

This is consistent with a scale-drift explanation rather than increasing
row-direction alignment. It does not prove a causal attribution, convergence
failure, or that directly controlling row scales will improve our objective.
The completed decay experiment already shows that shrinking P can damage fitting.

The distinction is motivated by, but not a replication of, [Muown's LLM study](https://arxiv.org/html/2605.10797v1).
That paper separates row magnitude and row coherence; here we examine small ODE
matrices and retain our own task/budget limits. No new optimizer is introduced.
For PD designs, additionally analyze A(h)=P D(h), since raw P admits compensating
rescalings with an unconstrained D. See ../logic-chain.md for the exact algebra.

## Identifiability case study

On the cycle1 affine-PD Product16 checkpoint at1k updates, rescale each hidden
channel by a fixed positive diagonal C and compensate P/controller parameters
using the exact identity in ../logic-chain.md. In the first solve, P sigma changes
1.4419 to7.5612 while the conditional generator changes by1.39e-16. The complete
64-case prediction changes at most2.22e-15; all six full local layer Jacobians
agree within4.03e-16. See gauge-case.json for checkpoint SHA and per-layer values.

This is a frozen float64 reparameterization check, not training or a quality
improvement. It demonstrates that raw P spectra are coordinate-dependent in
the unconstrained affine-D family. It does not assert arbitrary compensation
inside the fixed bounded-sigmoid parent.

## Flatter energy is not automatically better conditioning

The enriched CSV retains every singular value, smallest singular value, condition
number, singular-value coefficient of variation and top-eight energy fraction.
Final six-layer means at the same selected rates:

| Task/method | Singular-value CV | Top8 energy fraction | Mean condition number |
|---|---:|---:|---:|
|Product16 AdamW.001|0.805|0.528|940.8|
|Product16 Muon.02|0.571|0.353|1017.0|
|Ring8 AdamW.001|1.006|0.689|2299.1|
|Ring8 Muon.02|0.677|0.457|1397.4|

Muon spreads energy more broadly by these summaries, but Product16's mean
condition number is not lower. Minimum singular directions remain important;
stable rank and spectral CV do not certify a well-conditioned or invertible map.
These are descriptive layer averages, not confidence estimates. All rates and
per-layer values remain in the CSV rather than choosing only favorable metrics.
