# Cycle5: the absolute-diagonal hypothesis fails

All14 registered20k trajectories completed at frozen f81663a and passed native
replay. Generated arrays match exactly; field and NumPy SW2 discrepancies are
below3.30e-10 and7.51e-9. Ordinary bilinear references ran first. All rates are
published, including the negative abs results; no failed trial was retried.

## Purpose and observations

The abs arm replaced h*h by h*abs(h), retaining K1 and exactly the same inherited
tensors/RNG. Its conditional PD singular values match the raw arm at the same
P,h, while its increment has different parity. The ordinary bilinear control
uses native Swi64 projections with only the gate SiLU removed. No loss,
constraint, data, sampler or extra parameters were added.

| Task/family, lowest validation within family | Validation MSE | Associated SW2 | Optimizer/rate |
|---|---:|---:|---|
|Ring8 rawK1, previous|0.010913|0.177924|Muon.02|
|Ring8 absK1|0.012391|0.173648|Muon.02|
|Ring8 bilinear64|0.012045|0.181391|Muon.02|
|Ring8 Swi64|0.011279|0.150384|Muon.02|
|Product16 rawK1, previous|0.023711|0.111139|Muon.006|
|Product16 absK1|0.026569|0.089814|Muon.02|
|Product16 bilinear64|0.035447|0.077846|Muon.02|
|Product16 Swi64|0.018781|0.081061|AdamW.001|
|Product16 Swi192|0.015668|0.079206|AdamW.0003|

Abs worsens validation in all six matching optimizer/rate comparisons:8.1–17.1%
on Ring8 and3.9–22.9% on Product16. No provisional keep qualifies. Some sampled
distances improve against raw, but all abs cases remain behind the stronger
ordinary sampling controls. Stop this absK1 setting without a parity penalty,
projection or higher-order rescue.

The ordinary bilinear reference gives an instructive disagreement. On Product16,
its AdamW.0003 cell has field MSE0.068265 but SW2 0.070540, lower than the shown
strong SwiGLU SW2. This is retained as a diagnostic case, not used to select a
model/checkpoint. A large field error does not by itself prove bad terminal
generation. Conversely, one finite random-direction distance cannot certify
all mixture structure. Inspect saved samples in known latent directions before
choosing a new architecture or declaring this score a generative advance.

## The registered parity follow-up

All four validation-selected new-family checkpoints use Muon.02. Selection pools
retain all7 cells per task and match the final-validation minimum exactly.
Original field scores replay within3.30e-10; the decomposition residual is below
2.34e-16. No odd-projected model was trained or sampled.

On Ring8, abs increases even energy from raw's0.005208 to0.006027 and odd error
from0.006029 to0.006719. On Product16, abs's even energy0.011044 is only slightly
below the validation-selected raw's0.011520, while odd error increases from
0.012049 to0.015536. These are within-family validation selections, not a new
matched-rate causal experiment. The full-model parity hypothesis is not supported.

Product16 bilinear has lower even energy0.007262 but much larger odd error0.028034.
Its even fraction20.6% therefore must not be read as a better predictor: the
denominator is also larger. Improving one diagnostic is not model acceptance.

## What this closes

The exact singular-value counterexample remains mathematically valid, but its
chosen feature intervention is empirically unsuccessful. Hidden increment parity
does not control the composed network's parity or its task fit. Retain raw's
small Ring8 field lead as a tradeoff, not a general method. Do not reopen stopped
potential/PSD/midpoint families. The next selected action is a read-only terminal-
sample structure diagnostic, not another training trick or image transfer.

[Endpoints](endpoints.csv), [decisions](decisions.csv), [curves](curves.csv),
[replay](verification.json), [comparison](comparison.png),
[parity rows](parity/endpoints.csv), [145 selection receipt](parity/145.json),
[Lenovo selection receipt](parity/lenovo.json).
