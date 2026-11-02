# Cycle3: the tied second-order term is not the current source of benefit

All22 registered seed0 trajectories reached20k and passed native checkpoint
replay. Generated arrays reproduce exactly; maximum field discrepancy is1.67e-9
and independently recomputed SW2 discrepancy1.18e-8. Source14ab235, data, loss,
optimizer assignments and final-checkpoint selection stay fixed.

## Order contrast

Removing the second-order term improves the original scalar model in all eight
same-task/optimizer/rate comparisons, by13.3–21.4% on Ring8 and14.6–19.7% on
Product16. Five of those eight also improve sampled distance; the other three do
not. This is evidence against attributing the present gains to the tied K2
Taylor term. It does not establish that every higher-order construction fails.

For direct D(h)=diag(h), K1 improves all three Ring8 field comparisons by6.3–13.6%.
Product16 is mixed: K1 is19.3% worse under AdamW,4.1% better under Muon.006 and
0.3% better under Muon.02. None of the Product16 comparisons establishes a5%
validation improvement. Raw K1 is not a generally better architecture.

## Ordinary controls and the local lead

| Dataset/model | Validation field MSE | Test SW2 |
|---|---:|---:|
|Ring8 raw K1, Muon.02|0.010913|0.177924|
|Ring8 scalar K1, Muon.006|0.011619|0.171432|
|Ring8 SwiGLU64, Muon.02|0.011279|0.150384|
|Ring8 SwiGLU128, Muon.02|0.011484|0.164520|
|Ring8 SwiGLU192, Muon.02|0.011231|0.165805|
|Product16 raw K1, Muon.006|0.023711|0.111139|
|Product16 scalar K1, Muon.02|0.019950|0.091687|
|Product16 SwiGLU64, AdamW.001|0.018781|0.081061|
|Product16 SwiGLU128, AdamW.0003|0.015668|0.081848|

Rows summarize the lowest validation error in each shown family, not a sampler-
selected result. All rates remain in endpoints.csv and the comparison figure.
Ring8 raw K1 improves field error3.3% against matched SwiGLU64 and2.8% against
the strongest larger ordinary reference; its SW2 is18.3% worse than SwiGLU64.
The field margin is below the preregistered5% improvement scale. This is a local
field-fitting lead, not an accepted generative method. Product16 still favors
ordinary controls, including the newly completed smaller reference.

Product16 raw K1 and SwiGLU64 have58,512/58,572 parameters and57,408/57,024 dense
linear MACs. These match closely in both resources. Counts are analytic, not
latency claims; common dimension-dependent counts differ on Ring8.

## What the block now computes

In column-vector notation, direct raw K1 is exactly

```
F1(h) = h + P (h * h)
J_F1(h) = I + 2 P diag(h).
```

The multiplication is elementwise. K2 adds the tied cubic expression
`0.5 P [h * P(h*h)]`. Thus an economical interpretation of the Ring8 result is
a residual quadratic feature map. Calling it an ODE discretization does not
prove that a continuous-flow inductive bias caused the improvement. The full
FFN still composes two such blocks with its existing SiLU.

The next useful D comparisons should distinguish this quadratic feature from
a constant diagonal, its bounded input version, and a minimal affine input
diagonal, while retaining K1 and the same P initialization. The existing curves
do not establish full convergence: several20k endpoints oscillate or remain
above their15k value. No asymptotic-superiority claim is made.

[Complete endpoints](endpoints.csv), [decisions](decisions.csv),
[curves](curves.csv), [verification](verification.json),
[comparison](comparison.png).
