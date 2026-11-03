# Cycle11 registration: complete the controller/output contrast

Cycle10 is closed and does not qualify for continuation. This test fills one
missing cell in the existing controller experiment, not a radial-normalization
rescue. Keep the original entrance-frozen PD K2, both solves and middle SiLU.

## The missing comparison

| Logit controller | Bounded sigmoid output | Affine output |
|---|---|---|
| Original32-hidden MLP | Original vector-PD K2 | Cycle1 pd_mlp_affine |
| Rank32 factorized linear | Cycle2 pd_factorized_linear | **New pd_factorized_affine** |

The full-rank affine controller in cycle1 is not substituted for the rank32
linear row: rank/capacity would be a different factor. The new cell copies the
cycle2 factorized tensors and changes only gate_transform to affine. No new
parameter, rate, order, optimizer, loss, branch activation or P constraint.

Question: does the field effect of removing the MLP nonlinearity depend on the
outer output transform? Three corners are already replayed. A nonlinear-control
or saturation explanation from just those corners is incomplete. This test
closes that gap; it does not assume the missing corner will beat raw K1 or SwiGLU.
The earlier failed K1/DP range screen remains negative evidence, not a new claim
that saturation alone causes the gap.

## Mathematical control

Use the existing cycle2 zero-anchor logit value/Jacobian matching, preserving
the32-dimensional factorization and tensor shapes. The new affine diagonal is

```
d(h)=a+B h, rank(B)<=32
q(h)=diag(a)h + h*(B h)
F2(h)=h+Pq(h)+.5P[d(h)*Pq(h)].
```

It is a cubic polynomial per solve. At K1 it would be quadratic, but no K1
training cell is added. The full composition still contains middle SiLU,
LayerNorm and modulation; it is not a global cubic velocity network.

At the shared zero anchor, replacing the nonlinear controller with its affine
Taylor controller gives delta_d=O(||h||^2) for either output transform. Hence
Q2(PD_new(h))h-Q2(PD_parent(h))h=O(||h||^3). Each solve's value, Jacobian and
Hessian at zero agree within a fixed output column; cubic/higher behavior may
differ. This is a local matching statement, not global initial-function equality
or a claim that native hidden states lie near zero. Across output columns the
sigmoid/affine match is only at zero logit, as already registered in cycle1.

## Ten new20k trajectories

Three new candidates/task retain the original common settings: selected-P
AdamW.0003, Muon.006 and Muon.02; auxiliary AdamW.0003, seed0, batch512,
clipping1, same cached banks and native evaluation/Heun32. No new rate search.
Each host first repeats two hash-bound references: factorized-sigmoid at
Muon.02/Ring8 or Muon.006/Product16, and the earlier selected SwiGLU64. All
four references must reproduce archived20k numerical states and samples.

Reuse all three frozen corners at each of the same six task/optimizer settings,
including their source/checkpoint hashes. Keep the full20k ordinary frontier
from the freshly repeated cycle10 controls and the completed80k context visible.
Do not rerun every wider reference simply to increase experiment counts.

## Decision and analysis fixed before training

Final validation oracle MSE is primary; test SW2 is reported separately and
cannot choose a rate/checkpoint/model. For every setting report both controller
contrasts, the log-MSE difference-in-differences, D/sign/range use, complete
local Jacobians, P/update histories and wall cost. This interaction is a
fixed-recipe descriptive comparison, not a significance test over three rates.
Report all six settings and all four corners, even if the effect changes sign.

The external ledger's5% parent rule remains provisional. No continuation is
selected unless the new cell improves both factorized-sigmoid and MLP-affine
by at least5% on validation and beats the strongest ordinary20k field reference.
Any sampling deterioration remains an explicit tradeoff, not acceptance.
No image transfer, K1/range/strength rescue or added mechanism follows failure.

- [x] Implement minimal adapter/grid and verify CPU algebra/zero-anchor jets.
- [x] Publish/deploy immutable source and pass fresh documented native CUDA
      five-cell continuation and two parent8 reference checks perhost.
- [ ] Run monitored baseline-first queues, complete/replay ten endpoints and
      four exact final reference states; publish the complete factorial table.
- [ ] Apply the original decision rules before selecting another family.

CPU witness passed in the existing Mac CPU environment. All six native-fold
zero-anchor full-Jacobian discrepancies are below2.7e-11 and scalar-pullback
Hessian discrepancies below7.4e-11; same stored tensors/RNG as cycle2. These
checks verify implementation and local matching, not expected training gains.
