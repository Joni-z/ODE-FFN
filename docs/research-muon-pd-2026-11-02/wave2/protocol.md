# Wave2: direction, growth and diagonal placement

Registered before second-wave training on2November2026. All runs use model
seed0,20k updates,batch512 and the existing cached banks, independent linear FM
loss, initialization, auxiliary AdamW3e-4, clipping1 and evaluation protocol.
The image trainer and report PDFs remain unchanged.

## Evidence that changes the next test

First-wave checkpoints were replayed on their original hosts. All36 reproduced
their saved generated arrays exactly; independent NumPy SW2 agreed within1.1e-8.
The analytic validation-field MSE was also reproduced. Muon improves original
ODE fitting on Product16, but ordinary SwiGLU remains better.

The strongest Product16 scalar Muon endpoint has mean P spectral norm19.17,
versus2.03 for default AdamW. In the first ODE solve, the mean gate is0.2038,
95.3% of probe entries satisfy the native saturation criterion, and the largest
recorded layer Jacobian norm across the fixed two-point probes is137.9.
The corresponding AdamW probe maximum is4.09. These are local observations,
not global Lipschitz certificates. Vector DP/PD gates have substantial channel
variation; the negative PD result is not simply an untrained constant gate.

## Three controlled blocks,30 new runs

Run all15 arms below on Ring8 (Lenovo) and Product16 (145). Reuse all audited
first-wave controls on the same host/device/data stream and20k budget, including
both AdamW rates, both Muon rates and both SwiGLU widths. A continuation preflight
must establish bit-identical original-arm initialization and updates under the
new adapter. No Gaussian claim is made from this nonlinear-task screen; any
proposed successor must subsequently retain the simpler control behavior.

| Block | New arms per task | Purpose |
|---|---|---|
| Direction at matched magnitude | scalar ODE and SwiGLU128, momentum control at LR0.006/0.02:4 arms | Does spectral reshaping matter beyond the update's Frobenius magnitude? |
| Growth control | scalar ODE and SwiGLU128, official Muon LR0.02 with selected-matrix decay0.1/0.5:4 arms | Can feedback on matrix magnitude reduce large P/Jacobians while retaining fit? |
| Diagonal placement | DP and PD with Muon LR0.006/0.02:4 arms; balanced sandwich with AdamW3e-4 and Muon0.006/0.02:3 arms | Does the optimizer interact with channel modulation, and does balanced placement help without new parameters? |

No weight decay is combined with the new diagonal placements in this wave.
No new activation, time distribution, loss, normalization or width is introduced.
The generated `wave2/grid.json` is the exact executable registration.

## A. Magnitude-matched momentum counterfactual

Let M be the momentum buffer and G the clipped gradient. Use the official
beta0.95 recurrence and Nesterov direction R, including its first-step convention:

```
M <- beta M + (1-beta) G
R <- (1-beta) G + beta M
U <- official five-step BF16 Newton-Schulz(R) * official shape scaling
control_update <- R * ||U||_F / ||R||_F
```

Both U and the counterfactual have the same Frobenius norm for the **same current
R**; the control preserves R's singular-value ratios instead of flattening them.
Zero R gives a zero update. This does not force two separately trained runs to
follow identical norm trajectories, since their gradients diverge. Log both
the actual update and its same-gradient hypothetical Muon norm. Copy the
official momentum/NS functions unchanged and keep only this ablation local.
The control is a mechanistic experiment, not a proposed new optimizer.
The logged `same_gradient_muon_update_frobenius` is `||U||_F` before
learning-rate multiplication; multiply it by the selected-matrix LR to compare
with `delta.frobenius`, the actual parameter displacement. These control arms
have zero decay, so no shrinkage term enters that comparison.

## B. Decay of P, not an asserted hard spectral projection

Use the official SingleDeviceMuon weight_decay argument on selected matrices
only. Gates, biases and the common backbone retain AdamW with zero decay.
The previous zero-decay Muon arms are exact controls; add the same intervention
to ordinary SwiGLU projections.

For a hypothetical uniform bound ||U_t||_2 <= C and0<=eta*lambda<=1, the update
`P_next=(1-eta*lambda)P-eta*U` gives

```
||P_t||_2 <= (1-eta*lambda)^t ||P_0||_2
            + C/lambda * (1-(1-eta*lambda)^t).
```

This follows by the triangle inequality and a geometric sum. It explains why
adding decay tests growth feedback. Our approximate BF16 NS updates are not
exact polar factors; do not set C=1 or claim a certified1/lambda spectral cap.
The positive-D gate derivative still contributes to the full layer Jacobian.
Report field/SW2, P/update spectra, gate saturation and fixed-probe Jacobians
together. Smaller P with worse fitting is not a success.

At each registered update diagnostic also record the exact identity
`||P+Delta||_F^2-||P||_F^2 = 2<P,Delta> + ||Delta||_F^2` in float64.
This separates the measured radial contribution from update energy. It is logged
at diagnostic steps only and must not be presented as a sum over unobserved
updates. The first wave did not retain this inner product.

## C. Balanced diagonal placement

Keep the same two-layer vector controller and positive range(0.2,1), with D
computed once per solve. Alongside A=DP and A=PD, use

```
S = sqrt(D)
A = S P S
F(h) = h + A h + A(A h)/2
```

At common P,D, `PD=S^(-1)(SPS)S` and `DP=S(SPS)S^(-1)`. Thus all three frozen
generators share eigenvalues. A benefit must involve their action, singular
geometry, full input-dependent Jacobians or learned trajectories, not different
initial frozen-generator eigenvalues. The sandwich adds no trainable parameter
or dense matrix multiplication. Retain gradients through both square-root gates
and verify the full Jacobian against dense algebra and finite differences.
It is not an exact continuously recomputed state-dependent ODE solve.

## Decision rule

Primary endpoint is final20k validation oracle MSE; independently reported test
SW2 must not be hidden when it disagrees. Show all rates/decays, both ordinary
controls and full curves. A gain over old ODE alone does not establish FFN
advantage. No inferential p-values are formed from checkpoint repetitions.
If matched-magnitude momentum matches Muon, weaken the spectral-direction story.
If decay reduces saturation/Jacobians but destroys quality, retain that tradeoff.
If sandwich or PD only improves after changing optimizer, report the interaction
with its scalar and DP controls. Do not launch another sweep automatically.

## Primary-source context

- [Official Muon](https://github.com/KellerJordan/Muon/tree/f98f1cacc0263b04290753e32be8d498c1efc806): pinned implementation for momentum, five-step approximate NS, shape scaling and decoupled decay.
- [Chen, Li and Liu,2025](https://arxiv.org/html/2506.15054v1): spectral-constraint interpretation explicitly includes decoupled decay and assumptions. It does not certify our previous zero-decay approximate implementation.
- [Nguyen et al.,2026](https://arxiv.org/html/2605.13079v1): spectral flattening is analyzed through learning-rate and curvature models. This motivates a local direction-versus-magnitude counterfactual, not an assumption that an LLM/optimizer result transfers to this ODEFFN.

The balanced sandwich similarity identities above are direct algebra for this
proposed test; no literature novelty claim is made.
