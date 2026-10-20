# What this ablation can distinguish

Write the layer as F(h) = T_K(s(h)P)h, where
T_K(A) = I + A + A²/2! + ... + A^K/K!.

## A constant duration is a redundant scale

For any learned constant s > 0 and fixed s0 = 0.6, define P0 = (s/s0)P.
Then T_K(sP)h = T_K(s0 P0)h exactly, for every h and every K.
The same identity holds separately in both solves around the SiLU.
Thus fixed and learned scalar durations give the same function class when P
is unrestricted. The learned scalar adds a redundant parameterization.
AdamW trajectories and gradient clipping need not be invariant under this change;
different optimization outcomes remain possible.

This identity would need re-examination under a norm cap or a penalty on P.
That is one reason not to combine gate ablation with spectral constraints now.

## Input-dependent duration changes the derivative

The duration is computed once from the layer input. Differentiating gives:

    J(h) = T_K(sP) + [P T_(K-1)(sP) h] grad(s)^T

The second term has rank at most one at each h. It can still change the leading
singular value appreciably. Its spectral norm equals

    ||P T_(K-1)(sP) h|| * ||grad(s)||

For K=2, the multiplying vector is P h + s P² h.
Fixed and scalar durations remove this derivative term with respect to h.
The entire ODEFFN remains nonlinear because of the SiLU between the two layers.

For the original sigmoid gate:

    grad(s) = (0.8/10) sigmoid(g/10) [1-sigmoid(g/10)] grad(g)

Hence ||grad(s)|| <= 0.02 ||grad(g)||. Temperature 10 does not by itself guarantee
a small correction: the learned gate network and P can amplify the two factors.
The step value s and the full input derivative must therefore both be measured.

## Existing evidence that motivates, but does not answer, the experiment

In the archived Ring8 K=2 input-gated run at step 5,000, layer
`blocks.1.ffn.ode1` has mean gate-correction norm 2.097600 and mean frozen
polynomial norm 2.600018, on the existing 16 hidden-input probes.
Its sampled durations range from 0.370918 to 0.938032.
The correction is not trivially small on these probes.

Source: `docs/research-2026-10-20/diagnostics/gmm_ring8_2d/gate_k2_input-step0005000.json`.
Checkpoint SHA256:
`d7d81a148eebbeffdea7640b7813dcd99509a7085aa586d4cc82158acc1c5edb`.

These two mean norms cannot be added to obtain the mean full-Jacobian norm.
They do not show that the gate improves generated samples, nor describe a
global derivative bound. Retraining the controlled alternatives addresses the
former question; layer diagnostics explain what each alternative learned.

## Decision after this round

If the input gate loses, simplify before designing constraints on it.
If it wins consistently at the primary and validation-selected endpoints,
separate its input adaptation from its extra network capacity in the next
experiment. If scalar and fixed differ, study their effective matrices sP and
optimization trajectories; do not describe that difference as new expressivity.
