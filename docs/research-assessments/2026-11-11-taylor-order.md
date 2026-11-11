# Taylor order, the frozen-diagonal flow, and the ISRU-D name

11 November 2026. Evidence clarification; no experiment launched.

## Corrected interpretation

K1 remains a genuine first-order Taylor approximation. For the original layer,
compute s from the entrance h0 and hold it fixed during the internal solve:

```text
dy/dr = P y, y(0) = h0
y(s) = exp(sP) h0
T1 = h0 + sP h0
T2 = h0 + sP h0 + (sP)^2 h0 / 2
```

For A = sP, the omitted leading term is A^2 h0 / 2. In a compatible induced
matrix/vector norm, the remainder is bounded by
0.5 * exp(||A||) * ||A||^2 * ||h0||. Small internal time or small P can justify
a local approximation; accuracy at the actual learned s and P must be measured.
K1 also has the form of an ordinary gated residual map. These statements are
compatible: a numerical interpretation alone does not identify an exclusive
architectural advantage.

The previous conversational statement that ISRU-D retains only the scalar
activation's ODE origin was incomplete. It has a direct connection to the
professor's input-conditioned matrix exponential as well:

```text
d0 = (1 + 1.2 * h0^2)^(-1/2)
D0 = diag(d0)
A0 = P D0
dy/dr = A0 y, y(0) = h0
y(1) = exp(A0) h0
T1 = h0 + A0 h0 = h0 + P ISRU(h0)
```

Hold D0 at its entrance value throughout this internal flow. It remains a
differentiable function of h0: holding it constant in internal time does not
mean detaching it in backpropagation. Higher Taylor terms reuse the same P
and D0 and are therefore a coherent continuation of the original idea.

In particular, the second-order term is
0.5 * P D0 P D0 h0. It is not 0.5 * P^2 ISRU(h0).
An implementation can avoid constructing D0:

```text
d0 = rsqrt(1 + 1.2 * h0^2)
term = h0
out = h0
for k in 1..K:
    term = P(term * d0) / k
    out = out + term
```

This is a prospective generalization of the current ISRU-D K1 case, not a
claim that it has already been trained at K2/K4/K8.

Do not confuse this with the state-dependent ODE dy/dr = P ISRU(y). For that
nonlinear ODE the second-order state-time term is
0.5 * P diag(q'(h0)) P q(h0), with q'(h0) = d0^3. The two models agree to
first order and differ at second order. Recomputing D at evolving states
changes the internal ODE rather than merely improving the original frozen-D
exponential approximation.

## What the original K4/K8 study actually measured

The [20 October protocol](../research-2026-10-20/protocol.md) trained the
original input-clock scalar architecture at K1/K2/K4/K8, width 64, depth 3,
AdamW 3e-4, batch 512 and 5,000 updates on three 2D laws. The primary sampling
readout was SW2 with Heun32. These are not the later 20k/Heun128 ISRU-D runs.

Native scores, rounded to six decimals, from the
[recorded table](../research-2026-10-20/stage-report.md):

| Trained order | Gaussian SW2 | Ring8 SW2 | Moons SW2 |
|---|---:|---:|---:|
| K1 | 0.061061 | 0.200826 | 0.031507 |
| K2 | 0.067437 | 0.184820 | 0.027083 |
| K4 | 0.070309 | 0.175176 | 0.027682 |
| K8 | 0.070221 | 0.174829 | 0.027664 |

Ring8 improves materially from K1 to K4. Gaussian favors K1, and Moons favors
K2. K4-to-K8 changes SW2 by less than 0.2% on each law while full-network
forward dense MACs rise from 154,304 to 252,608, about 63.7%. These are matrix
work counts, not measured full-training runtime. The corresponding
[contrasts](../research-2026-10-20/analysis-output/contrasts.csv) retain all rows.

K8's measured mean exponential-action relative errors are about 1.33e-7,
1.82e-6 and 6.16e-8 respectively. Better exponential approximation therefore
does not imply a cross-task monotonic improvement in generation. It does not
follow that higher order is always ineffective or that K1 is always optimal.

The [frozen-order follow-up](../research-2026-10-20-followup/analysis-report.md)
replaced the evaluation polynomial while holding trained weights fixed.
K1/K2-to-exponential worsened the recorded SW2, whereas K8-to-exponential
changed it by at most about 0.003%. This shows adaptation to the training
operator; it is not the outcome of training a fresh exact-flow model.

The [later 20k comparison](../autoresearch/professor-20261103/cycle3-results/findings.md)
found lower field MSE for scalar K1 in all eight matched optimizer/rate/task
comparisons, but better sampling in only five. Its direct raw-diagonal cases
were also task dependent. This explains the historical focus on K1; it does
not determine the best order for the subsequently introduced ISRU diagonal.

The current [ISRU-D builder](../../studies/muon_pd_20261102/autoresearch_20261103/cycle13/trial13.py)
and registered family are K1. The scoped search of published study code,
grids and reports found no completed ISRU-D K4/K8 comparison. Original-scalar
order evidence should not be presented as such a comparison.

## Name

ISRU means **Inverse Square Root Unit**, an existing activation introduced in
section 5 of [Carlile et al., 2017](https://arxiv.org/html/1710.09967v2#S5):
q(x) = x / sqrt(1 + alpha*x^2). Our setting uses alpha = 1.2. This name is not
an assertion that the activation is new.

The suffix **D** is this project's shorthand for its use as the diagonal
D(h) inside P D(h): D(h)h = ISRU(h). It does not mean depth, diffusion or
derivative. `ISRU-D K1` names the currently trained residual construction;
it should not hide the distinction between response function and Taylor order.

## Consequence for the image-transfer recommendation

The K1 image candidate is a practical performance hypothesis, not a settled
answer to the closed-form-approximation question. The user's objection keeps
that question central. Do not silently fix K1 as the only main architecture
or exclude K2/K4 solely on earlier scalar-clock results. A future order study
must hold the new parameterization fixed, define frozen-D versus state-feedback
semantics, train each order rather than only swap frozen inference, and measure
both exponential-action error and downstream quality/cost. K8 can provide an
accuracy reference where cost permits. No new order grid, budget or training
dispatch is selected in this clarification.

Finalized reports, code and all recorded experiment results remain unchanged.

## Verification

All twelve rounded native scores above match the archived frozen-order CSV
within rounding tolerance. The two full-network MAC counts give a 63.708%
increase. Six local source links resolve and all eleven protected Report2
hashes match. A two-dimensional numerical witness confirms the K1 identity
and distinguishes frozen-D from state-feedback second-order terms; the latter
agrees with a centered directional finite difference within 2.51e-12.
These are algebra/source checks, not a new training experiment.
