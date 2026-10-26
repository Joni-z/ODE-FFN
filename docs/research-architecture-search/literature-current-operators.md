# Two current operator papers, with transfer boundaries

Reviewed2026-10-24 during the identity-location experiment. These are bounded
methodological readings, not evidence that an ODEFFN candidate succeeds.

## Mamba-3, ICLR2026

[Primary paper, Sections3.1 and A.2--A.3](https://arxiv.org/html/2603.15569v1).
Its recurrence separates homogeneous state evolution from input forcing,
using two endpoint contributions. Its error-rate statement requires the
learned endpoint weight lambda=1/2+O(step); unconstrained learned weights
need not retain that second-order guarantee. The paper reports better
empirical performance without imposing that condition. Thus richer learned
dynamics and formal integration accuracy are distinct claims. Its axis is
sequence memory, rather than an FM channel FFN.

Our independent scalar check: integrating g(u)=u on[0,delta] gives delta²/2,
while endpoint weights give lambda*delta². A fixed lambda other than1/2
therefore has local error proportional to delta². Calling a learned rule
trapezoidal alone does not make it second order. No new integrator is
selected here.

## Kimi Linear, technical report2025

[Primary paper, Sections2.2,3 and6.2](https://arxiv.org/html/2510.26692v1).
KDA combines a channelwise diagonal decay with a constrained rank-one
correction and a separate memory write. Its two low-rank vectors are tied
to the key/decay, enabling a specialized parallel algorithm. This is a
structured operator and implementation design, beyond merely replacing a
scalar gate by a vector. The learned object is a matrix-valued recurrent
memory; its language-model results do not validate an image/FM FFN.

The transferable question is which operator components should share
parameters, with capacity and actual hardware cost controlled. Borrowing
its recurrence or efficiency numbers without its state/algorithm is unwarranted.

## Consequence for this project

The current K1 layer can be rewritten exactly as

```text
F_gamma(h) = (gamma*I+cP)h + (D(h)-cI)P h.
```

P simultaneously sets the constant reference map and the value directions
modulated by the controller. Changing gamma while preserving the initial
reference map changes that coupling. The identity-factorial test therefore
does not exhaust the question of operator parameter sharing.

A later direction might test whether the constant and adaptive components
should be coupled multiplicatively at all. That requires its own algebra,
matched initialization, capacity/compute controls and registered outcomes.
Neither paper supplies a theorem that the change will improve FM. No new
architecture is launched by this review, and no failed screen is extended.
