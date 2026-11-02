# Professor-guided ODEFFN: questions, tests and decisions

The question is whether an ODE-inspired FFN is a useful architectural bias for
flow matching. An ODE interpretation alone is insufficient. These experiments
use small, time-conditioned synthetic velocity networks; they do not establish
an advantage for JiT image prediction.

## 1. Does Muon make the learned dynamics better behaved?

Muon changes the geometry of matrix **updates**. In the recorded Product16
comparison, update stable rank increases from about 3.08 to 32.74. The learned
P can nevertheless grow: its mean leading singular value increases from about
3.37 under AdamW to 19.17 under Muon. Update isotropy is not weight isotropy.

The original scalar layer at K2 is

```
F(h) = h + s(h) P h + .5 s(h)^2 P^2 h.
```

Its factorial series converges for every finite P. The open issue is finite-order
accuracy and sensitivity, including derivatives of s, not a geometric-series
condition such as spectral radius below one. Equal eigenvalues also do not
control transient amplification for a nonnormal matrix.

Recorded norm-matched momentum controls show that update magnitude alone does
not reproduce Muon's fitting behavior. Decay lowers norms but harms fitting in
the tested settings. Thus neither “Muon flattens everything” nor “make P small”
is supported as the architecture mechanism.

Evidence: [update and norm controls](../../research-muon-pd-2026-11-02/wave2/analysis/analysis-report.md),
[row/column decomposition](spectrum/findings.md), [local-step derivation](muon-metric.md).

## 2. What does an input-dependent diagonal add?

The professor's PD proposal changes the conditional operator to P diag(d(h)).
We separated controller construction, output transform, polynomial order and
deterministic feature choices instead of changing all of them together.

| Question | Controlled test | Observation | Decision |
|---|---|---|---|
| Is the learned controller the limitation? | Affine, nonlinear MLP and rank-matched factorized controllers; bounded versus affine output | Some gains over weak parents; no ordinary-control win | No accepted design |
| Is a simple diagonal sufficient? | d(h)=h or a bounded transform, without a learned controller | Direct h is useful on Ring8; task dependent | Retain as a reference |
| Are tied higher powers needed? | K1 versus K2, same model family and optimizer/rate | Scalar K1 improves field fitting in all eight matched settings | Do not use more Taylor terms as a default improvement |
| Is the raw diagonal's sign structure useful? | Constant, bounded, shifted h, and absolute h | No general new advantage; abs worsens all six matched field comparisons | Close the abs setting |
| Was the small lead just a short budget? | Raw K1 and SwiGLU64/128/192 all trained to80k with exact20k prefixes | Raw beats matched Swi64 field error but loses sampling and wider controls | A limited tradeoff, no accepted FFN replacement |

The sign test has a specific mathematical premise: P diag(h) and P diag(abs(h))
have identical singular values at the same P,h, but their K1 increments are
P(h^2) and P(h abs(h)). Their different feature maps are invisible to that
singular spectrum. The negative training result also shows that a locally
attractive symmetry argument need not improve the complete network.

Evidence: [cycles1–2](cycle2-results/findings.md), [order/control test](cycle3-results/findings.md),
[feature controls](cycle4-results/findings.md), [abs rejection](cycle5-results/findings.md),
[long-budget results and examples](cycle6-results/findings.md).

## 3. Should P's singular values be constrained directly?

Raw P is not the whole function. With a free affine diagonal controller,
P'=PC^-1 and D'=CD preserve PD while changing P's spectrum. Fixed bounds and
direct h restrict that freedom, but conditional operators and complete network
derivatives still matter.

We therefore tested a concrete intervention rather than treating a pretty
spectrum as evidence: flatten each frozen P's singular values while preserving
its Frobenius energy, paired with scalar contractions of the same leading
singular value. Across six checkpoints, the only small favorable shape case
does not persist with longer training; stronger flattening always harms fitting.

This rejects a post-hoc repair. It leaves a distinct adaptation question:
can the network learn effectively if a **weak** shape constraint is present
throughout training? Cycle8 tests that one factor with native, shape and paired
scalar arms for both original scalar K2 and raw K1. It keeps the same loss,
model, rates and ordinary controls. A failure will not automatically trigger
a new strength/gate sweep.

Evidence: [gauge identity](logic-chain.md), [frozen shape/scale controls](cycle7-results/findings.md),
[constrained-training registration](cycle8.md).

## Current judgment

The useful progress is a sharper separation of mechanisms: optimizer geometry,
weight amplitude, spectral shape, diagonal features and polynomial order are
different hypotheses. Several attractive explanations have failed controlled
tests. There is no accepted new ODEFFN method from this study yet. The next
decision depends on the registered adaptation experiment, not on selecting
another favorable synthetic endpoint or starting image training prematurely.
