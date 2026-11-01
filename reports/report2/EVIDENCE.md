# Report2 evidence map — 1 November 2026

Draft: [report2.tex](report2.tex), [report2.pdf](report2.pdf).
The three-page narrative selects evidence after finalized Report1. It does not
claim that every completed project experiment appears in the main text.
The wider inventory remains in [evidence-index.md](evidence-index.md).

## Evidence cutoff and preservation

The image run is ongoing. Use all completed registered FID points at5k/10k/25k,
validation through40k, and matrix snapshots at0/2k/5k/10k/25k. Final50k FID is
not available in this draft. This is a fixed snapshot, not best-checkpoint
selection. The current training process and frozen runtime are untouched.

The CPU-only collector is [collect_evidence.py](collect_evidence.py).
Private snapshot: `/Users/mr.z/ode-workspace-prep/report2-evidence-20261101-v1`.
It checks every frozen source/data binding, all completed generated/feature
hashes, and the reference feature bank. It independently recomputes every
completed FID from saved features using the low-rank trace identity, rather
than the producer's covariance-eigenvalue implementation. The maximum absolute
difference is below0.000024, immaterial to the reported two-decimal scores.
It recomputes cold and25k FFN matrix summaries from the actual saved checkpoint
tensors for all arms. It performs no new training, sampling or feature extraction.

## Claim-to-artifact map

| Report location | Evidence | Allowed reading and limit |
|---|---|---|
| Sections1–2: cubic response and free maps | [Original implementation](../../docs/research-cubic-split-2026-10-26/cubic_core.py), [registered protocol](../../docs/research-cubic-split-2026-10-26/protocol.md), image [model](../../docs/autoresearch/isru-cifar10/model.py) | `y'=-y^3` gives `phi_s(u)=u/sqrt(1+2su²)`. The two free maps start at the same Taylor half-flow and then untie. This is not a new activation or proof that ODE coupling supplies the gain. |
| Section3/Table1: every synthetic law and ordinary budget | [synthetic.csv](data/synthetic.csv), source paths in each row; initial/dependent-sign/half-mixture comparison CSVs | All fifteen field/SW2 endpoints are retained. Product16 ISRU field MSE0.0371592 vs Swi1600.0781781; Gaussian and dependent signs lose to both budgets. Ring and mixture improve over the small control only. |
| Section3: generation does not mirror field fit | Same CSV, Product16 `sw2` | ISRU0.0928982, Swi640.0927582, Swi1600.0978353. No comparable generation gain over the smaller ordinary control follows. |
| Section4/Figure1: actual CIFAR recipe/curves/FID | [config](../../docs/autoresearch/isru-cifar10/config.json), [snapshot](data/snapshot.json), [validation](data/cifar-validation.csv), [FID](data/cifar-fid.csv), [numerical check](data/FID-verification.json) | The25k EMA scores252.96/262.93/268.74 are interim observations on1024generated/5000reference images. No architecture superiority, convergence or competent-baseline claim. At10k, wide SwiGLU is better than ISRU. |
| Section4/Figure2: raw versus EMA cases | Six exact grids `data/*-25k.png`; source PNG hashes in snapshot | The same first four source indices are shown, without quality selection or interpolation filtering. Raw texture and EMA blur are visual observations, not raw FID or semantic-quality scores. |
| Section4: EMA lag | Validation CSV and exact decay0.9999 in frozen source | ISRU raw/EMA loss0.148339/0.171237 at25k,0.147174/0.149046 at40k. Cold parameter mass is `0.9999^25000=0.0820747`. This does not decompose FID or equate parameter and prediction averaging. |
| Section5/Eq3/Figure3: sensitivity and learned maps | [matrix CSV](data/cifar-matrices.csv), [tensor verification](data/matrix-verification.json), direct chain rule | Mean spectral maxima1.26683→1.83575(W1)/1.60362(W2); mean entropy ranks186.578/188.328 at25k. The bare-FFN norm bound is not a complete residual-network guarantee or evidence for a spectral penalty. |
| Section6: direction-conditioned operator case | [curated comparison](data/conditional-frame.csv), [original analysis](../../docs/autoresearch/cycle11-conditional-frame/analysis.md), frozen generation cases there | Thin frameK2 MSE0.0006851 vs original0.002729 and Swi1920.003095. K1 nearly matches field error and has better tangent SW2. Thick frameK2 tangent SW20.31734 vs Swi1920.24708. No K2-specific mechanism or robust-transfer claim. |
| Section7: next-two-week hypotheses | Prospective tests, not experimental outcomes | The report does not launch a new family, retune EMA, change sampling or claim future results. |

## Mathematical checks and boundaries

For `f(h)=W2 phi_.6(W1 h)`, the chain rule gives
`J=W2 diag((1+1.2(W1h)^2)^(-3/2)) W1`. Every diagonal entry lies in(0,1],
so the operator norm is at most `||W2|| ||W1||`. Equality at some inputs is
not asserted. Added residual and normalization terms must be differentiated
separately. Earlier native preflight verified the same analytic Jacobian.

At fixed input, orthogonal similarity preserves eigenvalues/singular values
of P; derivatives of the input-dependent frame contribute to the whole map.
The old conditional-frame experiment is reused from its audited artifacts,
not rerun or presented as a new image experiment.

## Output verification

[verification.json](verification.json) records final PDF checks and Report1
identity. The current source is English, sole author Zhizhe Zhang, with no
bibliography, no statistical-significance claim, and no claim of instability
from the choice of model seed. The next-two-week hypotheses end the report.
