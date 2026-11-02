# Report2 claim-to-evidence map — 2 November 2026

Current deliverables: [PDF](report2.pdf), [source](report2.tex),
[full attempt ledger](attempts.md). Original ODEFFN is the mainline.
The source snapshot is `5c48a97bc0edb255da3f62f940b0500b00f3a4eb`.
This is a retrospective synthesis, not a live training-status report.

## Main-text claims

| Report location | Primary evidence | Interpretation and limit |
|---|---|---|
| Section 1: synthetic model and protocol | [Model audit](../../docs/synthetic-audit-2026-11-01/EXPERIMENT_AUDIT.md), [readout protocol](../../docs/research-conservative-readout-2026-10-26/protocol.md), [frame protocol](../../docs/autoresearch/cycle11-conditional-frame/protocol.md) | Width64/depth3 vector MLP has no attention/patches. Most reported direct-velocity studies use 5k updates, batch512, AdamW3e-4. Thin/thick uses a clean head. This is not a claim that every archived study uses identical settings. |
| Section 2: complete original-layer Jacobian | [Gate derivative derivation](../../docs/research-2026-10-21-gates/gate-counterexample.md), [observations](../../docs/research-2026-10-21-gates/findings.md) | Differentiate both the frozen polynomial and its scalar input clock. P spectra omit the rank-one clock derivative. Normalized residual-backbone stability needs additional terms. |
| Section 2: prior spectral and order context | [Cap study](../../docs/research-2026-10-19/results.md), [order intervention table](../../docs/research-2026-10-20-followup/frozen-order-results.csv) | Six lower Jacobian norms but only three SW2 improvements. Ring8 native K2 .1848201 versus exact .3813333 at the same frozen checkpoint. Earlier Report1 context, not newly obtained results. |
| Section 2: Ring8 case and P histories | [Findings](../../docs/research-2026-10-21-gates/findings.md), [endpoint diagnostics](../../docs/research-2026-10-21-gates/results/endpoint-diagnostics.json) | Far-from-mode fractions: fixed19.9%, input8.3%, Swi3.4%, target1.0%. Mean P norms1.837/2.139 and complete local Jacobians3.176/2.703 use the saved six-layer probes. Parameter counts differ. |
| Section 2: static free matrices and other local variants | [Static study](../../docs/research-architecture-search/evidence/static-summary.md), [full ledger](attempts.md) | Ring8 .076519 to .028901 at matched parameters and lower matrix work; feasible-map sets and optimizer coordinates differ. Other variant claims are scoped to their own comparisons. |
| Section 3: complete gradient readout | [Comparison CSV](../../docs/research-conservative-readout-2026-10-26/results/comparison.csv), [orientation](../../docs/research-derivative-orientation-2026-10-26/results/analysis-report.md) | Product16 ODE .325527→.037320; gradient Swi192 .033826 at183104/182336 inference MACs. Ring8 ODE .014462 versus gradient Swi192 .016419. Gaussian SW2 worsens .079723→.094981 despite better field MSE. Extra parent compute and ordinary controls stay visible. |
| Section 3: directions and geometry | [Frame table](../../docs/autoresearch/cycle11-conditional-frame/results/comparison.csv), [protocol](../../docs/autoresearch/cycle11-conditional-frame/protocol.md), [readout limitation](../../docs/autoresearch/cycle08-readout-invariant/analysis.md) | Thin K2 .0006851 versus original .0027286 and Swi192 .0030951. K1 .0007131 has lower tangent SW2 .1950 than K2 .2041. Thick K2 .3173 loses to Swi192 .2471 and Swi257 .2593. Conditional frames preserve pointwise spectra but add input derivatives. |
| Section 4/Table 1: tied cubic and free ISRU | [Cubic table](../../docs/research-cubic-split-2026-10-26/comparison.csv), [implementation](../../docs/research-cubic-split-2026-10-26/cubic_core.py), [all task endpoints](data/synthetic.csv) | Original/tied/free/Swi160 Product16 field .325527/.092938/.037159/.078178. Free matrices are independent after special initialization. Dependent-sign and half-mixture losses to Swi160 remain. Exact local cubic integration gives no full-network stability or ODE-specific benefit theorem. |
| Section 4: error diagnoses | [Affine error](../../docs/research-affine-error-2026-10-26/analysis-report.md), [frozen features](../../docs/research-frozen-features-2026-10-26/analysis-report.md), [local teacher](../../docs/research-local-ffn-teacher-2026-10-26/analysis-report.md) | Affine share3.89–3.93% and head-refit gap closure about1% are separate frozen-probe metrics, not the uniform-time training endpoint. Teacher fits are finite-optimization evidence. |
| Section 4: objective-level counterexample | [Derivation and numerical fixtures](../../docs/autoresearch/cycle22-denoiser-dynamics/analysis-report.md) | For Gaussian data, the conditional mean's material derivative is nonzero along the true flow. This rules out identifying that target with an invariant endpoint map, not every possible learned hidden ODE. |
| Section 4: image limit | [Validity correction](../../docs/autoresearch/baseline-validity-review.md), [preserved image snapshot](data/cifar-fid.csv), [upstream validation](../../docs/upstream-baseline/validation.md) | FID<50 is the user's prerequisite for an informative comparison, not a universal convergence theorem. No image result is used to select/reject ODEFFN here. |
| Final section: next two weeks | Prospective hypotheses using the preceding controls | No new training was launched. Failed screens remain failed; any extension needs an explicit new protocol and the researcher's decision. |

## Numerical provenance and figure

[make_mainline_figures.py](make_mainline_figures.py) extracts all relevant rows
from three original tables into [ode-mainline.csv](data/ode-mainline.csv).
Each metric keeps its source path and unique row selector. Source hashes are
in [ode-mainline-sources.json](data/ode-mainline-sources.json).
The figure compares within each task only. Bars are fixed endpoint values,
not confidence intervals. Dense MAC counts exclude non-matrix overhead and
must not be equated with measured latency.

The complete ledger covers each area of [attempt-inventory.json](attempt-inventory.json)
and additional root-level subquestions. It hashes the exact primary source
for every local record and distinguishes training, frozen interventions,
proofs, implementation and proposals. The six legacy ImageNet configuration
files lack a newly recovered validated completed-run/result binding.

This pass checks catalogue coverage/links, source hashes, selected numerical
claims, chart source rows, PDF layout/fonts and Report1 identity. It does not
recompute every historical experiment. The November1 checkpoint replay remains
separate evidence; no new training or sampling was used for this report.
Old image snapshots, scripts and figures are retained for historical
reproduction, not presented as current process observations.
