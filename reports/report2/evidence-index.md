# Report2 evidence index — ODEFFN mainline

The November 2 rewrite supersedes the previous ISRU-centered narrative.
Start with the [report](report2.pdf), then the [complete attempt ledger](attempts.md).
All post-finalization research belongs to Report2. Early material is included
as labeled context; Report1 remains unchanged.

| Research question | Evidence route | Current interpretation |
|---|---|---|
| Do P constraints or higher Taylor orders help? | [Spectral/order analysis](../../docs/research-2026-10-20/analysis-output/analysis-report.md), [frozen exact-flow intervention](../../docs/research-2026-10-20-followup/analysis-report.md) | Smaller local sensitivity or more accurate integration does not guarantee better fitting/generation. Frozen substitutions are not cold-training comparisons. |
| What does the original clock contribute? | [Gate case study](../../docs/research-2026-10-21-gates/findings.md), [derivative study](../../docs/research-2026-10-21-gate-geometry/findings.md), [architecture review](../../docs/research-architecture-search/next-direction-evidence.md) | Helps Ring8 separation against constants, with extra capacity; vector clocks, range, reuse and ordering have limited transfer. |
| Does changing the whole field help? | [Complete readout](../../docs/research-conservative-readout-2026-10-26/results/analysis-report.md), [orientation](../../docs/research-derivative-orientation-2026-10-26/results/analysis-report.md) | Original ODE improves strongly on Product16, but the useful computation is also available to ordinary backbones. |
| Can P act in useful input-dependent directions? | [Conditional frame](../../docs/autoresearch/cycle11-conditional-frame/analysis.md), [independent second coefficient](../../docs/autoresearch/cycle14-independent-coefficient/analysis-report.md) | Thin-target gain; no established K2 requirement or thick-target sampling gain. |
| Is the gap linear, noisy-label driven, or confined to the head? | [Affine decomposition](../../docs/research-affine-error-2026-10-26/analysis-report.md), [population labels](../../docs/research-architecture-search/analysis-population-noise/analysis-report.md), [head refit](../../docs/research-frozen-features-2026-10-26/analysis-report.md), [local teacher](../../docs/research-local-ffn-teacher-2026-10-26/analysis-report.md) | These limited repairs do not close the main nonlinear-task gap. This is not an approximation lower bound. |
| Does a nonlinear internal flow help? | [Cubic split](../../docs/research-cubic-split-2026-10-26/analysis-report.md), [dependent signs](../../docs/research-joint-dependence-2026-10-26/analysis-report.md), [half mixture](../../docs/autoresearch/cycle02-full-support/analysis.md) | Tied cubic improves Product16, but the better free-ISRU control gives up the tying and has clear task boundaries. |
| Which dynamics are supervised? | [Exact ordinary-flow realization](../../docs/autoresearch/cycle03-gated-shear/analysis.md), [denoiser dynamics](../../docs/autoresearch/cycle22-denoiser-dynamics/analysis-report.md) | An ODE realization is not sufficient evidence of a useful bias. The conditional mean is not a conserved endpoint map. |
| Do Muon and diagonal generators improve original ODE? | [Replayed36-cell study and full curves](../../docs/research-muon-pd-2026-11-02/wave2/analysis/analysis-report.md), [registered second wave](../../docs/research-muon-pd-2026-11-02/wave2/protocol.md) | Muon improves Product16 original-ODE fitting, but ordinary SwiGLU remains better. Update magnitude and spectral shape are confounded; PD does not beat DP field error. The controlled follow-up is pending. Current report PDF is unchanged. |
| What do image runs establish? | [Baseline validity correction](../../docs/autoresearch/baseline-validity-review.md), [author-code checks](../../docs/upstream-baseline/validation.md), [synthetic identity audit](../../docs/synthetic-audit-2026-11-01/EXPERIMENT_AUDIT.md) | No adequate image baseline and no justified FFN quality verdict. Synthetic MLP evidence does not establish JiT transfer. |

For every smaller branch, use the ledger's stable IDs and primary source links.
The archived area inventories in [scope.json](scope.json) and the current
[attempt inventory](attempt-inventory.json) serve different purposes: the former
tracks post-Report1 Git changes, the latter covers the preserved documentation.
Neither is a count of independent training runs.
