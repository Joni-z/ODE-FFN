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
| Do Muon and diagonal generators improve original ODE? | [Replayed36-cell study and full curves](../../docs/research-muon-pd-2026-11-02/wave2/analysis/analysis-report.md), [registered second wave](../../docs/research-muon-pd-2026-11-02/wave2/protocol.md) | Muon improves Product16 original-ODE fitting, but ordinary SwiGLU remains better. Update magnitude and spectral shape are confounded; PD does not beat DP field error. The [30-cell follow-up is complete and replayed](../../docs/research-muon-pd-2026-11-02/wave2/results/findings.md): tested decay harms fitting and diagonal placements do not beat strong controls. Current report PDF is unchanged. |
| What do image runs establish? | [Baseline validity correction](../../docs/autoresearch/baseline-validity-review.md), [author-code checks](../../docs/upstream-baseline/validation.md), [synthetic identity audit](../../docs/synthetic-audit-2026-11-01/EXPERIMENT_AUDIT.md) | No adequate image baseline and no justified FFN quality verdict. Synthetic MLP evidence does not establish JiT transfer. |

For every smaller branch, use the ledger's stable IDs and primary source links.
The archived area inventories in [scope.json](scope.json) and the current
[attempt inventory](attempt-inventory.json) serve different purposes: the former
tracks post-Report1 Git changes, the latter covers the preserved documentation.
Neither is a count of independent training runs.

## Professor-guided autoresearch,3November

[Current logic chain](../../docs/autoresearch/professor-20261103/logic-chain.md), [new spectral decomposition](../../docs/autoresearch/professor-20261103/spectrum/findings.md), and [cycle1 registration](../../docs/autoresearch/professor-20261103/cycle1.md). Row-magnitude growth dominates the observed raw P growth more than normalized row-coherence growth; this remains descriptive. The [first18 autoresearch trials are complete and replayed](../../docs/autoresearch/professor-20261103/cycle1-results/findings.md), with no ordinary-control win. A frozen case proves raw P spectra can change while the affine-PD function is unchanged. The [second18 trials are complete/replayed](../../docs/autoresearch/professor-20261103/cycle2-results/findings.md): direct h is a Ring8 lead with no established ordinary-control advantage. Full precision/predictor-Jacobian checks are retained. [Cycle3 completed/replayed22 order and smaller-ordinary controls](../../docs/autoresearch/professor-20261103/cycle3-results/findings.md): scalar K1 improves all8 paired field settings; raw K1 Ring8 is a small field lead with worse sampling, not an accepted method. The [frozen six-checkpoint solver check](../../docs/autoresearch/professor-20261103/solver-results/findings.md) retains ordering at Heun32/64/128. This index changes; the submitted report PDFs do not.

[Cycle4 diagonal-feature outcomes](../../docs/autoresearch/professor-20261103/cycle4-results/findings.md) and [frozen parity decomposition](../../docs/autoresearch/professor-20261103/symmetry-results/findings.md) are complete. No accepted ODE advantage; the next controlled sign/orientation contrast follows paired-error evidence and a pointwise singular-spectrum identity. Submitted report PDFs remain unchanged.

[Cycle5 abs/bilinear comparison and registered parity readout](../../docs/autoresearch/professor-20261103/cycle5-results/findings.md) completed: abs fails all6 matched validation comparisons and does not consistently reduce full-model even error. Stop this setting; inspect stored terminal-sample structure before new architecture selection. Submitted PDFs remain unchanged.

[Saved Product16 sample audit](../../docs/autoresearch/professor-20261103/sample-structure-results/findings.md) demonstrates a blurred axis despite low aggregate distances. [Cycle6](../../docs/autoresearch/professor-20261103/cycle6.md) tests only80k convergence with raw and all three native SwiGLU widths; exact numerical-prefix checks preserve20k comparability. Existing report PDFs are unchanged.

[Cycle6 final80k comparison](../../docs/autoresearch/professor-20261103/cycle6-results/findings.md) is complete with exact prefixes and generated-array replays. Raw K1 has a field-versus-sampling tradeoff against matched Swi64 and no wider-control win. Submitted PDFs remain unchanged.

[Cycle7 frozen spectral-shape controls](../../docs/autoresearch/professor-20261103/cycle7-results/findings.md):54cases and six exact anchors distinguish Frobenius-preserving flattening from operator-norm-matched contraction. No general post-hoc repair; constrained training remains a separate question. PDFs unchanged.

[Cycle8 constrained-training comparison](../../docs/autoresearch/professor-20261103/cycle8-results/findings.md) completed12replayed runs with four exact native anchors. Flatter P spectra do not improve fitting or sampling in this setting. The [research map](../../docs/autoresearch/professor-20261103/research-map.md) now connects both professor proposals through controls, mathematical distinctions and decisions. Submitted PDFs remain unchanged.

[Cycle9 native-feature accounting](../../docs/autoresearch/professor-20261103/cycle9-results/findings.md) reconciles P spectra, actual first-term responses and finite K2 alignment on all14fixed checkpoints. It is descriptive evidence, with no whitening/new-optimizer claim. PDFs unchanged.

[Cycle10 completed radial-diagonal comparison](../../docs/autoresearch/professor-20261103/cycle10-results/findings.md)
separates radial growth from fixed scaling, with all three ordinary widths.
Response amplitudes decrease substantially but neither task qualifies for
continuation under the predeclared criterion. Submitted report PDFs are unchanged.

[Cycle11 controller/output factorial](../../docs/autoresearch/professor-20261103/cycle11.md)
is running after fresh native CUDA/continuation checks. It fills a missing
rank-matched corner, not a normalization rescue. No outcome yet; submitted
report PDFs remain unchanged.
