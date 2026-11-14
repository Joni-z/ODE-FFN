# Evidence notes

- Current implementation: `ffn.py`, two square frozen-diagonal Taylor maps around SiLU, internal width 1200, native width 768.
- Data source: `docs/celeba128-wide-20261114/continuation-160/matched-results.md`; same seed, training exposure and FID recipe.
- Live read: baseline completed 154 epochs; K1 complete 160 (FID 27.7887); K8 training complete 160, generation 50/196. No final epoch-160 comparison yet.
- Previous scalar state feedback, clocks and nonlinear flows need review before recommending another feedback mechanism.

## Reviewed evidence

- The earlier feedback study was vector, not just scalar: D_sigmoid(y) P y.
  Its midpoint arm lost to frozen K2 on Product16 and to ordinary controls.
  Source: docs/research-architecture-search/analysis-state-clock/analysis-report.md.
- Cubic split used y'=Py-y^3, not P ISRU(y). Free ISRU was stronger.
  Source: docs/research-cubic-split-2026-10-26/analysis-report.md.
- Current source has frozen D within each layer, two layers around SiLU,
  and width1200. Native SwiGLU has intermediate width2048.
- The potential test isolates the second response evaluation using identical
  P-call counts: frozen midpoint equals Taylor K2; refreshed midpoint instead
  integrates P ISRU(y). It is a conditional proposal, not a proven fix.
- The broad idea of feedback has been tried. The remaining concrete question
  concerns this response, matrix order and image task; it is not a novelty claim.

## Primary sources checked online

- https://arxiv.org/html/2511.13720v2 : JiT section4.4, native SwiGLU and
  general-purpose backbone design. Scope: why the baseline remains fixed.
- https://arxiv.org/html/2503.01329v1 : DiffEqFormer abstract, related work
  and explicit hidden dynamics. Scope: prior art, not image-performance evidence.
- https://aclanthology.org/2026.eacl-short.48.pdf : EACL2026 paper, abstract
  and ISRU/RMSNorm derivation. Scope: existing dynamic ISRU/outlier response;
  no empirical claim about this generator.
- https://arxiv.org/html/1710.09967v2 : original activation attribution;
  inverse-square-root saturation is established prior art.

## Verification

verification.json records a centered directional-derivative check, the frozen
midpoint/Taylor2 identity, nonlinear midpoint local expansion, exact MACs,
resolved evidence links and all11 protected Report2 hashes. No code under
the active trainer or either final report was changed.
