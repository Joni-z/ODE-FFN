# Report2 recap and work since the report

11 November 2026. Current candidate: **ISRU-D K8**, explicitly selected by
the user. This note is separate from the finalized report.

## Report identity and scope

[Report2](../../reports/report2/report2.pdf) is three pages, dated 4 November
2026. The user finalized it on 9 November; its source and protected artifacts
are bound by [finalization.json](../../reports/report2/finalization.json).
The last report-content commit is `06470b3`. No report content is revised here.

The report asks which ODE-inspired feature constraints actually help
flow-matching FFNs. Most evidence uses width-64, three-block time-conditioned
velocity MLPs on synthetic data, not complete JiT image models. Earlier
comparisons use 5k updates; the recent main comparisons use 20k, batch 512.
Field MSE measures fitting, and SW2 measures generated distributions.

## What was already in Report2

### Page 1: clocks, order, P and Muon

The original two-map ODEFFN has input-dependent internal clocks and a shared
P across each layer's Taylor terms. Internal feature time differs from global
flow-matching time. Clock, exact-solve and order comparisons did not yield a
uniform quality/cost improvement. The later scalar K1/K2 comparison favors K1
on field MSE in all eight matched settings; longer training does not establish
an overall advantage over strong ordinary references.

The Muon analysis separates the spectrum of updates from that of learned
weights. On Product16 scalar K2 at 20k, AdamW/Muon update stable rank is
3.08/32.74, while mean P spectral norm is 3.37/19.17. Field MSE is
0.03343/0.02385; ordinary SwiGLU192 reaches 0.01567. Norm-matched momentum
does not reproduce all effects. The tested decay and spectral-flattening
settings harm fitting. These results are already in the report, not new
post-report findings.

### Page 2: field structure, directions and nonlinear features

Three explorations lead to the diagonal-feature study:

- A full-gradient field readout improves Product16 fitting for both ODE and
  ordinary backbones. At 5k, original ODE MSE falls from 0.325527 to 0.037320;
  gradient SwiGLU gives 0.033826 at nearly equal dense inference work.
- Input-dependent orthogonal directions help a thin 2D mode plane in 128D:
  K2 MSE 0.000685, K1 0.000713, SwiGLU192 0.003095. Adding variance outside
  the plane reverses the sampling comparison. This supports geometry-dependent
  direction adaptation, not a universal high-order advantage.
- A nonlinear internal flow supplies the exact cubic response
  q(u) = u/sqrt(1 + 2r u^2). This motivates the ISRU diagonal inside P D(h).
  The report includes linear/MLP controllers, output transforms, direct
  diagonals and radial controls, followed by ISRU-D and Free ISRU.

The main Ring8 table uses the **K1** ISRU-D model, Heun128 and eight common
4,096-sample banks:

| FFN / optimizer | Field MSE | Mean bank SW2 |
|---|---:|---:|
| ISRU-D K1 / AdamW | 0.00977 | 0.17110 |
| Free ISRU / AdamW | 0.01000 | 0.17930 |
| SwiGLU64 / Muon | 0.01128 | 0.17145 |

ISRU-D has a fitting lead on Ring8; sampling nearly ties the selected Swi64.
Product16 favors ordinary references: mean SW2 is 0.08900 for ISRU-D,
0.08596 for Free ISRU and 0.08350 for selected SwiGLU192. The K1 results must
not be relabeled as performance of today's K8 selection.

### Page 3: fitting improves while mode probabilities drift

ISRU-D K1 from 15k to 20k improves field MSE 0.01026 -> 0.00977, but its
eight-bank mean SW2 worsens 0.16054 -> 0.17110. On pooled saved samples,
centered-residual SW2 improves 0.01768 -> 0.01502 while sector-frequency L1
error to equal weights grows 0.03827 -> 0.05450. All three compared families
show this residual/mass pattern; conditional covariance changes are mixed.

Common-source path comparisons and oracle-prefix replacements locate much
of the particle-to-mode difference early or midway along the measured paths.
For one ISRU-D case, oracle-prefix replacement reduces endpoint MSE from
0.04934 to 0.00075. This is an oracle-relative map diagnostic, not an optimal
distributional coupling or a new image-quality result.

The report already derives boundary-normal probability flux and a first-order
sector-mass response, and relates matrix updates to J_downstream DeltaP q(h).
It ends with two proposed tests: calibrate the flux prediction and compare
functional parameter updates with proper ordinary controls. Those predictions
remain uncalibrated; the report does not present them as completed experiments.

## What was done afterward

The published history from the report-content commit through `620808b`
contains documentation, delivery, meeting preparation and research assessment
changes. It adds no new training code, run endpoints or empirical study.

| When | Work | What actually changed |
|---|---|---|
| 4 November | Verified report delivery (`f33f994`) | Delivery/evidence records, not a new experiment |
| 9 November | User finalization and meeting preparation (`56a57b5`, `315a801`) | Protected report identity and separate professor briefing, delivered to MBP |
| 11 November | [Novelty assessment](2026-11-11-novelty.md) (`761a2a7`, `70505aa`) | Bounded primary-literature review and evidence/claim separation; no new performance result |
| 11 November | Initial image-transfer proposal (`d67c46f`) | Exact K1 candidate and controls documented; discovery that image --ffn isru is ordinary Free ISRU |
| 11 November | [Taylor-order clarification](2026-11-11-taylor-order.md) (`620808b`) | Rechecked old K4/K8 results and derived frozen-D versus state-feedback distinctions |
| 11 November, current decision | User selects ISRU-D K8 | Main candidate changes to eight Taylor terms with entrance-fixed D; implementation and training pending |

The old scalar K4/K8 experiments are from 20 October, before Report2. Their
recent re-examination is new analysis, not a new experiment. Likewise, Muon,
the various diagonal designs, ISRU-D K1 and the mode-allocation diagnosis
must not be presented as post-Report2 work.

## Current candidate and next concrete work

The [selected operator](2026-11-11-image-candidate.md) is
T8(h;P) = sum from k=0 to 8 of (P D(h))^k h/k!, with D fixed at that layer's
entrance. The full FFN is T8_2(SiLU(T8_1(h))). P1/P2 are independent and each
is reused across its eight terms. The second layer calculates its own D.

This restores the explicit shared-operator exponential-approximation test.
K8 is finite order; its actual accuracy and quality have not been measured.
It uses the same 2d^2 P parameters as K1 but 16d^2 forward dense MACs per token,
about twice native ratio-4 SwiGLU's FFN matrix work at the same model width.

Next work is the exact K8 implementation in the minimal author-derived FFN
hook, numerical checks and a measured image-training budget with ordinary
controls. The current image CLI still exposes only native SwiGLU and ordinary
Free ISRU. No new experiment is dispatched in this recap, and Report2 stays
finalized. The standing training seed remains 0.

## Verification

The recap's thirty quoted report values were checked against finalized TeX.
All eleven protected report hashes match; neither report directory changes
in this pass. Local links and the selected K8 parameter/MAC arithmetic pass.
The post-report file list contains Markdown and JSON records, with no newly
published training implementation or run endpoints.
