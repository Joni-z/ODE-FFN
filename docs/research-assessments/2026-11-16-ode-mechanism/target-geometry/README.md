# From hidden dynamics to the learned target response

## Actual-law counterexample

The Product16 oracle is exactly a coordinatewise quantile flow in its known
latent basis. An analytically marginal-preserving perturbation has full-scale
linear endpoint fraction-12.579despite every population marginal remaining
correct. This rules out treating first-order prediction failure as generation
failure. [Derivation](product-quantile-derivation.md),
[complete oracle/control evidence and figures](product-oracle-control-results/analysis-report.md).
The registered twelve-model calibration continues; its outcomes concern
approximation usefulness, separately from distribution quality.
[Targeted primary-source scope/novelty check](control-literature-check.md).

## Later precision audit

The next native-only qualification is complete: all twelve fixed models pass,
ten at4096steps and two at8192. All original256sources/model are retained;
the independent reconstruction error is<=4.17e-17.
[Native grid evidence and figures](native-time-results/analysis-report.md).
This clears native endpoint precision, not the oracle/tangent/blend test.
[Registered refined calibration](refined-trajectory-protocol.md).

The256-source/twelve-checkpoint transported-error calibration is complete,
but none meets every endpoint/tangent precision gate by the registered1024cap.
Its approximation outcome remains unresolved. A separate verified construction
finds that the native dyadic grids cannot distinguish two continuous fields
using different time frequencies. Original fixed-grid sampling values and the
focal response association remain preserved; a continuous causal explanation
requires the next precision check.

- [Complete calibration and all unresolved rows](trajectory-results/analysis-report.md).
- [Actual-model temporal-alias counterexample](time-alias-results/analysis-report.md).
- [Next non-dyadic/high-resolution qualification](native-time-resolution-protocol.md).

## Confirmed spatial response comparison

**The matched stage-order effect now has a confirmed output-space correlate.**
Palindrome lowers full velocity-Jacobian error on Ring8 and raises it on
Product16, matching their opposite sampling effects. The direction holds in
every one of eight independent input banks per task.

| Task | PAL J error | ALT J error | PAL SW2 | ALT SW2 |
|---|---:|---:|---:|---:|
| Ring8 |0.156|0.191|0.086|0.108|
| Product16 |0.261|0.236|0.072|0.066|

The full comparison uses all8x4096existing confirmation source/target pairs,
five fixed times and four unchanged20kcheckpoints. Adjusted conditional input
intervals exclude zero. This confirms the association, not its causal role.
Ordinary Product16SwiGLU remains a counterexample to a general Jacobian-error
ranking: it samples better despite greater average derivative error.

- [Exact target derivation](derivation.md): Gaussian-source Bayes denoiser
  Jacobian equals scaled posterior covariance. This is prior theory about the
  **whole predictor**, not a constraint automatically inherited by hidden P.
- [Twelve-checkpoint selection observation](results/analysis-report.md): all
  models retained; original128-pair uncertainty and actual figures preserved.
- [Independent full confirmation](confirmation-results/analysis-report.md):
  all source/bank bindings, paired statistics, figures and independent checks.
- [Four-paper review](literature.md) and [coverage](literature-coverage.json).
- [Next transported-error calibration](trajectory-protocol.md): validate its
  approximation range before using it to guide another architecture.

Implementation: `studies/muon_pd_20261102/target_geometry_20261116/`.
geometry.py computes posterior moments; oracle_check.py verifies the identities;
frozen_probe.py observes whole-model derivatives; verify_probe.py independently
reconstructs them; analyze.py/analyze_confirmation.py produce these evidence
bundles. All raw arrays/checkpoints remain outside Git under
`/Users/mr.z/ode-workspace-prep/coefficient-20261116/target-geometry/`.

Only warm one-thread Mac CPUworkers were needed. Reports1/2, image FFNs,
native image training/loss/sampling and all frozen scientific runtimes stay fixed.
