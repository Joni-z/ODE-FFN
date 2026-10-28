# Finite posterior-state eligibility

This is a data-free mathematical gate for the selected
[question](../post-cold-control-review/research-question-card.md), not an image
Gaussian-mixture fit, a trained FFN or a quality improvement. The
[derivation](derivation.md) keeps the third-moment term and separates fixed-y
precision evolution from fixed-z FM evolution. The
[protocol](protocol.md) fixes all cases, tolerances and the equivalence decision.

## Documented identity command

After committing clean source, prepare a new private runtime with a copy of
this folder, settings/source-commit and SHA256 bindings. Preserve the cold
and Gaussian reference seals. Use the unchanged warm Mac mini .venv-test
environment; no package installation or GPU allocation.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python -B /Users/mr.z/ode-workspace-prep/autoresearch/oct28-posterior-state-eligibility/source/check_identities.py --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-posterior-state-eligibility
```

The output must retain all 72 fixed-observation cases, 72 FM-coordinate
cases and 54 integrated endpoints. An absent/truncated observation is not
permission to restart an existing process. Retain source, logs, actual tool
exit and every failed case. This command supplies no training readiness or
image-performance evidence for a later model.

## Result and decision

All fixed checks pass under registered source aa74345. The finite component
state retains the third-moment term; its endpoint is exactly the stated
ordinary direct map. **Do not train a new ODELayer based solely on integrating
this known-prior state.** See [decision.json](results/decision.json).

| Check | Maximum absolute discrepancy |
|---|---:|
| Independent component conditioning | 1.78e-15 |
| Cubature third moment | 9.99e-16 |
| Complete precision derivative | 8.88e-16 |
| Ordinary endpoint, including FM coordinates | 5.56e-16 |
| Global fixed-z FM derivative | 1.47e-13 |
| Integrated full state and mean, 54 endpoints | 6.53e-11 |

The complete component-time vector has maximum absolute difference5.41e-9.
This passes the frozen absolute-plus-relative criterion: the affected logit
has magnitude265,699, with permitted error7.97e-5. One post-outcome arithmetic
check reproduces that case and retains both full derivative vectors in
[tolerance-witness.json](results/tolerance-witness.json). It does not rerun
the original full cohort or change a tolerance. It is not honest to report
all errors as below the absolute tolerance3e-10.

The unequal-covariance logit-ratio Hessian reaches norm1.273, so general
mixtures are not ordinary linear-key attention. They have direct quadratic
gates and affine experts. Dropping the FM observation-change term produces
a discrepancy up to2.163 on these fixed cases; fixed observation and fixed
FM input are different derivatives. These are identity witnesses, not a
sample of image-model quality or statistical effect sizes.

The two [counterexample posteriors](results/counterexample.json) have the same
mean0 and variance1 but precision drifts0 and-.75. The finite family carries
the distinction through component weights/means/covariances. This repairs a
specific closure error; it does not show that a learned hidden-state model
has discovered those posterior statistics.

## Evidence and scope

- [72 fixed-observation cases](results/fixed-observation.csv),
  [72 FM-coordinate cases](results/FM-coordinates.csv),
  [54 integrated endpoints](results/integrated-states.csv).
- [Closure verification](results/artifact-verification.json) and
  [actual archive exit](results/archive-tool-exit.json).
- [All six fixed priors](results/priors.json),
  [summary and environment](results/summary.json),
  [consumed original tool exit](results/tool-exit.json) and
  [post-outcome arithmetic exit](results/arithmetic-tool-exit.json).
- The calculation takes0.70seconds after imports, peak RSS0.263GiB; one CPU
  thread, no GPU, no data fit, no training and no new quality metric.
  No wall-clock comparison against a trained architecture is claimed.

A useful learned parameter constraint or cheaper approximation is not ruled
out merely because it has a finite expression. The rejection is narrower:
we have not specified such a restriction or tied a hidden feature to the
known observation model. An ODE implementation of these exact known states
alone supplies no distinct endpoint computation. No new parameterization,
loss, benchmark or mixture image-reference rescue is selected.

The [actual hidden-input check](hidden-input-check.md) adds a concrete
boundary: RMSNorm plus fixed time modulation bounds the token input. It is
not exactly an independent additive-Gaussian observation of a clean hidden
feature. The matching [frozen source check](results/norm-input-source-check.json)
preserves the actual native operator. A new posterior-inspired constraint
would need an explicit approximation or a different state role. No such
architecture is selected; the broader successor objective remains open.
