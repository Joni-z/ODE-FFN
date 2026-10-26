# Independent sigmoid-GLU implementation review

Reviewed 2026-10-25, before formal GLU training. **No blocking implementation or protocol issue found in the reviewed scope.** This is one known generic baseline control, not an ODE architecture or evidence that the project objective has been met.

## Implementation and pairing

Against the actual frozen Stage21 source at `/Users/mr.z/ode-workspace-prep/architecture-search/state_clock_20261025_m1/source`, the model changes are the small `SigmoidGLUFFN(SwiGLUFFN)` subclass, the `glu` dispatch and the CLI choice. The subclass inherits initialization and replaces only `SiLU(gate)*value` with `sigmoid(gate)*value`. Both biased projections, the nominal-to-actual hidden-width convention, state keys and RNG draws are preserved. No new parameters, temperature, centering, initialization adjustment or analytic teacher weights enter training. The old ODE implementations are unchanged; the diagnostic collector and study runner are byte-identical to that frozen source.

Equal native states do **not** imply equal hidden functions or initial NTKs. The formal zero output head masks both models' outputs and upstream gradients, but its own parameter gradients differ. A common nonzero head also exposes different outputs and input Jacobians. This limitation is stated correctly in the protocol and tested explicitly.

For Product16, GLU128 and SwiGLU128 each have **83,598 parameters and 81,792 dense MACs** per model evaluation. Actual executed `F.linear` work agrees with the existing counter. Pointwise operation costs differ; the review does not claim equal elapsed runtime.

## Independent verification

Command:

```text
/Users/mr.z/ode-workspace-prep/.venv-test/bin/python -m unittest tests.test_sigmoid_glu -v
```

**All four tests passed in 0.044 seconds.** They cover exact native tensor/buffer/RNG pairing and executed dense MACs; the zero-head and nonzero-head distinctions above; the analytic input Jacobian and finite gradients for every parameter; and the fixed-time raw-input representation construction.

The construction is checked at `t=.1,.5,.9` against an independently enumerated rotated 3D eight-component Gaussian mixture and the Product16 population oracle. Analytic weights occur only inside this unit test. The result concerns a biased raw-input GLU at each fixed time, not a shared-time trained model, the normalized backbone, or the cached empirical-bank posterior. It establishes no wide/deep SwiGLU lower bound.

The three plans contain nine cells: Gaussian2D, Ring8 and Product16 crossed with SwiGLU96, SwiGLU128 and GLU128. Independently verified all nine reference occurrences against their actual step-zero checkpoint SHA256 values, along with task sets, fixed options and validation sampling. The protocol preserves seed0, training objective/data, optimization and evaluation settings. Its corrected motivation includes confident coordinates: the registered regional readout rejects an ambiguous-boundary-only explanation. The predefined lead requires an overall improvement against both matched/lower-cost SwiGLUs with cross-task and Jacobian/SW2 guards; SwiGLU256 remains explicitly higher-cost context. Failure ends this fixed substitution without an activation rescue.

This review does not itself certify full trainer/resume/sampling behavior, historical rerun reproduction, runtime monitoring or final checkpoint replay. Those remain the separately required preflight and completion audits. No training was performed during this review.

## Reviewed SHA256 values

Paths are relative to the repository root. These bind the reviewed contents, not a later edit.

| File | SHA256 |
|---|---|
| `synthetic/models.py` | `71b1298c628daf633ce5ad0d2be28e4569a047f1f606a076dfe9635f4a9f8b8d` |
| `synthetic/train.py` | `7164d98ea2bd870c8d4fcc98b48d8e9b8938dfa1ac33fa69f708fbdc9766f90b` |
| `synthetic/diagnostics.py` | `11ecc22b36132680780b52cdfa42e18f8cfb57e60d06a3c47a112ae1a13762c6` |
| `scripts/run_architecture_study.py` | `57f40bbf6bb91687f41116348c0ba3018ad1e8ea04c33ff0e2a38e8ea37da8d5` |
| `tests/test_sigmoid_glu.py` | `3edebb7da0a340048d411e969887fae08c8303f70ed3cce6a79b4085b42a0283` |
| `docs/research-architecture-search/protocol-sigmoid-glu.md` | `62ef92afc5656efa2da0a2998a8f6f790d7bb6d1a2c6d323df259b837899028e` |
| `configs/synthetic/architecture_search/sigmoid_glu_v1_20261025_m1.json` | `fa7fccd096b816ff260010052d4d8ca20b0922dc6b2be3d3a9ecc15706e25b66` |
| `configs/synthetic/architecture_search/sigmoid_glu_product16_20261025_m1.json` | `a4d2d40590b5c18b0e0ca6177774ea12ab8df9c41abfab750a593e515e083946` |
| `configs/synthetic/architecture_search/sigmoid_glu_queue_20261025_m1.json` | `72466a97216c23f448398f0ced8308fa6c29c9f2a4d97faaa3e52a82d0162e1b` |
| `docs/research-architecture-search/posterior-basis-prior-review.md` | `0ae12e861b7fb881aa74cc6796d7de2b61126521f1ba462e1a58c2e155ea88d4` |
