# Independent affine-forcing code review

Reviewed 2026-10-25 before formal training. **No blocking issue found in the stable implementation, focused tests or registered plans.** The intervention is bias completion in the original K1 block; this review establishes neither an improvement nor a novel ODE mechanism.

## Source and mathematical checks

Compared `synthetic/models.py` with the actual frozen Stage22 source at `/Users/mr.z/ode-workspace-prep/architecture-search/sigmoid_glu_20261025_m1/source`. The new `forcing_mode` defaults to `none`. That path adds no parameter or RNG draw and preserves the old forward arithmetic. The trainer, diagnostic collector and study runner are byte-identical to that source. Full historical optimizer-trajectory compatibility remains a separate preflight obligation.

For either added mode, each original solve allocates one zero vector without RNG use. The existing K1 computation remains `h+DPh`, followed by exactly `D*b` for conditional forcing or `(shift+scale/2)*b` for static bias. The latter uses the same numerical midpoint as the clock. Both independent solves and their middle SiLU are retained. Whole-model guards require the original vector family, K1, constant-paired initialization and independent original controllers; they reject combinations with changed order, clock range, reparameterization, rate order, solve returns, shared features, additive generator or nondefault spectral cap. No extra dense operator is introduced.

The affine/static pair shares initial functions and **all** parameter responses at `D=cI,b=0`, including under a nonzero receiving head. The old parent has no b columns and is not a fully matched initial kernel. A zero bias with nonconstant clocks is also insufficient to pair the new columns. These distinctions are tested, not hidden by the native zero output head.

The statistics retain homogeneous `DPh` separately and replace the reported first-term RMS with the actual biased increment. Conditional correction, mean and centered RMS describe `(D-cI)b`; static mode reports zero conditional correction. The closed-form local sensitivity is `diag(b)J_s`, with the SiLU controller and outer sigmoid derivatives included. Its scalar/vector broadcasting is correct. Raw/effective P spectrum records still concern P, not an augmented forcing operator or the full velocity Jacobian.

## Independent verification and plans

```text
/Users/mr.z/ode-workspace-prep/.venv-test/bin/python -m unittest tests.test_affine_forcing -v
```

**Seven tests passed independently in 0.115 seconds.** Coverage includes exact common tensors/buffers/RNG, executed `F.linear` MACs, every output-coordinate response to every parameter with a nonzero head in float32/float64, active analytic input and parameter derivatives, truthful nonmutating statistics, single-layer translation conjugacy and unsupported-setting rejection. The translation test makes no full-backbone equivalence claim and supplies no training initializer.

An additional active width64/gate32 probe compared the logged local sensitivity against CPU float32 autograd: both gave `0.013011190108954906`. Actual MPS float32 forward/backward and statistics also passed, with finite gradients for all parameters and the same reported norm. This is a focused device check, not a formal MPS training result.

Both task plans and the queue were inspected. They specify exactly six arms across three tasks, seed0, 5,000 updates, two CPU workers and **gate_hidden32** for all original ODE arms. Merging each arm with the actual base configuration reproduces Product16 counts: parent83,664 parameters/81,984 MACs; either bias arm84,048/81,984; SwiGLU96 71,376/69,696; SwiGLU128 and GLU128 each83,598/81,792. Actual functional P work is counted once per K1 solve, rather than relying on a hook for its unused module forward. The budgets are close, not identical.

All18 initialization-reference occurrences were checked against the three actual step-zero checkpoint hashes. No analytic teacher or new initializer enters the plans. The protocol distinguishes a practical static-bias lead from additional conditional-bias evidence, retains the failed generic GLU and fixes the stop rules. Its final fixed-weight conditional-to-static replay is an output-dependence diagnostic, not a retraining result or pure state-variation isolation.

Full trainer/checkpoint/resume compatibility, monitor operation, frozen-runtime integrity and final endpoint/counterfactual audits remain separately required; these seven tests do not certify them. This review performed no formal training and changed no source or frozen runtime.

## Reviewed SHA256 values

| Repository-relative file | SHA256 |
|---|---|
| `synthetic/models.py` | `9fe326f400862597ac28a392b797f51b406955fdfaf5d373eed3907e71caf337` |
| `synthetic/train.py` | `7164d98ea2bd870c8d4fcc98b48d8e9b8938dfa1ac33fa69f708fbdc9766f90b` |
| `synthetic/diagnostics.py` | `11ecc22b36132680780b52cdfa42e18f8cfb57e60d06a3c47a112ae1a13762c6` |
| `scripts/run_architecture_study.py` | `57f40bbf6bb91687f41116348c0ba3018ad1e8ea04c33ff0e2a38e8ea37da8d5` |
| `tests/test_affine_forcing.py` | `b26116ed339658c908b8a291204dc275cc8fbc225279cdfac8bc192759ea0059` |
| `docs/research-architecture-search/protocol-affine-forcing.md` | `a2c145c90224d2b7dd95cd6d468216b8972e6d9bb5eca4b3598e19c6ad5d03cb` |
| `docs/research-architecture-search/affine-forcing-design-review.md` | `5115e72e139173158abe3041c2df21b44f477c0584fef1d7efd50bd2fff4c994` |
| `docs/research-architecture-search/affine-forcing-prior-review.md` | `0a69515ebbbbdac17b033a178bb7bcce03c1d724d68318468f1f37c87ad646d8` |
| `configs/synthetic/architecture_search/affine_forcing_v1_20261025_m1.json` | `20acdd4bc194f35bbba461fe6a369de5f0c1cbb74258c202dec4f9697aa00990` |
| `configs/synthetic/architecture_search/affine_forcing_product16_20261025_m1.json` | `72bfefed2835eabb0d33d9695c7c1263ca478a7bcc59cbd08c9beddfa1d83c4d` |
| `configs/synthetic/architecture_search/affine_forcing_queue_20261025_m1.json` | `530542c440e2e4c977ca49925fc6fed7309ee0896f537540085b06f4f9475163` |
