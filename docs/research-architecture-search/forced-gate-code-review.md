# Forced-gate implementation review

Reviewed 2026-10-24 after the final helper/statistics fixes. **No blocking implementation finding.** The reviewed implementation matches the frozen-coefficient forced-response construction and the five-arm protocol. This is approval of the implementation for the bounded comparison, not evidence of architectural benefit or completion of the separate trainer/resume/monitor checks.

Scope: `ForcedGateResponse`, `ForcedGateSwiGLUFFN`, model/config/CLI propagation, diagnostic collection, focused tests, plan configurations and relevant runner/checkpoint paths. Only this review document was written; no code, tests, configurations or training states were changed.

## Checks and evidence

- **Mathematics:** `exact` evaluates `SiLU(a)*phi1(tanh(eta)*sigmoid(a))`; K2 evaluates `SiLU(a)*(1+c/2)`. Both agree with the augmented affine matrix formulation. For adaptive duration, the implemented derivative is `q*gain + a*q*(1-q)*response_factor`, where the factor is `1+c` for K2 and `exp(c)` for exact. This includes the clock derivative. Static duration uses the ordinary SiLU derivative multiplied by the static gain.
- **Numerics:** the small-argument degree-eight series has the correct value, slope and curvature at zero. The inactive quotient never divides by zero. Float32 quotient Hessians near the `.05` branch boundary have a separately documented `8e-5` test tolerance; value and first-derivative checks remain stricter. These are numerical tests, not exact floating-point identities away from zero. The tested dtype/device scope is CPU float32/float64 and MPS float32, not mixed precision or CUDA.
- **Initialization:** projections are constructed through the unchanged SwiGLU constructor before zero eta allocation. The tests confirm identical random-generator state, all common tensors, complete velocity outputs and common-parameter gradients in float32/float64, with a nonzero output-head probe. K2/exact full FFN parameter Jacobians agree, including nonzero eta directions. No new random draws or dense projections were introduced.
- **Static control:** trained static gain multiplies output columns; the output bias remains unchanged. The test checks complete FFN values and input Jacobians with nonzero projection biases. This is function-class equivalence, not equivalence of optimizer trajectories.
- **Budget and diagnostics:** nominal128 means actual85 channels. Three blocks add255 eta parameters and no dense MACs; instrumenting executed `F.linear` calls agrees with the runner's count. Additional pointwise work remains real. Hooks capture gate preactivations and are removed even on failure. Records distinguish latent diagonal rates, conditional rates, forcing, gate-response statistics and the full velocity Jacobian; they do not claim that bounded rates bound the augmented shear or network.
- **Protocol:** the configurations contain exactly five arms across three tasks, seed0, two CPU workers and unchanged baseline projections. They use the same hash-bound non-FFN references. The protocol acknowledges that K2/exact have different response shapes/ranges despite equal initial Jacobians. It also reserves budget129 and fresh holdout/transfer checks for a nominated lead. Thus a positive exact-versus-K2 result would concern higher-order gate shape, not isolated numerical order or an FM-specific theorem.

Independent verification command:

```text
/Users/mr.z/ode-workspace-prep/.venv-test/bin/python -m unittest \
  tests.test_forced_gate tests.test_forced_gate_integration -v
```

Result: **12 tests passed in 0.615 seconds**, including the actual MPS forward/backward/statistics test. No training or sweep was run by this reviewer. Full trainer/checkpoint/resume/sampling smoke verification, historical baseline trajectory reproduction and live monitor heartbeat are separate protocol requirements; this review does not mark them complete.

## Reviewed SHA256 values

```text
7f4a7defa6409d96bf7a94a5b50da6b8aca06b1bf55bfffd659e59589b59b428  synthetic/forced_gate.py
d5bf3e7e94ef651a279ac9a25f29d3bc01c1438a9e8d78a264365e6837d6221d  synthetic/models.py
9de06fda53f4abbc649c010c8484f427c53d4ce6164590863a8a5bdbf5d8e054  synthetic/diagnostics.py
490dee4e5f9eae844e97933b5e14f75df943f7f3a57d85300c99206f70052ad0  synthetic/train.py
2654c28088e823407b73d885abbfa6196a860fba3db903219c679585df2567ed  tests/test_forced_gate.py
f7df4f21acc8dc5a1caa5d8b47962e7979947e9a5fe46afe7d7e1df97740de01  tests/test_forced_gate_integration.py
0b7fe410888d2ea85691a6f83403fb789f8f9b33943b22e6ce6507988270d887  scripts/run_architecture_study.py
0247f815ccacd6c7694cb50f8175c696f839048b11f8c9a1241c0f4811514dc4  docs/research-architecture-search/protocol-forced-gate.md
7edf1aaf98cf309b190f3cc18076c1e8989a0003d2e54bd3f19e392bf9c1a646  configs/synthetic/architecture_search/forced_gate_v1_20261024_m1.json
c2beec932c76440f30c38e7482bbbfed5c9f5184f779d4b09cc5604caf91edd5  configs/synthetic/architecture_search/forced_gate_product16_20261024_m1.json
0c7889145f57d0edd53f07af0f362ec9ccf1510619a393394d79f33fef880376  configs/synthetic/architecture_search/forced_gate_queue_20261024_m1.json
```
