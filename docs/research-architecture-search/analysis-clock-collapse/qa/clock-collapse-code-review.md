# Clock-collapse pre-execution review

2026-10-25. **PASS for the registered read-only execution.** No architecture or training is selected. This review inspected the stable script before its formal run, reran its data-free self-test independently, and verified all 595 predecessor source hashes. No new diagnostic outcomes were accessed or produced during this review.

The command `/Users/mr.z/ode-workspace-prep/.venv-test/bin/python scripts/diagnose_clock_collapse.py --self-test` exited 0 with PASS (0.89 seconds including interpreter startup). It checks native float32 capture, a fixed vector per layer, independent folded-matrix outputs/input derivatives, finite downstream changes, exception-safe patch restoration, and risk signs/units. This toy self-test is not the required independent replay of all fifteen real panels.

## Implementation checks

- **Actual runtime:** the script verifies the pinned Stage23 manifest and all 595 file digests, inserts that runtime before importing the model/trainer/Jacobian helper, and asserts those module origins. CLI execution starts a fresh process; a conflicting preloaded module causes an assertion rather than silent use. Config and checkpoint identities, data manifest, probe/validation hashes, and native array provenance are checked. The actual parent configs have width64, depth3 and controller hidden32. Their generic `spectral_cap=1` option is inactive for `ode_vector`; the instantiated-layer guard correctly requires `spectral_cap is None`.
- **Calibration:** only float32 `probe['xt'][:256]` and `probe['t'][:256]` enter one native forward. Each of six clocks is captured exactly once, with shape256×64. The six64-dimensional means are fixed across every time/input. Oracle quantities enter later evaluation only. Overall means are checked against the native final logs with explicit float32 aggregate comparison tolerance (`atol=5e-7`, `rtol=1e-6`), not bit identity; the current replay and native run are both CPU. Full durations, means and calibration inputs are retained and hashed.
- **Finite complete function:** replacing `step_size` by a globally constant vector changes the layer map and therefore every downstream state. External time conditioning stays live. The checked parent is homogeneous K1, after-projection, map-returning, multiplicative, independent and unforced. Its fixed map is `I + diag(sbar) P`, with row scaling. This differs from the old `check_gate_geometry.gate_context`, which held sample-wise native clocks and preserved every anchor prediction while changing derivatives.
- **Precision and differentiation:** the separately restored evaluation model receives the checkpoint's native float32 weights and runtime buffers before conversion to float64. Native output and full input Jacobian must replay saved arrays before collapse. The Jacobian helper differentiates the whole model; it does not reuse anchored derivatives. All rows are independent in this backbone, so summing one output coordinate over a batch yields the per-example input derivatives. Parameters, buffers, training flags, forcing modes and temporary clock patches are checked unchanged after intervention.
- **Arithmetic and coverage:** field risk averages squared error across inputs and coordinates. Jacobian risk sums both matrix axes before averaging inputs. `delta=collapsed-native` gives `risk_change=2*mean(residual*delta)+mean(delta²)` in the corresponding units; both signs and the zero-change case are tested. All five fixed times, three datasets and equal-time means remain visible. The script claims neither primary random-time validation nor sampling, retraining or fresh-holdout results.

## Required result-stage check

The script intentionally does **not** claim that its clock patch independently verifies itself. After execution, a separate implementation must replace all six solves with folded matrices and reproduce the fifteen collapsed fields/Jacobians, while independently recomputing panel and aggregate arithmetic. The summary explicitly leaves this check outstanding. Results should not be treated as fully audited before it passes. No additional knob, oracle fitting or trained comparison follows automatically from either sign of the result.

## Reviewed hashes

| Artifact | SHA256 |
|---|---|
| `scripts/diagnose_clock_collapse.py` | `3294b970074cadf2fac182c6f9ef6c05fe7d4dc7ea802b19221bdde7cee5fa75` |
| `protocol-clock-collapse.md` | `df6eb527a0f6bd82ab5660251b878f834e9027c38e099e654686f9c1dc221411` |
| `newmath-reassessment-after-bias.md` | `97d47d495af1fb69d2fb0e6dc8894dddaa40c38e5433c914ed580fdef9c50635` |
| Stage23 `source-manifest.json` (595 files) | `47e833e60e63c5fffb820ce5939f15ee9f595ba5c5fdc6e5a906ea39fa0dbb5c` |
| Frozen `synthetic/models.py` | `9fe326f400862597ac28a392b797f51b406955fdfaf5d373eed3907e71caf337` |
| Frozen `synthetic/train.py` | `7164d98ea2bd870c8d4fcc98b48d8e9b8938dfa1ac33fa69f708fbdc9766f90b` |
| Frozen `synthetic/diagnostics.py` | `11ecc22b36132680780b52cdfa42e18f8cfb57e60d06a3c47a112ae1a13762c6` |
| Frozen/current `scripts/check_gate_geometry.py` | `d466ef319b26a8967ea913df335fb7f4405b3dc89e9bd7f02632def839628978` |
| Earlier `gate-placement-readout.md` | `5624f9906c8d8fe790fb50e1aa13fa6ce5a0606ad83431f7227f559b4893e4ce` |

The reviewed predecessor is `/Users/mr.z/ode-workspace-prep/architecture-search/affine_forcing_20261025_m1/source`. This review changes only this new review file; frozen source, model code, protocol and prior evidence remain untouched.
