# Stage 20 audit and numerical appendix

Posthoc analyzer: [analyze_forced_gate.py](source/analyze_forced_gate.py), SHA256 `162f4a477852e7292790f07d0785690b66f67d88c500b8dd4a735012f82d9846`. The trusted endpoint replay helper is archived beside it with SHA256 `142bbe47e3ddf7f25befb1f0eda3b67d187122505805fcf94d4a5c62477640b7`. The independently written [screen calculation](source/check_forced_gate_screen.py) uses the terminal raw results rather than importing the analyzer's decision function; both produce STOP_FAMILY.

## Exact endpoints

| Dataset | Arm | Oracle MSE | Five-time field MSE | Jacobian error | SW2 |
|---|---|---:|---:|---:|---:|
| gaussian_aniso_2d | swiglu_budget | 0.00418822109699 | 0.00387387602841 | 0.0286276567834 | 0.0750632882118 |
| gaussian_aniso_2d | swiglu | 0.00392725408077 | 0.00365914913011 | 0.0264225467812 | 0.0779446959496 |
| gaussian_aniso_2d | forced_static | 0.00392728631496 | 0.00365886804318 | 0.0264496866894 | 0.0779456049204 |
| gaussian_aniso_2d | forced_k2 | 0.00392478864193 | 0.00365693459231 | 0.026472854944 | 0.0779752880335 |
| gaussian_aniso_2d | forced_exact | 0.00392464072704 | 0.0036569276847 | 0.0264779047721 | 0.0779787749052 |
| gmm_ring8_2d | swiglu_budget | 0.0182577634811 | 0.0189212476448 | 0.508748077963 | 0.227947875857 |
| gmm_ring8_2d | swiglu | 0.0152962903976 | 0.0144475384296 | 0.624635515229 | 0.226360142231 |
| gmm_ring8_2d | forced_static | 0.0152962000847 | 0.0144173135849 | 0.619462487661 | 0.226720958948 |
| gmm_ring8_2d | forced_k2 | 0.0154010988235 | 0.0145278339989 | 0.611716757784 | 0.227902218699 |
| gmm_ring8_2d | forced_exact | 0.015408045578 | 0.0145346539232 | 0.611652585061 | 0.228010371327 |
| rotated_product_bimodal_16d | swiglu_budget | 0.109160767746 | 0.106571702146 | 26.7538852564 | 0.0968339517713 |
| rotated_product_bimodal_16d | swiglu | 0.0923257699966 | 0.088920842745 | 23.9578347798 | 0.0957215353847 |
| rotated_product_bimodal_16d | forced_static | 0.0923963935852 | 0.0890375943939 | 23.9772296156 | 0.0956621095538 |
| rotated_product_bimodal_16d | forced_k2 | 0.092542307663 | 0.0892058559465 | 24.0087976968 | 0.0958093926311 |
| rotated_product_bimodal_16d | forced_exact | 0.0925520744324 | 0.0892149219138 | 24.0113891396 | 0.0957846492529 |

Primary oracle MSE averages componentwise squared velocity error on the fixed uniform-time validation stream. The separate field metric averages five common panels at t=0.1,0.3,0.5,0.7,0.9. Jacobian error is squared Frobenius error averaged on those anchors, not normalized per Jacobian entry. SW2 is the saved Heun 32 validation sampling statistic.

## Integrity and initialization

Both child studies are COMPLETE:10 v1 cells and 5 Product 16 cells. Frozen-native audits checked 90 checkpoints and 75 diagnostic arrays, including identities, data/source/config hashes, common anchors and exact analytic oracles. All 15 final checkpoints replay saved field predictions and Jacobians with maximum differences exactly 0. No new generated sampling trajectories were evaluated by the audit.

All 15 initial checkpoints match the hash-bound Stage 15 non-FFN tensors and actual runtime buffers. Native FFN tensors and post-construction Torch RNG are preserved. The three new arms match SwiGLU 128's projection weights, biases and RNG exactly, with only three additional zero eta vectors. The formal states satisfy the conditions of the archived full-width nonzero-head function/common-gradient preflight checks. Exact and K2 have matching complete initial parameter Jacobians; ordinary SwiGLU lacks eta. Adding eta can change whole-model clipping even when the common raw gradients match, so no identical baseline optimizer trajectory is claimed for the modified arms.

All four non-reference arms share the narrower SwiGLU's data RNG at all six saved steps:72 exact checks. Both ordinary SwiGLU baselines reproduce the historical Stage 19 model, optimizer, buffers, data RNG, Torch RNG, NumPy RNG and Python RNG exactly at step 0,1000,...,5000:36 complete saved-state comparisons. These are deterministic compatibility reruns, not independent repeats. Dataset-manifest hashes and training/evaluation settings also match. Config/source identities and timing metadata differ as expected.

The runtime contains 571 frozen files and inherits the 561-file Stage 19 base through an explicit overlay manifest. Both manifests are verified, and the formal source is unchanged. [runtime-source-manifest.json](runtime-source-manifest.json) records every hash; the separate [launch review](launch-review.json) binds the approved preflight and source. Analysis writes only outside the frozen runtime.

## Preflight corrections and authority

Only corrected attempt 3 is authoritative. Preflight attempt 1 stopped because the old MAC helper fed float32 inputs into a double model. The harness was changed to instrument actual linear calls with the model's dtype; training source was not changed. Attempt 2 then completed but its predecessor path pointed to itself, so its historical-compatibility conclusion was explicitly invalidated. Attempt 3 asserts a different predecessor, verifies the actual 561-file Stage 19 manifest, and reruns all assigned checks. Its summary SHA256 is `dbf3346457d1f805e9d9881043809221d5a0246d7174f2e6b1600f528baa6ab5`.

Attempt 3 passed 42 focused tests,16 full-width initialization probes,15 common-backbone checks, eight old-model three-update comparisons, three forced-arm full-trainer 3 versus 2+resume 3 state comparisons with nonzero eta, both historical SwiGLU trainer comparisons, actual parameter/MAC counts, worker-config interception and a real psutil heartbeat. Attempts 1/2 remain in the preflight directory; their mistakes were harness failures before formal launch, not failed training cells. [The archived harness](source/preflight_forced_gate.py) and [receipts](review-preflight-receipts.json) identify the valid run.

The first posthoc analyzer invocation stopped before native audits because it expected a standalone replay helper inside the frozen training source. Its traceback is preserved in analysis-attempt1-path-error.log. The corrected analyzer uses the hash-verified standalone helper; the second invocation completed all audits. No formal job was restarted and no formal result was overwritten.

## Stored internal measurements

The bundle includes 165 metric-curve rows,297 gate/layer/time rows,25,245 full eta/lambda channel rows and 270 saved checkpoint/block projection summaries. Raw eta/lambda vectors match checkpoints exactly; lambda=tanh(eta) also passes a cross-precision check with atol 8e-8 and rtol 2e-7. The latter accounts for float32 versus float64 transcendental evaluation, not trajectory mismatch.

P=diag(lambda), the conditional rate c=lambda*sigmoid(a), the forcing SiLU(a), relative response gain and gate derivative are logged separately. Conditional/input statistics summarize the existing fixed probes; individual per-input matrices are not stored. Gain/correction statistics are not new validation evaluations. Gate/value/output spectral norms and drift are recomputed from saved matrices in double precision. Static absorbed output columns use phi1(.5lambda), an algebraic change with no retraining. These norms do not bound the complete velocity Jacobian.

The smooth phi1 implementation verifies the identity value, first derivative 1/2 and second derivative 1/3 at zero, with series handling below|c|=.05. The code review documents float32 quotient second-derivative cancellation near the switch (observed absolute error 4.80249e-5); this is separate from exact trajectory/endpoint replay and does not establish an explanation of the quality result. Ordinary first-derivative training and the registered dtype checks passed.

## Monitoring and execution

The root submitted one queue at 15:29:45UTC. This audit independently observed queue 53003, v1 supervisor 53012 and active workers 53087/53162 at 15:31:08UTC; the exact process/telemetry sample is preserved in independent-process-observation.json. v1 completed at 15:32:28, Product 16 at 15:34:01 and the queue at 15:34:06UTC. Both children used two one-thread CPU workers, actual 10s psutil telemetry and no reserved GPU. The v1 monitor contains 17 samples with minimum available RAM 4.8262GiB; Product 16 contains 10 with minimum 4.7940GiB. A final telemetry sample can precede the terminal status and is not relabeled as a terminal process observation.

Wall times include training-loop evaluations and diagnostics under concurrent execution. Dense MACs exclude added sigmoid/tanh/phi1 pointwise work. No isolated speed, fresh holdout, checkpoint selection, generated-path replay or architecture success is claimed. The negative result follows the registered thresholds and is retained without a rescue sweep.
