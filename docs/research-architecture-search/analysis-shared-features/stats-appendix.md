# Stage17 audit and metric appendix

The posthoc script is [analyze_shared_features.py](source/analyze_shared_features.py). It imports the frozen Stage17 model and never calls a training launcher. Its SHA256 is `e60555a50fdee45ab5bdfc12bb48444cce83db739ad32faa0d8066f42fb11951`. The independent endpoint replay script is also preserved in [source](source/).

## Exact validation endpoints

| Dataset | Arm | Oracle MSE | Five-time field MSE | Jacobian squared Frobenius error | SW2 |
|---|---|---:|---:|---:|---:|
| gaussian_aniso_2d | vector_parent | 0.00377155036926 | 0.00327892917521 | 0.0166377091882 | 0.0847856327891 |
| gaussian_aniso_2d | vector_shared | 0.003735927701 | 0.00321924513866 | 0.0158394861518 | 0.0844446718693 |
| gaussian_aniso_2d | vector_untied64 | 0.00382160563469 | 0.00331724514047 | 0.0172667031342 | 0.0848042666912 |
| gaussian_aniso_2d | swiglu_budget | 0.00418822109699 | 0.00387387602841 | 0.0286276567834 | 0.0750632882118 |
| gaussian_aniso_2d | swiglu | 0.00392725408077 | 0.00365914913011 | 0.0264225467812 | 0.0779446959496 |
| gaussian_aniso_2d | swiglu_large | 0.0042083391428 | 0.00395376013754 | 0.0275577379152 | 0.0772035866976 |
| gmm_ring8_2d | vector_parent | 0.0199296352386 | 0.0165984579818 | 0.828581294732 | 0.234700664878 |
| gmm_ring8_2d | vector_shared | 0.0262030687332 | 0.0241660071186 | 1.15978727914 | 0.261507660151 |
| gmm_ring8_2d | vector_untied64 | 0.020539034462 | 0.0176360814645 | 0.71280072023 | 0.237490877509 |
| gmm_ring8_2d | swiglu_budget | 0.0182577634811 | 0.0189212476448 | 0.508748077963 | 0.227947875857 |
| gmm_ring8_2d | swiglu | 0.0152962903976 | 0.0144475384296 | 0.624635515229 | 0.226360142231 |
| gmm_ring8_2d | swiglu_large | 0.0154476243019 | 0.013817717844 | 0.509617060669 | 0.233545079827 |
| rotated_product_bimodal_16d | vector_parent | 0.266155192566 | 0.256487652025 | 50.0019370965 | 0.120646044612 |
| rotated_product_bimodal_16d | vector_shared | 0.315822078705 | 0.307056016052 | 56.6552352621 | 0.117499008775 |
| rotated_product_bimodal_16d | vector_untied64 | 0.252547953796 | 0.244331133881 | 47.8840891292 | 0.136012479663 |
| rotated_product_bimodal_16d | swiglu_budget | 0.109160767746 | 0.106571702146 | 26.7538852564 | 0.0968339517713 |
| rotated_product_bimodal_16d | swiglu | 0.0923257699966 | 0.088920842745 | 23.9578347798 | 0.0957215353847 |
| rotated_product_bimodal_16d | swiglu_large | 0.0704225543976 | 0.0669670786701 | 18.6077050661 | 0.0927088558674 |

Primary oracle MSE is the mean componentwise squared velocity error on the fixed uniform-time validation stream. The separate field diagnostic averages componentwise velocity error at t=0.1,0.3,0.5,0.7,0.9. Jacobian error averages the squared Frobenius norm over the same five-time anchors; it is not normalized per Jacobian entry. SW2 is recorded from the original Heun32 validation samples. These metrics should not be interchanged.

## Integrity and pairing

- Both child studies are COMPLETE. Native audits cover108 checkpoints and90 diagnostic arrays, including config, identity, data manifests, source, common anchors and independently recomputed analytic oracles. They also recompute endpoint primary oracle MSE.
- Reloading all18 final checkpoints reproduces saved predictions and Jacobians with maximum differences exactly0. Native audit and replay logs/results are retained for each child.
- All547 frozen source hashes match. Every one of18 step0 checkpoints has the hash-bound non-FFN state and runtime buffers, including nonpersistent Fourier frequencies. Each native FFN and post-construction Torch RNG matches reconstruction from its own config.
- Shared64 and untied64 have identical complete step0 state tensors. Their first feature weights equal P, first biases are zero, and final clock affines are zero. The three ODE arms have the same P draws. Common nonzero-head initial forward and input-Jacobian differences are0 on all three tasks; the declared Jacobian tolerance was2e-13.
- Each of54 ODE checkpoints (3arms x3tasks x6snapshots) was reloaded into the frozen model. Shared projection and first controller weight remain the same Parameter object; untied ones remain separate. Optimizer parameter lists match the deduplicated model parameter count, and shared state entries agree exactly.
- At every saved step0,1000,...,5000, the data RNG of each of the other five arms equals the parent:90 exact comparisons across the three tasks. This is a checkpoint-state check, not an independent replay of all training updates.
- P singular maxima and stable ranks in recorded diagnostic rows were checked against the corresponding saved matrices at every checkpoint. Other activation statistics are reported from the existing diagnostic logs; no new probe distribution was introduced.

## Recorded monitoring

| Study | Actual telemetry rows | Minimum available RAM (GiB) |
|---|---:|---:|
| shared_features_v1_20261024_m1 | 17 | 4.932266 |
| shared_features_product16_20261024_m1 | 10 | 4.855179 |

The parent observed initial workers48951/48952 live at13:40:08UTC. This audit independently observed queue48948 and supervisor48950 live around13:41UTC, with current child processes recorded by the real monitor. Those first workers had already exited. The terminal queue marker is13:44:18UTC. Monitoring is CPU psutil telemetry; no GPU was reserved. Terminal monitor files can retain the previous10-second process sample, so completion is established by status/COMPLETE markers and the audited checkpoints.

## Tooling record and scope

The first posthoc invocation used a relative analysis output path while the native audit ran with the frozen source as its working directory. It created two new audit artifacts under source/docs/.../analysis-shared-features/v1-native, then the endpoint replay failed to resolve its relative script path. No existing source, checkpoint, training result, or formal worker changed. The two artifacts were relocated to this analysis directory, the new empty directories removed, and the original547-file source tree and every hash reverified. The script now resolves both input and output paths before any subprocess. The audit was rerun successfully; no training was restarted and no numerical tolerance changed.

The seven shared-feature implementation tests and seven tangent-guard tests were run before formal training. Gate-context capture can change float64 derivative summation order at nonzero clock weights, covered by the predeclared2e-13 tolerance; forward values were exact. This is distinct from the native endpoint replay above, where observed derivative differences are0.

All statistical decisions are the registered engineering screen. No p-values or fresh-holdout claims are made. Existing validation evidence is exposed, and the synthetic direct-velocity backbone is not a JiT image experiment. No generated trajectory was replayed and no trained checkpoint was selected retrospectively. The registered failure is preserved without extending the same axis with posthoc settings.
