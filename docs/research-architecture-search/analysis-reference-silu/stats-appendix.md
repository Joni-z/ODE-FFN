# SiLU reference audit appendix

Analysis source: [analyze_reference_silu.py](source/analyze_reference_silu.py), SHA256 `d2990cc847142bf0c6648d83daca34dbda71d14977eb7c40ab3e0119330cc798`. Endpoint replay source is preserved beside it. The frozen runtime contains the exact547 Stage17 files plus four new protocol/plan files; no implementation file changed.

## Exact endpoints

| Dataset | Arm | Oracle MSE | Five-time field MSE | Jacobian error | SW2 | Parameters | Dense MACs |
|---|---|---:|---:|---:|---:|---:|---:|
| gaussian_aniso_2d | vector_parent | 0.00377155036926 | 0.00327892917521 | 0.0166377091882 | 0.0847856327891 | 81858 | 80192 |
| gaussian_aniso_2d | vector_shared | 0.003735927701 | 0.00321924513866 | 0.0158394861518 | 0.0844446718693 | 82050 | 80192 |
| gaussian_aniso_2d | vector_untied64 | 0.00382160563469 | 0.00331724514047 | 0.0172667031342 | 0.0848042666912 | 106626 | 104768 |
| gaussian_aniso_2d | swiglu_budget | 0.00418822109699 | 0.00387387602841 | 0.0286276567834 | 0.0750632882118 | 69570 | 67904 |
| gaussian_aniso_2d | swiglu | 0.00392725408077 | 0.00365914913011 | 0.0264225467812 | 0.0779446959496 | 81792 | 80000 |
| gaussian_aniso_2d | swiglu_large | 0.0042083391428 | 0.00395376013754 | 0.0275577379152 | 0.0772035866976 | 106818 | 104768 |
| gaussian_aniso_2d | ffn_silu | 0.00438411016464 | 0.00363367085209 | 0.0201349624589 | 0.0687680393457 | 81858 | 80192 |
| gmm_ring8_2d | vector_parent | 0.0199296352386 | 0.0165984579818 | 0.828581294732 | 0.234700664878 | 81858 | 80192 |
| gmm_ring8_2d | vector_shared | 0.0262030687332 | 0.0241660071186 | 1.15978727914 | 0.261507660151 | 82050 | 80192 |
| gmm_ring8_2d | vector_untied64 | 0.020539034462 | 0.0176360814645 | 0.71280072023 | 0.237490877509 | 106626 | 104768 |
| gmm_ring8_2d | swiglu_budget | 0.0182577634811 | 0.0189212476448 | 0.508748077963 | 0.227947875857 | 69570 | 67904 |
| gmm_ring8_2d | swiglu | 0.0152962903976 | 0.0144475384296 | 0.624635515229 | 0.226360142231 | 81792 | 80000 |
| gmm_ring8_2d | swiglu_large | 0.0154476243019 | 0.013817717844 | 0.509617060669 | 0.233545079827 | 106818 | 104768 |
| gmm_ring8_2d | ffn_silu | 0.0337137849808 | 0.0256498793529 | 1.26057313865 | 0.272278785706 | 81858 | 80192 |
| rotated_product_bimodal_16d | vector_parent | 0.266155192566 | 0.256487652025 | 50.0019370965 | 0.120646044612 | 83664 | 81984 |
| rotated_product_bimodal_16d | vector_shared | 0.315822078705 | 0.307056016052 | 56.6552352621 | 0.117499008775 | 83856 | 81984 |
| rotated_product_bimodal_16d | vector_untied64 | 0.252547953796 | 0.244331133881 | 47.8840891292 | 0.136012479663 | 108432 | 106560 |
| rotated_product_bimodal_16d | swiglu_budget | 0.109160767746 | 0.106571702146 | 26.7538852564 | 0.0968339517713 | 71376 | 69696 |
| rotated_product_bimodal_16d | swiglu | 0.0923257699966 | 0.088920842745 | 23.9578347798 | 0.0957215353847 | 83598 | 81792 |
| rotated_product_bimodal_16d | swiglu_large | 0.0704225543976 | 0.0669670786701 | 18.6077050661 | 0.0927088558674 | 108624 | 106560 |
| rotated_product_bimodal_16d | ffn_silu | 0.153892194366 | 0.148797616428 | 33.5775288574 | 0.099677875638 | 83664 | 81984 |

Primary oracle MSE averages componentwise velocity squared error over the fixed uniform-time validation stream. The separate field MSE averages five fixed-time panels at0.1,0.3,0.5,0.7,0.9. Jacobian error is squared Frobenius error averaged over those anchors, not divided by the number of Jacobian entries. SW2 comes from the registered Heun32 validation evaluation.

## Source, initialization and historical reuse

- All551 new-runtime file hashes match; all547 inherited files are byte-identical to immutable Stage17. New files are one preregistered protocol and three plans. Runtime source hashes and exact overlays are recorded in [runtime-source-manifest.json](runtime-source-manifest.json).
- The existing common-backbone helper copies22 non-FFN state tensors and the actual time-frequency buffer in each new run from the same Stage15 step0 reference as every Stage17 arm. Native FFN tensors and post-construction Torch RNG are unchanged. There is no trained-weight transfer.
- Each native plain FFN has16576 parameters and16384 dense MACs. Executed linear-operation instrumentation agrees with generic accounting. Full model counts exactly match vector_parent at both data dimensions.
- The preflight checked identical surrounding initialization against all18 historical controls, the unchanged eight-model default trajectory check plus the plain FFN, and a three-step full-trainer run versus a two-step run resumed to three. All state/optimizer/runtime/RNG comparisons were bit-identical. These are implementation checks, not extra formal experiments.
- New and historical configs agree on training steps, batch size, learning rate, weight decay, clipping, all evaluation/sample/checkpoint intervals, sampling count/directions/solver, width/depth and common-backbone references. Intended FFN choices remain different.
- All18 new snapshots were audited for identity, config, manifest, endpoint completion, source and checkpoint consistency. Their data RNG matches every one of the six historical arms at each saved step:108 exact comparisons. This does not claim regeneration of all training updates.
- Native audits independently recomputed analytic oracles on15 arrays and endpoint primary oracle MSE. Three endpoint model reloads reproduced both predictions and input Jacobians with observed maximum difference0. Historical audit/replay artifacts were reused with input hashes retained in [provenance.json](provenance.json).

## Launch and telemetry

Exactly one formal queue launch was made, in exec session93048, after preflight passed. The v1 supervisor50101 recorded training workers50102/50103 live at14:11:19UTC, with both runs at update2000. Product16 supervisor50124 recorded worker50125 live at14:11:44UTC, at update2500. The queue reached COMPLETE at14:11:53UTC. Minimum recorded available memory was5.055GiB for v1 and5.187GiB for product16. No GPU was allocated.

The direct post-launch psutil inspection occurred at14:12:16UTC, after all work had ended. It therefore observed exited processes, not live training. The live-PID evidence above comes from actual monitor telemetry; [launch-evidence.json](launch-evidence.json) preserves that distinction. No timeout or exited process triggered a restart.

The plain SiLU recorded training times were18.016s on Gaussian2D,17.626s on Ring8 and15.985s on Product16, including in-loop validation. Product16 ran alone whereas the historical controls used two workers. These are contextual wall times, not an isolated throughput comparison.

## Boundaries

This is a fixed-protocol reference comparison, with no new architecture nomination or significance threshold. Anchors, coordinates, time panels and snapshots are not independent experiment repetitions. Existing validation data are exposed; this audit introduces no fresh holdout, new sampling paths, or model selection. The complete mixed outcome is retained: plain SiLU improves Product16 and worsens Ring8 relative to ODE. No general superiority theorem for either family follows.
