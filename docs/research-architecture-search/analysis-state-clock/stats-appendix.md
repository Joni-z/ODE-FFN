# Stage21 numerical and audit appendix

The [posthoc analyzer](source/analyze_state_clock.py), SHA256 `8237c823638effa684ecbbebbb8bd826e8179c365d3a59844f78fb980fb8f608`, executed once after all 18 cells and the queue were COMPLETE. The separate endpoint replay helper is hash-pinned to `142bbe47e3ddf7f25befb1f0eda3b67d187122505805fcf94d4a5c62477640b7`. [Independent review](source/independent_review.py) recomputes the registered decision from raw terminal results and audits the historical Euler initialization.

## Exact endpoints

| Dataset | Arm | Oracle MSE | Five-time field MSE | Jacobian error | SW2 |
|---|---|---:|---:|---:|---:|
| gaussian_aniso_2d | state_euler | 0.00377154889107 | 0.00327892951942 | 0.0166377084493 | 0.0847856327891 |
| gaussian_aniso_2d | state_frozen | 0.00440309967995 | 0.00364670214135 | 0.0173684389087 | 0.0820977762341 |
| gaussian_aniso_2d | state_midpoint | 0.00433194348812 | 0.00363689254074 | 0.0169691602699 | 0.0817862376571 |
| gaussian_aniso_2d | swiglu_budget | 0.00418822109699 | 0.00387387602841 | 0.0286276567834 | 0.0750632882118 |
| gaussian_aniso_2d | swiglu | 0.00392725408077 | 0.00365914913011 | 0.0264225467812 | 0.0779446959496 |
| gaussian_aniso_2d | swiglu_compute | 0.00446846113205 | 0.00402529036377 | 0.0241118232065 | 0.073745585978 |
| gmm_ring8_2d | state_euler | 0.0199296356201 | 0.0165984545069 | 0.828581011851 | 0.234700635076 |
| gmm_ring8_2d | state_frozen | 0.0238427726746 | 0.0228391602901 | 0.99395732304 | 0.24697189033 |
| gmm_ring8_2d | state_midpoint | 0.023265255928 | 0.0218762001572 | 0.903356317639 | 0.248764082789 |
| gmm_ring8_2d | swiglu_budget | 0.0182577634811 | 0.0189212476448 | 0.508748077963 | 0.227947875857 |
| gmm_ring8_2d | swiglu | 0.0152962903976 | 0.0144475384296 | 0.624635515229 | 0.226360142231 |
| gmm_ring8_2d | swiglu_compute | 0.0165928561211 | 0.0150356968418 | 0.468551071924 | 0.227947250009 |
| rotated_product_bimodal_16d | state_euler | 0.266155200958 | 0.256487655958 | 50.0019372034 | 0.12064601481 |
| rotated_product_bimodal_16d | state_frozen | 0.301849081421 | 0.291611070782 | 55.0062348312 | 0.129229813814 |
| rotated_product_bimodal_16d | state_midpoint | 0.321165100098 | 0.310339791266 | 57.3012469259 | 0.136522710323 |
| rotated_product_bimodal_16d | swiglu_budget | 0.109160767746 | 0.106571702146 | 26.7538852564 | 0.0968339517713 |
| rotated_product_bimodal_16d | swiglu | 0.0923257699966 | 0.088920842745 | 23.9578347798 | 0.0957215353847 |
| rotated_product_bimodal_16d | swiglu_compute | 0.0586695946693 | 0.0564080568149 | 16.0550230256 | 0.0988497063518 |

Primary oracle MSE is the fixed uniform-time validation-stream componentwise squared velocity error. The field metric averages five common-time panels at .1/.3/.5/.7/.9. Jacobian error averages squared Frobenius error on those anchors; it is not divided by matrix dimension. SW2 is the recorded Heun32 validation statistic. These metrics are not interchangeable.

## Integrity and shared initialization

The frozen-native audits verified 108 checkpoints and 90 common diagnostic arrays, including identities, source/config/data hashes, analytic oracles and common anchors. All 18 final checkpoints replay saved predictions and input Jacobians with maximum difference exactly zero. No fresh sampling trajectory was reconstructed by the audit.

All 18 initial states match the hash-bound Stage15 non-FFN tensors and actual runtime buffers. Native FFN state and post-construction RNG are unchanged by that copy. The three ODE arms share P/controller tensors; all six final controller affines are zero. Frozen/midpoint initial functions and P/input derivatives match, while the nonzero-head controller-weight gradients differ. Euler's hidden function and output-head tangent are not paired with the K2 pair. Formal zero-head gradients and the K2 pair's initial output-head kernel are verified by preflight.

All five other arms have identical training data-generator states to the reference arm at six saved steps on each task: 90 exact comparisons. Both established SwiGLU baselines match all six historical Stage20 model, optimizer, runtime-buffer, data/Torch/NumPy/Python RNG states on three tasks: 36 complete saved-state comparisons. These are compatibility reruns, not independent repeated evidence. SwiGLU256 is a new compute-reference trajectory. Config identities and timing metadata are not expected to match across runtimes.

The Euler/history audit independently reads the Stage17 constant_paired configs and initial checkpoints. Initial model state, buffers and Torch/data RNG are bit-equal on all three tasks. The maximum final weight differences are 5.96e-7 for Gaussian, 2.36e-6 for Ring8 and 4.77e-7 for Product16. This records observed numerical differences without assigning an untested cause or claiming exact final ODE reproduction. The random-initialization distinction in the design discussion concerns the native constructor/professor proposal, not a new difference from the recent Stage17 parent.

## Preflight and source authority

The first preflight attempt passed 54 focused tests, 12 full-width float32/float64 cases, all 18 common-reference initializations, eight prior-model three-update trajectories and three prior forced-gate three-update trajectories against the explicit Stage20 source. All three new arms pass full-trainer three-step versus two-plus-resume-three equality for model, optimizer, buffers and RNG, with nonzero learned controller parameters and finite internal diagnostics/sampling. Both historical SwiGLU complete-trainer three-step comparisons are bit-identical.

Actual linear operations and controller calls were counted. Per solve, Euler executes one P and one controller, frozen K2 two P applications and one controller, midpoint two P applications and two controllers. The two-solve FFN therefore requires 16,384/24,576/32,768 dense MACs, respectively. All three share the same parameter count. Pointwise nonlinear operations are additional.

The frozen runtime contains 580 files inherited from the actual 571-file Stage20 source through explicit overlays. Both trees/hashes are reverified; all previous runtimes remain unchanged. Preflight summary SHA256 is `7765c28f52f6cd94c4e2e56c36887e06d51551ef5e59dabd12769d3412770559`. NumPy2.2.6, SciPy1.18.1 and psutil7.2.2 were checked. The [launch review](launch-review.json), [source manifest](runtime-source-manifest.json), and [archived receipts](review-preflight-receipts.json) bind the source and approval.

A metadata-only race occurred after root launched: a final dependency check briefly added two fields to ready.json, then restored the original bytes. The dependency evidence is now separate. The preservation request arrived after restoration, so the augmented copy is explicitly labeled a reconstruction of that append operation, not a captured earlier snapshot. The launch review binds source/preflight/code-review hashes, not ready.json; its command and frozen training code were unchanged. An output-empty assertion then encountered the already launched job, as expected. No preflight or formal training was restarted. The transparent [receipt](preflight-metadata-race-receipt.json) preserves this sequence.

## Internal data and execution

All 198 learning-curve rows, 594 layer/time summaries, 324 checkpoint/P-matrix summaries and 20,736 independently sorted singular/eigenvalue entries are retained. A spectral index is not a tracked mode or an eigenvector/singular-vector correspondence. Native-stage flags distinguish midpoint's executed controller update from Euler/frozen diagnostic-only probes. The raw P spectrum, conditional generators, clock derivatives and full velocity Jacobian are distinct objects.

The queue was launched once at 16:46:38UTC on October24 (local study date October25). v1 finished 12/12 at16:49:49.858, Product16 finished6/6 at16:51:44.757, and the queue completed16:51:48.302. This audit independently observed queue56284, Product16 supervisor56505 and workers56516/56600 alive at16:51:01. Prior v1 PIDs had already exited; they were not misreported as live. The v1 monitor has20 real samples, minimum available RAM4.8659GiB; Product16 has12, minimum4.9167GiB. Two one-thread CPU workers ran without reserving a GPU, with10s psutil telemetry and idle-sleep prevention.

No new labels, checkpoint selection, fresh holdout, isolated latency measurement, full-field stability proof or architecture success is claimed. The registered midpoint stop rule is preserved.
