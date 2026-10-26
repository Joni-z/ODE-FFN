# Audit and scope

The one authorized queue finished at 2026-10-24 17:33:12.788501 UTC: v1 completed 6/6 at 17:32:24.788893, Product16 completed 3/3 at 17:33:09.720057. Root closed launch session 26568 with exit 0. The analysis ran once after all terminal statuses, also exiting 0. No training or audit restart was needed.

| Evidence | Verified result |
|---|---|
| Frozen current source |587 files and exact tree; actual Stage21 parent 580 files unchanged |
| Preflight |58 tests; 9 common-reference initializations; 4 full-width float32/float64 GLU/Swi probes |
| Historical model compatibility |8 original, 3 forced-gate and 3 state-clock models: 3 updates each, exact states/optimizer trajectories |
| Complete trainer checks |GLU uninterrupted 3 versus 2+resume3 exact; both historical SwiGLU full trainers exact |
| Formal native audit |54 checkpoints, 45 arrays, all config/data/source bindings verified |
| Final endpoint replay |9 reloads; maximum prediction and Jacobian differences both 0 |
| Formal initialization |All 9 non-FFN references/buffers verified; GLU/Swi128 complete native initial tensors and RNG match |
| Historical formal baselines |36 saved states including optimizer, buffers and saved RNG exactly reproduce Stage21 |
| Cross-arm data RNG |36 snapshot comparisons, all equal |
| Curated outputs |9 endpoint rows, 99 learning-curve rows, 90 entropy rows, 3 higher-cost historical context rows |

The [independent check](independent-check.json) recomputes the decision from raw `results.json`, cross-checks the comparison table, verifies 181 analysis-input hashes and both exact source trees, and confirms all 12 recorded supervisor/worker PIDs have exited. Direct process enumeration ignores inaccessible system commands; explicit recorded PID checks are retained. An initial unrestricted process read encountered macOS AccessDenied on PID 1, then the bounded check completed; it did not affect training or numerical results.

Actual 10-second telemetry contains 7 v1 and 5 Product16 samples. The first recorded active v1 workers are 58317/58318 under supervisor 58316; Product16 begins with 58389/58390 under 58388 and later records 58400. Final monitor training subrows can lag the last checkpoint, so terminal statuses and native audits establish completion. The queue's saved PID equals the last child supervisor; it is not asserted to be the actual original queue PID. Wall times include concurrent training/evaluation and do not establish isolated inference speed or energy savings.

The immutable [launch review](qa/launch-review.json) binds ready, source manifest, preflight summary and independent code review. Dependency metadata was finalized before ready was exposed; ready was not changed after launch. [Archived QA](qa/) contains the exact receipts, source delta, preflight script, tests, monitor heartbeat and environment versions. No prior source, checkpoint, training output or Report1 was edited.

Regional calculations use all 512 anchors×16 coordinates at each of five times. The common orthogonal rotation and analytic oracle reproduce saved fields/Jacobians to 2.67e-15/6.67e-16, with MSE preserved to 5.56e-17. Empty bins retain zero contribution and no conditional mean. [Coordinate arrays](posterior-coordinate-errors.npz) permit the CSV to be checked without new model outputs. Fixed entropy masks describe target uncertainty, not learned gate probabilities or independent observations.

The screen is an engineering rule on exposed validation: 10% nonlinear primary improvement, no other primary regression above 5%, and nonlinear Jacobian/SW2 regressions within 10%, against the specified established baselines. It returns `STOP_GENERIC_SETTING`; no independent holdout or isolated timing study was spent on a rejected setting. No fresh-label test, image-model transfer, global capacity bound or ODE architectural success is claimed.
