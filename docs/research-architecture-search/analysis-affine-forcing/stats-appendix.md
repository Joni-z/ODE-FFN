# Verification and interpretation

One formal launch occurred at 2026-10-24 17:57:58.6 UTC. Actual queue PID 60655, caffeinate 60663 and first supervisor 60664 were captured live in [the launch observation](qa/launch-process-observation.json), with a separate [independent process observation](independent-live-observation.json). v1 finished 12/12 at 18:00:39.266435, Product16 finished 6/6 at 18:02:10.681278, and the queue completed at 18:02:14.934056 UTC. Root closed session 88144 with exit 0. The analyzer then ran once, session 6179, also exiting 0. No job or analysis restart occurred.

| Check | Result |
|---|---|
| Frozen source |595 files/exact tree; actual Stage22 parent 587 files unchanged |
| Preflight |65 tests; 18 common-reference initializations; 4 complete float32/float64 initial-response probes |
| Compatibility paths |16 executions across 15 distinct settings; the constant-paired vector setting occurs twice |
| Complete trainer |Both bias arms uninterrupted 3 versus 2+resume3 exact; all three historical generic trainers exact |
| Actual formal dimensions |Width 64, depth 3, gate-hidden 32; projection/controller call and dense-MAC counts verified |
| Native audit |108 checkpoints and 90 arrays, with config/data/source bindings |
| Final native replays |18; maximum prediction/Jacobian discrepancies both 0 |
| Generic history |54 saved model/optimizer/buffer/RNG states exactly reproduce Stage22 |
| Cross-arm data RNG |90 comparisons, all equal |
| Trajectories |198 learning rows, 594 layer-stat records, 324 P/b checkpoint records |
| Channel data |20,736 sorted P-spectrum entries and 13,824 bias-coordinate values |
| Fixed-weight intervention |15 panels plus 3 equal-time aggregate rows, all independently checked |

The compatibility count is clarified here without altering the frozen receipts: the eight original checks and eight additional paths are 16 executions, but one parameter setting overlaps. This changes no test outcome. The full-width probes used the formal gate-hidden 32 setting, not the constructor default 64. They measured zero initial forward, input-Jacobian and static/conditional parameter-response differences under nonzero receiving heads. The parent lacks the added bias parameters and is not asserted to share the entire tangent kernel.

The [independent raw screen](independent-check.json) verifies 279 input hashes, both exact source trees, launch bindings and the raw endpoint decision. It confirms 23 recorded queue/supervisor/worker PIDs have exited. Monitor files preserve actual 10-second samples; stale monitor subrows are not treated as active jobs after terminal checkpoint audits. Training used two one-thread CPU workers with idle-sleep prevention and no GPU reservation. Recorded wall times include concurrent training and evaluations, and are not isolated latency measurements.

For the [independent intervention](independent-intervention/receipt.json), a reviewer restored three cold float64 models from the actual frozen source and recorded nonpersistent buffers. All six mode attributes stayed conditional. Forward hooks subtracted `(D(input)-cI)b`; `vmap(jacrev(...))` recomputed complete per-example input Jacobians through all downstream changes. Maximum native field/Jacobian discrepancies were 0/2.043e-14; altered field/Jacobian discrepancies were 5.329e-15/2.774e-14. Independent CSV arithmetic agreed within 2.558e-13 across all 18 rows. Input reconstruction, checkpoint/config/data/array hashes, and unchanged weights/buffers/modes also passed. The [checker source](independent-intervention/check_intervention.py) and [recomputed rows](independent-intervention/recomputed-panel-metrics.csv) are archived.

The alteration uses exposed five-time validation anchors. Its field error is not the primary uniform-time oracle metric; no new sampling result, independently trained model or pure state-variation causal effect is claimed. It removes mean and varying components together. Internal P spectra and `diag(b)J_s` concern specific local terms, not full-network stability or an approximation lower bound.

The [protocol](source/protocol-affine-forcing.md) requires both a practical frontier gain and a separate affine/static contrast before attributing conditional-bias evidence. Both fail. No fresh holdout or isolated speed study was spent on rejected settings. Submitted Report1, previous runtimes and raw training outputs remain unchanged; the original architecture goal remains open.
