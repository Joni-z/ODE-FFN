# Order-by-response study, 15 November 2026

Authorized: complete the K1/K8 x ISRU/tanh comparison on Ring8 and Product16.
Use 145 physical GPUs 4-7 only; seed 0. Reuse existing K1 and ordinary controls.
Preserve finalized reports, native trainers, old runs and image baseline.

- [x] Verify resources and read source-bound K1 protocol/results.
- [x] Register four K8 runs and bind historical K1 checkpoints/evaluations.
- [x] Implement order-only adapter, shared evaluator bridge and term diagnostics.
- [x] Verify explicit Taylor powers, full entrance-diagonal Jacobian (dim2), unchanged initialization and native8->16 continuation.
- [x] Fresh documented witness; push/freeze source; launch four monitored pipelines.
- [x] Complete native/C8 sampling,20k order truncation and layer-term diagnostics.
- [x] Independently recompute metrics; compare response-by-order effects and ordinary controls.
- [x] Publish findings and next decision; record open research questions.

GPU4 Ring8/ISRU-K8; GPU5 Ring8/tanh-K8;
GPU6 Product16/ISRU-K8; GPU7 Product16/tanh-K8.
Same20k updates, batch512, AdamW3e-4, eight common4096sample banks and Heun128.
Equal parameters and updates; K8 uses eight P applications per internal solve.
No new optimizer sweep, scaling, spectral caps or rotation augmentation.

Current: four pipelines completed at frozen9f72bd3, all exits zero, GPUs idle.
NumPy recomputation passed (max sampled-metric difference2.8e-17).
Figures inspected. Findings and evidence published to GitHub in2639792.
The four-run study is complete; the proposed rotation-training control and
an ODE-specific quality/cost advantage remain open. Source sync is performed
through GitHub; the145 experiment worktree remains frozen at9f72bd3.

Readiness1 failed at helper import: historical check.py shadowed response
check.py. Explicit-path imports fix the wrapper; attempt2 uses new output
paths and preserves the original failure. Both documented checks then passed;
four real K1 sampling replays, four exact continuations/evaluator round trips,
and unchanged parameters through frozen-order diagnosis were verified.
