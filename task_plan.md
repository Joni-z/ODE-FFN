# Order-by-response study, 15 November 2026

Authorized: complete the K1/K8 x ISRU/tanh comparison on Ring8 and Product16.
Use 145 physical GPUs 4-7 only; seed 0. Reuse existing K1 and ordinary controls.
Preserve finalized reports, native trainers, old runs and image baseline.

- [x] Verify resources and read source-bound K1 protocol/results.
- [ ] Register four K8 runs and bind historical K1 checkpoints/evaluations.
- [ ] Implement order-only adapter, shared evaluator bridge and term diagnostics.
- [ ] Verify explicit Taylor powers, full entrance-diagonal Jacobian, unchanged initialization and native8->16 continuation.
- [ ] Fresh documented witness; push/freeze source; launch four monitored pipelines.
- [ ] Complete native/C8 sampling,20k order truncation and layer-term diagnostics.
- [ ] Independently recompute metrics; compare response-by-order effects and ordinary controls.
- [ ] Publish findings and next decision; record open research questions.

GPU4 Ring8/ISRU-K8; GPU5 Ring8/tanh-K8;
GPU6 Product16/ISRU-K8; GPU7 Product16/tanh-K8.
Same20k updates, batch512, AdamW3e-4, eight common4096sample banks and Heun128.
Equal parameters and updates; K8 uses eight P applications per internal solve.
No new optimizer sweep, scaling, spectral caps or rotation augmentation.

Current: protocol and implementation. No new training submitted yet.
