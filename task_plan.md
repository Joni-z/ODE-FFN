# Current task: inspect the completed CelebA128 pair, 13 November 2026

The user asks for results. Verify training completion and checkpoint/source
identity; compare matched epoch losses, runtime, GPU telemetry and learned P.
Then run the registered checkpoint-only image/FID evaluation if absent.
Keep finalized reports, frozen source and existing training artifacts intact.

- [x] Query Slurm and final logs: both100epoch jobs completed0:0 on L40S gl066.
- [x] Collect exact loss/time/GPU records and validate complete checkpoint/P history.
- [x] Confirm no existing sampling/FID; register and submit evaluations22993694/22993695.
- [ ] Evaluations are running on A100 ga013/ga022; monitor PID3150165 polls60s. Inspect samples/FID when available.
- [x] Publish training figures, checkpoint identities, P analysis and evaluation launch receipts.

## Fixed comparison

Native22822267 versus K8 22822268; seed0, CelebA128, JiT-B/16, batch256,
LR5e-5, noise.5,100epochs. Training source65864ad remains frozen.
Primary generation uses the original50-interval Heun sampler, first EMA.9999,
50,000images and deterministic training-split crop reference. No new model,
optimizer, training seed or additional training is selected by this check.

Prior journals are preserved in [history](docs/history/README.md).
