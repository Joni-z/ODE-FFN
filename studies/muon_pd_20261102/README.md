# Professor follow-up: Muon and PD

This isolated study reuses the audited vector-field trainer. It is **not** the
upstream JiT image trainer and cannot establish an image-generation advantage.
See the [registered protocol](../../docs/research-muon-pd-2026-11-02/protocol.md).

- `grid.json`: 36 fixed seed-0 cells; 20k updates each.
- `run.py`: model/optimizer adapter and additional spectral diagnostics.
- `preflight.py`: exact polynomial/derivative, pairing, native optimizer and
  physical checkpoint-resume witnesses, followed by a throughput measurement.
- `run_queue.py`: two-worker maximum, persistent logs/PIDs/resource samples,
  no automatic restart or overwrite. Failure stops dispatching new cells.
- `reference/`: four audited modules plus five unchanged transitive model
  dependencies. The reference edits are the optimizer-factory hook and CPU placement
  of restored RNG bytes, both recorded in `reference.patch`. Keeping the
  model, loss and integration logic unchanged avoids rewriting
  the established model; the grid exposes only the five specified families.
- `vendor/muon.py`: unchanged official SingleDeviceMuon; MIT license and pinned
  SHA in `source_provenance.json`.
- `data_identity.json`: exact archived manifests and NPZ hashes. Raw banks,
  checkpoints and execution logs stay outside Git.

PyTorch and NumPy are the only runtime Python dependencies. Reuse the verified
native environments; installing a different CUDA build is unnecessary.
The two optimizers have disjoint parameter groups. Actual update spectra are
captured after the optimizer; gradients are saved before Muon's in-place work.
Checkpoint resume restores both optimizers, RNG and nonpersistent buffers.

Commands and host paths are in
[validation.md](../../docs/research-muon-pd-2026-11-02/validation.md).
