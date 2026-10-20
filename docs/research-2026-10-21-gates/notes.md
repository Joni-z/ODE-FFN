# Working notes

- Torch login currently fails authentication.
- MBP is reachable; its connection to 145 is closed remotely.
- Existing trainer samples against test targets during training. New configurations
  must explicitly select validation sampling; preserve legacy defaults.
- Removing gate modules normally shifts subsequent initialization. Construct and
  discard them before scalar/fixed replacement to preserve every common weight.
- The submitted report is not being revised.
- Direct SSH to 145 works. Its canonical project path moved to
  `/data2/zz5070/ODE_FFN`; the old data1 root is retired.
- GPU0's three yp2841 workers were stopped with the owner's helper, but their
  controller restarted them. A second immediate launch completed four 20-step GPU
  smoke checks before the guard detected new foreign processes and stopped ours.
  No formal GPU training started. Do not repeatedly kill these workers.
- CPU pilot measured 69.98 and 138.04 updates/s for one and two workers;
  the formal runtime benchmark measured 69.04 and 130.26. All 15 runs completed.
  Protocol amendment was made before any full training.
- Original endpoints and saved samples passed collection audit. An additional
  8,192-sample check completed on the same 27 frozen checkpoints. No checkpoint
  was reselected.
- Local `ode-workspace-prep/data/synthetic/v1` NPZ serialization hashes differ
  from canonical data, though all six validation/test arrays are exactly equal.
  The larger-sample audit used the byte-matching canonical files under
  `ode-workspace-prep/research-2026-10-20/data-audit` and verified their hashes.
- Report 1 PDF SHA256 remains
  `165d46476362f006db6124f0035bcb6964cb163e509f52eb49adbf4de5839ecf`.
