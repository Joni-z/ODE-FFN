# Lenovo deployment notes

- 1 November 2026: SSH alias `lenovo` resolves and `hostname` returns `Lenovo`.
- Active source is `/Users/mr.z/ODE-FFN-upstream`, initially commit `8c3b377`.
- User authorized training on the restored host. GitHub main is the sync source.
- Next: inspect actual resources and use a new runtime without overwriting any
  earlier Windows environment, checkpoint or source snapshot.

## Host inspection
- RTX 4060 Laptop, 8188 MiB VRAM, 0% GPU load, no Python training process.
- D: free 600,379,768,832 bytes; physical memory about 16 GiB.
- Old `repo/` contains uncommitted edits: leave it intact. New code path is
  `D:/Research/ODE-FFN/upstream`; old environments and runtimes stay intact.
- The unsigned managed Python is now blocked by Windows application control.
  Use a PSF-signed official Python installer; no security settings are changed.
- Direct GitHub clone and Python download stalled. Stop only those own staging
  processes, preserve their partial artifacts, and relay the published Git
  history and official installer through Mac mini. Verify hashes on arrival.
- Initial Mac installer download ended with curl18; resume the same official
  URL and verify the completed installer signature before installation.

## Necessary source fixes
- Replace the crop lambda with a picklable `functools.partial` so Windows
  DataLoader workers can start. The image operation is unchanged.
- Call `DistributedSampler.set_epoch` in single-process mode as well, so a
  single-GPU run does not repeat the same sample ordering every epoch.

## Staging resolution
- Official Python3.12.10 installer SHA256
  `67b5635e80ea51072b87941312d00ec8927c4db9ba18938f7ad2d27b328b95fb`;
  Windows reports a valid Python Software Foundation signature. Installer exit0;
  new venv Python also has a valid signature and executes normally.
- CIFAR archive SHA256
  `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce` matches
  on arrival; extracted without modifying the old synthetic dataset.
- Torch wheel2.5.1+cu124 official-index SHA256
  `3c3f705fb125edbd77f9579fa11a138c56af8968a10fc95834cdd9fdf4f1f1a6` verified
  after a resumed range download. No alternate package or mirror substituted.
- Direct `clone -b origin/main` from a bundle failed because it names a remote
  ref, not a head. Explicit Git init/fetch/checkout succeeded; source8c3b377 then
  fast-forwarded to the published6643a53. The failed direct clone is archived.
- First offline auxiliary install stopped before installation because the
  cross-platform resolver omitted Windows-only colorama. Add its pinned wheel;
  do not bypass dependency checks.
- Large transfers were slow through the shared SSH master. Independent SCP
  with a131072-byte buffer passed16MiB in4.27s;262144 exceeded the server message
  limit. Relay official verified chunks and verify the final Windows wheel hash.
- Mac compatibility checks pass after both loader fixes and optional epoch-wise
  PNG retention. Native objective, gradients, both EMAs and sampler comparisons
  remain exact. These are correctness checks, not image-quality evidence.

## Capacity decision
- All packages installed, pip check exit0, official torch2.5.1+cu124 recognizes
  the RTX4060 and BF16. No Windows security policy changed.
- Full B model:129,881,868 parameters. Batch16:7.380GiB peak allocated,
  7.764GiB reserved,1.190s/update. Batch8:5.292GiB allocated,5.645GiB reserved,
  0.323s/update. Each is8 real-CIFAR updates including startup; not a definitive
  steady-state speed comparison. Select batch8 for memory headroom.
- Limit CIFAR Inception batches to the generation batch size (at most64) to
  fit alongside the trained model/Adam/twoEMAs. No FID definition changes.
- Both probes check active gradients, EMA updates and short native sampling;
  neither establishes generation quality. Fresh documented validation remains.

## Launch verification
- Fresh documented commands pass once each, four exit0 results. CUDA peak is
  5.292 GiB; official Inception dispatch returns4x2048 features.
- Scheduled task is running from detached3b07a57. Initial logs show growing
  steps0,20,40,...,560; steady0.19s/update, GPU99%, about6.2GiB resident.
- A combined read-only status query stalled; its own SSH probe was stopped.
  Individual status/launch JSON transfers succeeded and the monitor heartbeat
  advances. No training process was restarted. The first curator attempted to
  read that pending file before transfer completion and exited without edits;
  the completed JSON transfers were used instead. All12launch source hashes
  match the reviewed source. Reports remain unchanged.
