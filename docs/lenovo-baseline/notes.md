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
