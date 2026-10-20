# Storage audit, 20 October 2026

Values below are allocated disk space from `du -x -B1`, converted using 1 GiB = 2^30 bytes. They cover the named directories; they are not a report of all files owned by the account anywhere on the host or a per-user quota.

## Account directories on host 145

| Directory | Bytes | GiB |
|---|---:|---:|
| `/data1/zz5070` | 9,722,560,512 | 9.05 |
| `/data2/zz5070` | 127,090,376,704 | 118.36 |
| `/home/zz5070` | 25,124,958,208 | 23.40 |
| Total audited personal directories | 161,937,895,424 | 150.82 |

Most of this usage belongs to other projects and existing environments. `/data2/zz5070/miniconda3` occupies 36.90 GiB; these environments are shared by the user's projects and are not newly created ODE-FFN storage.

The final filesystem snapshot during this audit showed 136.40 GiB available on `/data1`, 11.13 GiB on `/data2`, and 12.92 GiB on the root filesystem. These are shared, changing filesystem values, not reserved account capacity; earlier snapshots had less available space. No usable per-user quota report was obtained.

## ODE-FFN on torch

| Directory | Bytes | GiB |
|---|---:|---:|
| Entire `/scratch/zz5070/ODE_FFN` | 79,059,692,544 | 73.63 |
| Historical `jit-codebase/results` | 77,293,547,008 | 71.99 |
| All current `runs/synthetic` | 361,411,584 | 0.34 |

The entire historical project is much larger than the current synthetic workload. The historical image results remain on torch.

## This migration

The selected 18 checkpoints occupy 18,988,338 file bytes, and the three selected dataset directories occupy 2,885,164 file bytes. The transferred archive also includes the other small files in the prepared dataset version, checkpoint configurations, and the frozen evaluation runtime: 31,105,568 compressed bytes. The extracted ODE-FFN directory initially occupied 33,472,512 bytes (31.92 MiB).

The target is `/data1/zz5070/ODE_FFN/frozen_order_20261020`. It reuses the existing `py312` environment. A 1 GiB working budget is ample for this 66-cell diagnostic stage, including saved sample arrays, JSON records, and logs. This is a planning estimate for this stage, not a storage forecast for future image training or the whole research project.

No historical results or unrelated account files were deleted during this audit.
