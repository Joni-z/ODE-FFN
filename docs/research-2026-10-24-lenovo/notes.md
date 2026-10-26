# Deployment notes

User authorized deployment, environment setup and continuation of experiments on Lenovo on 2026-10-24 (Asia/Shanghai).

No scheduler or shared-cluster allocation is needed for this user-owned Windows host. Source and results will live under a dedicated D-drive directory.

## Environment and persistence

- Windows11 Home, Ryzen9 7945HX (16 cores/32 threads), 15.69GiB RAM, RTX4060 Laptop 8188MiB, NVIDIA driver566.26.
- D drive initially615,528,075,264 bytes free (573.2GiB).
- Dedicated root `D:\Research\ODE-FFN`; managed Python3.12.13 and isolated torch2.8.0+cu126, NumPy1.26.4, Matplotlib3.10.9, psutil7.2.2. Package versions archived in environment-windows.txt.
- GPU tensor arithmetic verified.14 scientific tests passed on Windows;14+9 selected tests passed locally.
- Portable MinGit2.56.0.windows.3 installed under tools/git; SHA256 `2e5619966e588fde5f8122c5557cad50c0fbafcaed736689f40ec1dfb1d78ea7` matched the official Git-for-Windows release.
- Full Git history restored from a local bundle at base commit1cdb7f1, with the reviewed uncommitted research and continuation files copied explicitly. No claim that this working tree is a published GitHub commit.
- Direct SSH child-process launch was killed when the SSH command ended, before any scientific output existed. Switched to Task Scheduler `ODEFFN_clock_20261024`, signed-in-user context, and verified persistence from a separate command.

## Benchmark and execution

Kernel timings exclude30 warmup updates and measure200 updates per worker. Two concurrent workers use independent processes.

| Device | Workers | Aggregate updates/s |
|---|---:|---:|
| CPU |1|147.17|
| CPU |2|278.36|
| CUDA |1|123.08|
| CUDA |2|239.38|

Selected CPU with2 workers: about16% faster aggregate throughput than dual CUDA for this model/batch. The GPU environment remains available; no GPU allocation is held for the CPU training cells. New original/SwiGLU references are fresh matched-environment runs, not resumes of inaccessible145 weights. Old versus new numerical results must retain their environment/provenance labels.

## Staged decision

Native/SwiGLU reference cells completed first. The frozen native checkpoint diagnostic found the following relative RMS changes when deleting all centered curvature terms at s0=.6:

| Dataset | Field change | Jacobian change |
|---|---:|---:|
| Gaussian |1.97%|1.93%|
| Ring8 |10.10%|14.95%|

Both exceed the predeclared1% practical screening threshold; decision.json was written before launching the two gate-linearized training cells. This establishes a non-negligible frozen-weight effect, not a training advantage. The retrained control is necessary to test replaceability.
