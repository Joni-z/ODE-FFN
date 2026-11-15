# Frozen-weight resolution checks on145 GPU6

PhysicalGPU6 became idle during the investigation. This stage performs no
CUDA training: compare the already trained CPU weights under the same CUDA
evaluation backend, using the existing native sampler and sliced-Wasserstein
implementation. Physical0-3 are never selected. A runtime process check
refuses to begin/share a card with another compute process.

Environment: warm `/data2/zz5070/miniconda3/envs/py312/bin/python`,
Torch2.8.0+cu128. No install or rebuild. Portable assets include12hash-bound
CPU final checkpoints, exact configurations/manifests, confirmation arrays,
field predictions and generated samples. Training datasets are not copied.
Source is an isolated checkout of the canonical GitHub main, frozen before use.

The first witness compares512field predictions and64native Heun128 samples
per model against the actual CPU outputs. Fixed tolerances: field allclose
atol/rtol2e-5 and sample maximum absolute difference below1e-3. Failure stops
this stage; do not tune tolerances to the observed discrepancy.

Run verbatim:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=8 145 \
'CUDA_VISIBLE_DEVICES=GPU-8ffc220f-2b4f-5d14-0787-4f46ba75a023 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python /data2/zz5070/ODE_FFN/mechanism-gpu-20261116/repo/studies/muon_pd_20261102/coefficient_20261116/gpu_resolution.py --root /data2/zz5070/ODE_FFN/mechanism-gpu-20261116/portable --output /data2/zz5070/ODE_FFN/mechanism-gpu-20261116/doc-witness --witness'
```

Require COMPLETE.json passed=true,12portable model witnesses and exit0.
Then the same invocation without --witness writes a separate full output.
All eight4096banks are concatenated for efficient batch-separable inference.
The standard pooled score and each original bank score are retained.
Every original Heun128 score must agree with CPU pooled SW2 within1e-4.
Record full coordinate differences too. No CPU/CUDA training trajectories
are mixed in a learned-model ranking.

All six architectures per task are evaluated at outer Heun128/256/512.
State-flow weights also use internal4/8/16 at outer128 on all32768samples;
this strengthens the earlier bounded512sample refinement check. Save all
generated arrays, checkpoint/asset hashes, function-evaluation counts, timing
and GPU utilization history. Independent NumPy recomputation follows before
using these results. Later free-stage weights remain a separate registered
cohort until available; no missing arm is silently invented.

This checks numerical robustness and inference cost of synthetic velocity
models. It does not certify JiT image-model speed or image FID.
