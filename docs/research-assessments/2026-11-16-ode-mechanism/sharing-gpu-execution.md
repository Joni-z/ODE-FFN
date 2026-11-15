# Full-bank outer refinement on idle145GPU4

This executes P0 of [the mechanism follow-up](sharing-symmetry-followup.md).
It changes no weights, response, training seed or optimizer. The eight primary
sharing-cohort cases have completed independent evaluation. Four focal models
are exported with their exact CPU field and sample references, configurations,
manifests and hashes. Source publication and a fresh documented CUDA witness
precede the full inference job. This document alone is not a launch receipt.

Warm environment: `/data2/zz5070/miniconda3/envs/py312/bin/python`,
Torch2.8.0+cu128. No packages installed or rebuilt. Runtime root:
`/data2/zz5070/ODE_FFN/mechanism-sharing-20261116`.

Only physicalGPU4, UUID`GPU-6c644192-ce4a-5d10-b912-07a6fc24f352`, is visible.
Require no resident foreign compute process before launching. Physical0-3
are prohibited. Stop only our process if another task appears. Log occupancy,
power and utilization after each sampling resolution.

## Fresh documented witness

Execute exactly once from Mac mini after the source and portable files are
present on145 and the card has been checked empty:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=8 145 'cd /data2/zz5070/ODE_FFN/mechanism-sharing-20261116/repo && CUDA_VISIBLE_DEVICES=GPU-6c644192-ce4a-5d10-b912-07a6fc24f352 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python -u studies/muon_pd_20261102/coefficient_20261116/gpu_resolution.py --root /data2/zz5070/ODE_FFN/mechanism-sharing-20261116/portable --output /data2/zz5070/ODE_FFN/mechanism-sharing-20261116/doc-witness --gpu-uuid GPU-6c644192-ce4a-5d10-b912-07a6fc24f352 --witness'
```

Require all four cases, finite predictions, field portability at2e-5absolute
and relative tolerance, native64-source trajectory portability below1e-3,
exit0 and COMPLETE.json passed. The witness binds the portable manifest hash.
Do not replace a missing receipt with import success or partial progress.

## Full inference after readiness

Run `sharing_gpu.py` with the same environment, `--root` set to the portable
directory, `--readiness` to the witness COMPLETE.json, and `--output` to a
new `full-refinement` directory. Bind the same UUID and preserve frozen source.

Each of four models receives all32,768original confirmation sources at
Heun128/256/512. Heun128 must reproduce the completed CPU pooled score within
1e-4, otherwise stop without a cross-backend ranking. Save every generated
array, its hash, all eight bank scores and GPU telemetry. A separate NumPy
pass recomputes the12pooled and96bank scores. Report candidate differences
at every resolution; do not select the best solver setting afterwards.
