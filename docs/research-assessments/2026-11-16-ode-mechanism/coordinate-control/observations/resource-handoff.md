# Measured Product16 observation handoff

The Product16 density observer is running on145physicalGPU7. Ring8 continues
on Mac CPU. This replaces our slower CPU observation using the same frozen
model, all source points, observer and numerical gates. It adds no training.

## Why change the resource route?

The original CPU route was selected while all permitted cards were occupied.
After new CUDA native-entry witnesses passed, the existing benchmark entry
measured the actual256-source Product16 workload with32Heun steps:

| Field | CPU seconds | CUDA seconds | Speedup |
|---|---:|---:|---:|
| Original |4.895|2.361|2.074x|
| Projected |31.067|2.153|14.431x|

Combined observer time is7.968xfaster. Maximum cross-host endpoint error is
1.87e-14relative; maximum log-ratio error per dimension is2.98e-13. Independent
NumPy reconstruction of the benchmark agrees within2.49e-14. The registered
handoff threshold was3xcombined speed and1e-7portability error. Benchmark
timings are routing evidence, not model-quality evidence.

## Execution and fallback

- Frozen source:fd7f09b0064ceab36a13f7757fd2716d002cbba6on both hosts.
- Density data:af2e0bd2ae01c76aef97e8ce5812d8234602f140123d0547575686f194e02577.
- Warm CUDAenvironment, unchanged packages. Fresh formal/native checks cover
  both new models; a separate Product16adaptive witness passes independent
  reconstruction within1.25e-13. Witness cells remain unqualified populations.
- Exact local worker43094, parent43084, was checked by UID, full command and
  process start before SIGSTOP. Its nine complete rows and hashes are saved.
  Ring8worker43093continues unchanged.
- The existing guarded dispatcher launched queue641894/worker641939on empty
  physicalGPU7(UUIDGPU-6bb30f73-bd71-ebef-4d8b-8dc9e075449b). Actual process
  identity, allocation and advancing integration were checked after launch.
- GPUformal output:
  `/data2/zz5070/ODE_FFN/mean-preserving-observations-20261116/formal-product16`.
  Local fallback and raw receipts:
  `/Users/mr.z/ode-workspace-prep/coefficient-20261116/contrast-observations`.

The GPU reruns the entire16-row case, not only the missing CPUrows. Keep the
CPUprocess suspended until all16GPUrows finish and pass native-law and
cross-host audits against every completed CPUrow. Only then terminate our
suspended worker and explicitly record its supersession. On replacement
failure, recover the exact suspended CPUstate. Never treat an observation
timeout as a failed process or a reason to launch another copy.

The CPUqueue will record an intentional nonzero exit when supersession is
completed; preserve that record. Require a completed, identity-matched GPU
replacement rather than accepting an arbitrary failed queue. No model/point,
checkpoint, response, optimizer, numerical threshold or time window is selected.
Physical0-3remain prohibited. No foreign task is stopped or shared.

## Evidence

[Benchmark and portability](routing-benchmark.json),
[CUDA formal readiness](cuda-readiness.json),
[independent adaptive witness](cuda-adaptive-independent.json),
[CPU identity and retained rows](migration-start.json),
[verified replacement process](migration-live.json).

This record establishes a live handoff, not final population qualification.

The first eight complete formal GPUrows now match the corresponding retained
CPUrows: relative endpoint error7.09e-14, log-ratio error per dimension1.31e-12,
other observer errors per dimension1.39e-12. This covers all original-field
banks at1024/2048steps. The ninth retained CPUrow is projected and remains
to be checked after its GPUcounterpart completes. The CPUfallback is held.
[Prefix-only cross-host audit](handoff-prefix-independent.json).
