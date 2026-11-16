# CUDA portability and throughput of the same ODE-defect observer

This prepares a device for the conditional learned experiment. It does not
load a checkpoint/data bank, train, generate samples or choose an architecture.
The observer from4a409d9 remains unchanged. CPU implementation controls and
their exact whole-model JVP/finite-difference validation are already complete.

Use the existing145 py312 environment. The environment JSON records its observed
Python3.12.11/Torch2.8.0+cu128/NumPy1.26.4 versions; it does not request a build or
installation. Bind its canonical content hash in the provider ledger. Kernel
and fresh documented checks are required before calling the observer CUDA-ready.

## Controls fixed before execution

- Choose only a physical GPU4-7 with<500MiB and no resident compute process.
  Bind its UUID before importing Torch. Reuse sharing_dispatch.eligible and
  gpu_resolution.exclusive_gpu; recheck exclusivity around every observation.
  Stop only our process if a card becomes occupied. Never touch0-3or another job.
- Construct the same seed0untrained2D/16Dfixtures on CPU with the nonzero head
  prescribed in hidden-defect-transport.md. Copy the actual tensors/buffers and
  inputs to CUDA; do not regenerate frequencies or initialize a second model.
- At16reference steps, compare complete-model CPU/CUDAprimal and tangent:
  component discrepancy<=1e-10+1e-9*abs(CPUreference). TF32disabled, FP64only.
- On CUDA, independently compare with whole-model epsilon JVP at0:
  error<=1e-11*(1+abs(reference)). Central differences at1e-4/1e-5must satisfy
  the original2e-7+2e-4*abs(reference) tangent gate.
- Zero forcing yields zero exactly; reversing solver order negates tangent
  within1e-12. Require nonzero tangent norm>1e-10. Parameters/buffers/solve
  methods remain unchanged and native replay is exact afterward.
- Fixed throughput probes on the untrained16Dfixture: B256and B4096,16steps,
  CPU then CUDA, one timed call per batch after the small controls warm both
  backends. Synchronize CUDA before/after timing. Compare every output with
  the same parity threshold. Keep transfers outside compute timing but report
  their separate elapsed time. These are timing observations, not repeated
  training or statistical estimates of model quality.
- Record the native GPU name/UUID, peak memory and actual per-batch times.
  Do not claim that a small-fixture pass numerically qualifies learned paths.

Output is exclusive: preserve failed attempts and use a new documented output
path after any repair. No source or gate changes to the running Mac cohort.

## Invocation

Use the sparse GitHub checkout at
/data2/zz5070/ODE_FFN/hidden-defect-preparation-20261117/repo/.
Before execution bind its actual commit and all checked-out tracked source
files in ../source-freeze-cuda.json. The helper module from4a409d9must have the
same hash. Python bytecode caches are runtime artifacts, not manifested source;
preserve any generated caches. Do not change packages or the running Mac study.

The author executes once:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python studies/muon_pd_20261102/target_transport_20261117/check_cuda_hidden_defect.py --output /data2/zz5070/ODE_FFN/hidden-defect-preparation-20261117/cuda-author.json
```

After that actual command terminates, a fresh agent checks the source manifest
before/after and executes only this command, verbatim once from the same cwd:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python studies/muon_pd_20261102/target_transport_20261117/check_cuda_hidden_defect.py --output /data2/zz5070/ODE_FFN/hidden-defect-preparation-20261117/cuda-readiness.json
```

Store actual terminal exit, source/result hashes and all divergences in
../cuda-readiness-receipt.json. If no permitted GPUis idle, preserve exit75and
wait; do not reclaim/share a card or improvise a new command. After completion,
check that the witness process has exited and its GPUcontext is released.
GPUreadiness is pending until this exact documented execution passes.
