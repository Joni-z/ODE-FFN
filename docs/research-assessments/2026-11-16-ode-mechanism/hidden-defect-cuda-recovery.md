# Recover a lost terminal receipt without changing the numerical test

The author CUDAentry terminated0. The first fresh invocation produced passing
numerical outputs, but its agent-written post-run receipt wrapper failed with
TypeError: unsupported operand type(s) for /: 'str' and 'str'. The wrapper's
observed tool exit is1; the child's exit was not durably captured and remains
unknown. Its JSON/receipt and all1,218frozen source hashes are preserved. It is
not accepted as a completed readiness witness.

This separate recovery attempt repeats the SAME deterministic untrained test
with a new exclusive output. No mathematics, threshold, fixture, observer or
benchmark is changed. No learned checkpoint/data/sampling/training is involved.

Use the new frozen archive at
/data2/zz5070/ODE_FFN/hidden-defect-preparation-20261117/cuda-recovery-source/.
The parent source-freeze-cuda-recovery.json binds the GitHub commit and every
regular source file. The original c3529aacheckout and its freeze remain unchanged.
Use the existing hidden-defect-cuda@41450454environment, no package changes.

A new fresh agent verifies all manifested hashes before/after and executes
this command exactly once from that cwd:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python studies/muon_pd_20261102/target_transport_20261117/check_cuda_hidden_defect.py --output /data2/zz5070/ODE_FFN/hidden-defect-preparation-20261117/cuda-readiness-recovery.json
```

Run the remote command directly as the last command of its SSHshell; capture
the actual terminal status with the execution tool BEFORE writing a receipt
in a separate call. Do not put command execution and receipt-writing in one
Python wrapper. Record the tool session/chunk, actual exit, source/result hashes,
numerical/throughput results and any divergence in
../cuda-readiness-recovery-receipt.json. If exit75because no allowed card is
idle, preserve it and report; do not rerun or stop another job. Check the used
GPUcontext is released at termination. Preserve ordinary Python bytecode caches
as runtime artifacts. No source repairs or extra invocations.
