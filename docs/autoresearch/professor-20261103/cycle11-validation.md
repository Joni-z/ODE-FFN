# Cycle11 native algebra, zero-anchor and actual-entry readiness

Warm reuse of the existing environments; no package or model-backbone change.
Use freeGPU0 only, verifying source cleanliness and HEAD before execution.
Lenovo identity checks require D:/Research/ODE-FFN/tools/git/cmd/git.exe;
no global PATH edit or new Git installation is needed.

Each command runs once in a new output directory. Verify one new-function row,
all six zero-anchor layer comparisons, paired tensors/buffers/RNG, rank32
factorization, explicit cubic map and complete derivatives. The native float32
controller fold is checked in double precision within recorded rounding error,
not asserted bitwise equal to the nonlinear controller at every input.

Each host then tests its five actual entries:8->16resume exactly equals
uninterrupted16 for all five; two references also equal their original8-step
states. Require all ten parent/cache versions and eighteen archived-corner
source bindings. Preserve failures; no formal20k launch before both fresh passes.

```bash
ssh 145 'CUDA_VISIBLE_DEVICES=GPU-5d6078be-3b77-0dfd-0e6b-9cb486e62ff4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python /data2/zz5070/ODE_FFN/professor-autoresearch-cycle11-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle11/check11.py --device cuda --output /data2/zz5070/ODE_FFN/professor-autoresearch-cycle11-20261103/audit/native-witness.json --data-root /data2/zz5070/ODE_FFN/muon-pd-20261102/data --dataset rotated_product_bimodal_16d --smoke-output /data2/zz5070/ODE_FFN/professor-autoresearch-cycle11-20261103/preflight/documented'
```

```bash
ssh lenovo 'set CUDA_VISIBLE_DEVICES=0&& set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& set PATH=D:\Research\ODE-FFN\tools\git\cmd;%PATH%&& D:/Research/ODE-FFN/envs/synthetic/Scripts/python.exe D:/Research/ODE-FFN/professor-autoresearch-cycle11-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle11/check11.py --device cuda --output D:/Research/ODE-FFN/professor-autoresearch-cycle11-20261103/audit/native-witness.json --data-root D:/Research/ODE-FFN/datasets/synthetic --dataset gmm_ring8_2d --smoke-output D:/Research/ODE-FFN/professor-autoresearch-cycle11-20261103/preflight/documented'
```

Pass: exit0, passed=true, CUDA device, one function row/six anchor layers,
five exact native resumes and two parent8references perhost. Record clean
source commit and grid/trial/check SHA256. This is implementation readiness,
not positive evidence for the new corner.
