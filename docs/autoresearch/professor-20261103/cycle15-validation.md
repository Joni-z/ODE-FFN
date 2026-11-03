# Cycle15 native transport readiness

Warm Lenovo synthetic environment. No installation or training. Check the
published clean detached source in professor-transport15-20261104/repo and a
freshly free GPU0 (<500MiB). Run the following command verbatim once. Keep
logs; stop on divergence. No repair, retry, common-reference preparation or
formal queue launch by the witness.

```bash
ssh lenovo 'set CUDA_VISIBLE_DEVICES=0&& set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& D:/Research/ODE-FFN/envs/synthetic/Scripts/python.exe D:/Research/ODE-FFN/professor-transport15-20261104/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle15/transport.py --device cuda --witness --data-root D:/Research/ODE-FFN/datasets/synthetic --output D:/Research/ODE-FFN/professor-transport15-20261104/preflight/documented'
```

Expected: exit0, passed=true, device=cuda, three nonzero-head model derivative
fixtures, exact native step and constant-field policy fixtures, validated
analytic-oracle Jacobians, and six real-checkpoint rows. Each real row must
replay4096 archived bank0/Heun128 outputs bitwise, match a full128step helper
rollout exactly, match degenerate all-learned policies, have finite full-field
Jacobians and retain all parameters/buffers. All six physical checkpoints and
original cached data are hash-bound. Stored checkpoints and training streams stay unchanged.

Windows Git: D:/Research/ODE-FFN/tools/git/cmd/git.exe. Record source HEAD,
cleanliness and capacity before and after. Original checkpoint/sample paths
are in the existing cycle14 cohort, not moved into the new worktree.

After the fresh documented pass, prepare one shared reference using the same
entrypoint with --prepare (output runtime/common). Then run unchanged
run_queue.py with two workers, cycle15/grid.json, cycle15/transport.py,
experiment id professor_transport15_20261104, output runtime/output. Persist
30second monitoring and check progress after SSH exits. Preserve source and
all failures; an ADAPTER_COMPLETE marks a frozen diagnostic, not training.
