# Matched order: native CPU numerical witness

The four formal20k trajectories and all common-bank evaluations are complete.
The sampling-only portable export passes native full-field/bank0replay and
all four checkpoint restores. Use unchanged response-screen-cpu@04f11f75 in
`.aris/compute/macmini.md`; no package changes. Physical145GPUs4-7have resident
foreign contexts at the latest inspection, so CPUcan proceed independently.

From `/Users/mr.z/ODE-FFN-upstream`, execute the following FOUR commands
verbatim and sequentially. Every output directory must be absent beforehand.
These are bounded512-source actual-entry checks, not population results.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/coefficient_20261116/cpu_resolution.py --root /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order/portable/sampling-models --case gmm_ring8_2d__palindrome_operator_midpoint4__adamw_0.0003 --mode outer-subset --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order/numerical-witness/gmm_ring8_2d__palindrome_operator_midpoint4__adamw_0.0003
```

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/coefficient_20261116/cpu_resolution.py --root /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order/portable/sampling-models --case gmm_ring8_2d__alternating_operator_midpoint4__adamw_0.0003 --mode outer-subset --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order/numerical-witness/gmm_ring8_2d__alternating_operator_midpoint4__adamw_0.0003
```

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/coefficient_20261116/cpu_resolution.py --root /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order/portable/sampling-models --case rotated_product_bimodal_16d__palindrome_operator_midpoint4__adamw_0.0003 --mode outer-subset --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order/numerical-witness/rotated_product_bimodal_16d__palindrome_operator_midpoint4__adamw_0.0003
```

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/coefficient_20261116/cpu_resolution.py --root /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order/portable/sampling-models --case rotated_product_bimodal_16d__alternating_operator_midpoint4__adamw_0.0003 --mode outer-subset --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order/numerical-witness/rotated_product_bimodal_16d__alternating_operator_midpoint4__adamw_0.0003
```

Require all four exits0, four `COMPLETE.json` records,512sources and four outer
resolutions64/128/256/512per case. Each original checkpoint/asset hash must match
the portable manifest. No tensor or array is changed. Independently recompute
all sixteen SW2values from saved arrays and bind this witness before the full run.

After publication, use a separate frozen observer worktree and the existing
`resolution_queue.py --only numerical --full-outer --inner-substeps 4 8 16`.
Four models x two modes yield eight monitored CPUtasks, two workers. Each
mode retains all32768sources; the two modes repeat their common4/128anchor.
There are twenty unique model/resolution cells, with every bank retained.
Do not confuse the bounded witness with the full results. Keep CPUrunning
while a potential GPUroute merely waits; migration requires a live advancing
GPUchild and a complete replacement audit before stopping CPUwork.
