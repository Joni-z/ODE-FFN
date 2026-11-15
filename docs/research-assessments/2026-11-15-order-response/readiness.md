# Native readiness on145

Reuse /data2/zz5070/miniconda3/envs/py312 (Torch2.8.0+cu128).
Recheck physicalGPU4 is idle. These commands are short witnesses, not formal
20k training. Run in the separate order-response checkout, without modifying
old worktrees or datasets. No package installation is needed.

```bash
cd /data2/zz5070/ODE_FFN/order-response-20261115/repo
export CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/data2/zz5070/miniconda3/envs/py312/bin/python studies/muon_pd_20261102/order_response_20261115/check_order.py --device cuda --data-root /data2/zz5070/ODE_FFN/professor-followup-20261115/data --output /data2/zz5070/ODE_FFN/order-response-20261115/preflight/documented-v2
```

Required: kernel.json passed, four exactK1 anchor replays, four native8->16
continuations and exact actual-evaluator round trips in PASS.json.

Then exercise term instrumentation and truncation on a real16step K8 checkpoint:

```bash
cd /data2/zz5070/ODE_FFN/order-response-20261115/repo
export CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/data2/zz5070/miniconda3/envs/py312/bin/python studies/muon_pd_20261102/order_response_20261115/diagnostics.py --folder /data2/zz5070/ODE_FFN/order-response-20261115/preflight/documented-v2/resume/order_response_20261115/gmm_ring8_2d/gmm_ring8_2d__pd_isru_k8__adamw_0.0003/seed000 --evaluation /data2/zz5070/ODE_FFN/order-response-20261115/preflight/unused-evaluation --data-root /data2/zz5070/ODE_FFN/professor-followup-20261115/data --output /data2/zz5070/ODE_FFN/order-response-20261115/preflight/diagnostic-v2 --device cuda --step 16 --witness
```

Required: COMPLETE.json passed, all parameters/buffers unchanged, orders
restored to8; term reconstruction matches the native forward exactly. Keep
failed attempts and register a new path after correction; never delete them.

Attempt1 failed before kernel dispatch because a historical module shared the
name check.py. The two reused helper modules are now imported by explicit
file path. No training occurred. The second command requires a passing first
command; do not run it if its input checkpoint does not exist.
