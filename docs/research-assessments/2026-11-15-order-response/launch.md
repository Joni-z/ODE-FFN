# Four-GPU launch

Source is frozen in `/data2/zz5070/ODE_FFN/order-response-20261115/repo`.
The original five-run wave and its data remain in separate directories.
Readiness must pass before executing the template below for GPUs4,5,6,7.

```bash
cd /data2/zz5070/ODE_FFN/order-response-20261115/repo
GPU=4
CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python studies/muon_pd_20261102/run_queue.py --host 145 --device cuda --data-root /data2/zz5070/ODE_FFN/professor-followup-20261115/data --output /data2/zz5070/ODE_FFN/order-response-20261115/gpu$GPU --workers 1 --grid studies/muon_pd_20261102/order_response_20261115/gpu$GPU.json --runner studies/muon_pd_20261102/order_response_20261115/pipeline_order.py --experiment-id order_response_20261115 --detach
```

GPU4 Ring8 ISRU-K8; GPU5 Ring8 tanh-K8; GPU6 Product16 ISRU-K8;
GPU7 Product16 tanh-K8. The source-pinned native trainer executes20k updates.
Each queue automatically performs the common-bank evaluation and20k truncation
and term diagnostics. No restart on failure. The existing monitor records
processes, steps, metric snapshots and GPU use every30seconds. Pipeline phase
records distinguish training, frozen evaluation and term diagnostics.

Completion requires zero exit, native ADAPTER_COMPLETE, evaluator and
diagnostic COMPLETE.json passed, pipeline phase complete, and queue-finished
failed=false with no unstarted cells. Monitor locks are retained as records.
