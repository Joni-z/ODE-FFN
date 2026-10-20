# Gate necessity study

This follow-up keeps Report 1 unchanged. See [protocol.md](protocol.md) for
the predeclared comparison and [mathematical-rationale.md](mathematical-rationale.md)
for what the gate changes.

| Model | Parameters | Forward linear MACs/example |
|---|---:|---:|
| Original input gate, K=2 | 82,056 | 105,152 |
| Fixed duration, K=2 | 56,706 | 80,192 |
| Learned scalar duration, K=2 | 56,712 | 80,192 |
| SiLU | 81,858 | 80,192 |
| SwiGLU | 81,792 | 80,000 |

MACs exclude nonlinear operations, backward, optimizer and evaluation.
Gate removal reduces capacity and compute; these are not parameter-matched models.

## Execution

In a separate immutable checkout, link `data/synthetic/v1` to the existing prepared
dataset. Expose exactly one allocated/free GPU through `CUDA_VISIBLE_DEVICES`,
preferably using its UUID, then run:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/run_gate_study.py --output /absolute/new/output/directory
```

The runner freezes source/config/data hashes, benchmarks one and two concurrent
training processes, and chooses two only when aggregate throughput improves by
at least 10%. Each benchmark worker warms up for 100 updates and measures 1,000
updates using the real Ring8 training data. It then trains 15 models and evaluates
the fixed and validation-selected endpoints on test data.

`status.json` records the current phase or failure; `monitor.json` records each
run's last evaluation step and completion. `telemetry.jsonl` retains GPU readings
and progress every 10 seconds. Logs are in `logs/`; checkpoints and trajectory
diagnostics in `runs/`; endpoint samples in `samples/`; full Jacobian diagnostics
and scores in `evaluations/` and `summary.json`.

For CPU execution, pass `--device cpu`. The runner hides CUDA from every child
and uses at most two single-threaded training workers. This is the fallback
registered before full training when the GPU owner's controller repeatedly
reclaimed the card. All models, including both baselines, use the same backend.

An existing output directory is rejected. An interrupted launch requires inspection
before any restart; the runner never silently overwrites or resubmits it.

## Verification

Seventeen local unit/integration tests passed before GPU preflight, covering
legacy behavior, common-weight pairing, scalar gradients, checkpoint restoration,
constant-duration absorption into P, validation/test separation, endpoint selection
and existing intervention behavior. Short test runs validate software, not sample
quality. GPU smoke tests and measured utilization are recorded separately.
