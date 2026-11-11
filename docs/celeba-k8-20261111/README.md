# CelebA ISRU-D K8: submitted runs

The selected ISRU-D K8 and native SwiGLU control are queued on torch under
`torch_pr_637_tandon_priority`, seed 0, unconditional 64x64 CelebA, full JiT-B/4.
The [protocol](protocol.md) fixes data, optimizer, budget and evaluation.

| Job | Purpose | Submission-time state |
|---|---|---|
| 22659499 | Full-model CUDA readiness, both FFNs, batch 64 | Pending: Priority |
| 22659500 | ISRU-D K8 training | Pending: afterok 22659499 |
| 22659501 | Native SwiGLU training | Pending: afterok 22659499 |

Each training job requests one GPU, four CPUs, 24 GB host RAM and two hours.
The untyped GPU request accepts `a100|h100|h200|l40s|rtx6000|b200` and omits
an explicit partition. The scheduler currently routes this priority account
to A100/H100 Tandon and L40S/H200 public resources. Listing a GPU feature does
not grant access to another account's stakeholder allocation.

The readiness job has a fifteen-minute limit; failed readiness cancels the
dependent training jobs. Successful submission is separate from a running
GPU kernel. See the [launch manifest](launch.json), [verification snapshot](verification.json)
and [documented checks](validation.md).

## Source and outputs

Training is frozen at `882c46c94ddeb14bab7704e416f754138121e2fe` in
`/scratch/zz5070/ODE_FFN/runtimes/celeba-k8-20261111-882c46c`.
The active development checkout can advance without changing these jobs.

- Candidate: `/scratch/zz5070/ODE_FFN/runs/celeba/20261111-b4-isru_d-s0`.
- Baseline: `/scratch/zz5070/ODE_FFN/runs/celeba/20261111-b4-swiglu-s0`.
- Slurm logs: `/scratch/zz5070/ODE_FFN/runs/celeba/slurm/`.
- Setup, live queue status and CUDA receipt:
  `/scratch/zz5070/ODE_FFN/operations/celeba-k8-20261111/`.

The queue monitor updates `monitor.json` every 60 seconds and exits after all
three jobs become terminal. It never restarts or submits work. GPU telemetry
is recorded every 30 seconds while each training job is active. Epoch records,
TensorBoard events, full atomic checkpoints and P-only history remain on
scratch. A clean two-hour allocation may end with progress `paused` before
the registered 200-epoch target; no automatic continuation chain is submitted.

The verified raw dataset remains on scratch. Each training job extracts it
into a job-owned node-local cache, reducing network small-file reads. The
temporary cache is removed at exit; all scientific outputs remain on scratch.
