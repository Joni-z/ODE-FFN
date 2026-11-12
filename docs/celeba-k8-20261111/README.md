# CelebA ISRU-D K8: first allocation finished

The selected ISRU-D K8 and native SwiGLU control ran on torch under
`torch_pr_637_tandon_priority`, seed 0, unconditional 64x64 CelebA, full JiT-B/4.
The [protocol](protocol.md) fixes data, optimizer, budget and evaluation.
Both jobs paused cleanly at their first allocation's budget. No training is
currently running. See the [12 November status](status-20261112.md) for losses,
throughput, checkpoints and the telemetry correction. No FID has been run.

| Job | Purpose | Verified state, 12 November |
|---|---|---|
| 22659499 | Full-model CUDA readiness, both FFNs, batch 64 | Passed, H100 |
| 22659500 | ISRU-D K8 training | Completed 0:0; paused at 28/200 epochs |
| 22659501 | Native SwiGLU training | Completed 0:0; paused at 44/200 epochs |

Each training job requests one GPU, four CPUs, 24 GB host RAM and two hours.
The untyped GPU request accepts `a100|h100|h200|l40s|rtx6000|b200` and omits
an explicit partition. The scheduler currently routes this priority account
to A100/H100 Tandon and L40S/H200 public resources. Listing a GPU feature does
not grant access to another account's stakeholder allocation.

The readiness job passed in 3 min 24 s. The training allocations lasted 1 h 48 min 30 s
and 1 h 46 min 50 s, respectively. The original [launch manifest](launch.json) and
[submission-time verification](verification.json) retain their historical
states; the [new status snapshot](status-20261112.json) records completed jobs.

## Source and outputs

Training is frozen at `882c46c94ddeb14bab7704e416f754138121e2fe` in
`/scratch/zz5070/ODE_FFN/runtimes/celeba-k8-20261111-882c46c`.
The active development checkout can advance without changing these jobs.

- Candidate: `/scratch/zz5070/ODE_FFN/runs/celeba/20261111-b4-isru_d-s0`.
- Baseline: `/scratch/zz5070/ODE_FFN/runs/celeba/20261111-b4-swiglu-s0`.
- Slurm logs: `/scratch/zz5070/ODE_FFN/runs/celeba/slurm/`.
- Setup, live queue status and CUDA receipt:
  `/scratch/zz5070/ODE_FFN/operations/celeba-k8-20261111/`.

The queue monitor reached the terminal states and exited. It never restarted
or submitted work. The first allocation's GPU CSVs are empty because of a
launcher command-concatenation bug; the corrected launcher starts its collector
separately and checks its process. Epoch records,
TensorBoard events, full atomic checkpoints and P-only history remain on
scratch. A clean two-hour allocation may end with progress `paused` before
the registered 200-epoch target; no automatic continuation chain is submitted.

The verified raw dataset remains on scratch. Each training job extracts it
into a job-owned node-local cache, reducing network small-file reads. The
temporary cache is removed at exit; all scientific outputs remain on scratch.
