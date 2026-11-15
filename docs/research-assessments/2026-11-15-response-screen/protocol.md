# Response-screen execution amendment

User authorized the [response-family comparison](../2026-11-14-next-architecture/response-screen-proposal.md)
and then selected Mac mini for these synthetic models. Use CPU, existing M1,
8cores/16GiB, warm Python3.12.4/Torch2.10.0/NumPy2.2.6 in
`/Users/mr.z/ode-workspace-prep/.venv-test`. No package installation.

All32 focal trajectories are trained on this same host, plus four unchanged
ordinary reference configurations. Historical CUDA weights remain diagnostic
context rather than mixed-backend selection controls. This replaces the
earlier reuse9/train23 layout with36 CPU trajectories. Only seed0 is used.

Use the same pinned native trainer,20k updates,batch512,AdamW3e-4 for focal
cases. The copied ordinary references retain their registered optimizer and
rate labels. Initial trainable tensors/buffers/RNG are matched across responses;
their initial functions differ. A single versus two worker native200step
benchmark chooses concurrency. No GPU allocation and no other user's jobs.
The user's older permission to stop yp2841 tasks is revoked.

Fresh independent selection/confirmation arrays are generated before formal
training with the streams and sizes specified in the proposal. Their manifest
is at `/Users/mr.z/ode-workspace-prep/response-screen-20261115/assets/manifest.json`.
Training pipelines read only selection arrays. Confirmation follows a frozen
shortlist under the registered rule; no automatic CelebA launch.

Use an isolated Git worktree, persistent queues and CPU-load/process histories.
Record native sampling identity, checkpoint hashes, full formula/input-gradient
checks including the Softsign nonsmooth caveat, exact8->16 continuations and
the real selection evaluator witness. All final outputs must be finite.
Native run_queue stops scheduling after an arm failure; diagnose its record
before deciding whether to continue remaining prespecified cells. Do not
change failed-family settings or silently retry it.

Raw outputs live outside Git under
`/Users/mr.z/ode-workspace-prep/response-screen-20261115/`.
Preserve Reports1/2 and all existing image baselines and checkpoints.
