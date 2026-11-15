# Frozen sharing/symmetry intervention

The hypothesis, eight models and decisions are fixed in
[the follow-up](../sharing-symmetry-followup.md). No training or model selection.
Four new banks use stream2026112200; each has4096base source/target points
and eight group transforms. All evaluations use float64, native Heun128,
and the trained four hidden substeps. Product16's reflection subgroup flips
latent coordinates0,7,15; it covers only eight of the65536coordinate reflections.

The CPU native-reference, group and actual-entry checks pass. The linear C8
check retains both the trace and antisymmetric terms of a general2x2matrix.
The field/orbit MSE split is independently checked. Both bounded entry
witnesses independently recompute within1.12e-16. Failed preflight folders
remain preserved and are not experiment results.

## Environment and paths

Reuse `/data2/zz5070/miniconda3/envs/py312/bin/python`, Torch2.8.0+cu128;
no packages rebuilt or installed. Runtime:
`/data2/zz5070/ODE_FFN/mechanism-sharing-symmetry-20261116`.
Its `repo/` is a clean frozen worktree; `portable/` contains eight immutable
checkpoints, original configurations, manifests, fresh arrays and CPU float64
references. Local raw root:
`/Users/mr.z/ode-workspace-prep/coefficient-20261116/sharing-symmetry`.

Physical145GPUs0-3are prohibited. The dispatcher selects only physical4-7
with no resident compute process and less than500MiB used. It reserves each
selected UUID in its own queue before CUDA initialization, and exposes only
that UUID to its child. Children recheck occupancy between stages. Another
task appearing causes our own task to stop; no other task is touched.

## Fresh documented CUDA invocation

After publishing/syncing source and staging the portable data, a fresh agent
executes this exact command once:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=8 145 'cd /data2/zz5070/ODE_FFN/mechanism-sharing-symmetry-20261116/repo && /data2/zz5070/miniconda3/envs/py312/bin/python -u studies/muon_pd_20261102/sharing_symmetry_20261116/run_symmetry_queue.py --root /data2/zz5070/ODE_FFN/mechanism-sharing-symmetry-20261116/portable --output /data2/zz5070/ODE_FFN/mechanism-sharing-symmetry-20261116/doc-witness --witness'
```

Require all eight native field references within2e-9and64-source trajectories
within1e-7, seeded group/oracle/projection tests, two actual evaluator-entry
witnesses, unchanged parameters, exit0and COMPLETE.json passed. A learned
sampling shortcut is used only if its16-base-point comparison to actual
orbit integration is below1e-9. A failed shortcut retains its error and uses
direct projected integration; its tolerance is never relaxed.

## Full queue after readiness

Run the same dispatcher with `--readiness` pointing to the witness COMPLETE,
`--output` a new `runs/` directory, `--workers 4 --detach`, and no `--witness`.
It starts at most one process per currently idle permitted card. Status,
live PIDs, pending cases and resource snapshots are saved every10seconds.
Failures preserve partial outputs and stop additional launches; no retry.

Each of eight cases writes four original and four projected sample arrays,
two pooled scores, group-pulled field predictions and their MSE split, two
recorded trajectories, and both128/256step checks on512base points. Four-bank
scores use16384independent base points and131072correlated group points.
The numerical subsets are not quality rankings. Sampling uses the original
native Heun implementation; SW2 uses the original native function.

Completion requires eight successful process exits, all case COMPLETE files
with witness=false, a successful queue receipt, and an independent NumPy
recomputation of64bank and16pooled scores, field decompositions, gap masses,
mode counts, flux matrices and the32bounded-resolution scores. Source,
checkpoint and array hashes are checked before interpreting any ranking.
