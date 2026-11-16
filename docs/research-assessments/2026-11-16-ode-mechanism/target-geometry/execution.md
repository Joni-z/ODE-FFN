# Exact CPU execution

Use the unchanged environment response-screen-cpu@04f11f75 in
.aris/compute/macmini.md. No package installation, CUDA or MPS. One thread.
Protocol: [protocol.md](protocol.md). Source is frozen before the fresh pass at:
`/Users/mr.z/ode-workspace-prep/coefficient-20261116/target-geometry/repo`.

For the fresh agent-follows-doc witness, run the following three commands verbatim
in that frozen repository directory. Do not run the formal command in this pass.
Each output path must be absent before invocation. Report exits and divergences;
do not fix a failed command or replace an artifact.

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/oracle_check.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-geometry/readiness/oracle

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/frozen_probe.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-geometry/readiness/probe --witness

/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 studies/muon_pd_20261102/target_geometry_20261116/verify_probe.py --input /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-geometry/readiness/probe --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-geometry/readiness/verified
```

Expected: oracle PASS covers ten task/time cells and counterexamples; probe
COMPLETE has witness=true, twelve cases and twelve derivative rows; verified
COMPLETE has passed=true with twelve reconstructed rows. Witness geometry scores
are not the registered full observation and cannot decide its hypothesis.

Only after the fresh pass succeeds, the executor runs the formal observation in
the same frozen repository, saving stdout/stderr and its process identity:

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/frozen_probe.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-geometry/formal
```

Formal COMPLETE must have witness=false, twelve cases, sixty derivative rows,
128points per time and all five registered times. Independently verify with
verify_probe.py using `--input .../target-geometry/formal` and
`--output .../target-geometry/verified`. These observations evaluate existing
weights only. RUNNING.json identifies the process; progress.json counts actual
completed cases; COMPLETE.json and a zero process exit establish completion.
