# Exact CPU readiness invocation for ordinary population controls

Use the unchanged warm response-screen-cpu@04f11f75 environment. The provider
ledger is `.aris/compute/macmini.md` in the canonical checkout. Do not install
packages, edit source or start the formal dispatcher during this check.
The runtime below is a frozen Git archive with `source-freeze.json` at its
parent. Verify every listed regular-file hash before and after these commands.

Run from:

```text
/Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/ordinary-population/runtime/repo
```

Execute these five commands exactly, sequentially, and retain their actual
terminal exits. If any fails, stop and report the exact error without repair.
Fresh output paths must not already exist. The first command replays all four
historical native field/sample anchors, exercises both 128/2048 samplers on both
streams, and times one formal 4,096-source bank. The two real producer commands
cover low-dimensional Free ISRU and high-dimensional wide SwiGLU; independent
NumPyentries reconstruct all four rows each. Witness samples are two fixed
sources, not evidence of generation quality.

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python -u studies/muon_pd_20261102/target_transport_20261117/ordinary_population.py --mode check --binding /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/ordinary-population/inputs.json --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/ordinary-population/fresh/readiness.json

env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python -u studies/muon_pd_20261102/target_transport_20261117/ordinary_population.py --mode run --binding /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/ordinary-population/inputs.json --readiness /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/ordinary-population/fresh/readiness.json --index 0 --witness --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/ordinary-population/fresh/index0

env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python -u studies/muon_pd_20261102/target_transport_20261117/verify_ordinary_population.py --binding /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/ordinary-population/inputs.json --input /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/ordinary-population/fresh/index0 --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/ordinary-population/fresh/index0/independent.json

env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python -u studies/muon_pd_20261102/target_transport_20261117/ordinary_population.py --mode run --binding /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/ordinary-population/inputs.json --readiness /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/ordinary-population/fresh/readiness.json --index 3 --witness --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/ordinary-population/fresh/index3

env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python -u studies/muon_pd_20261102/target_transport_20261117/verify_ordinary_population.py --binding /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/ordinary-population/inputs.json --input /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/ordinary-population/fresh/index3 --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/ordinary-population/fresh/index3/independent.json
```

Return the source commit/manifest hash, all actual exit codes, input/readiness
and two completion/independent hashes, anchor/recording results, throughput,
memory, and any doc-versus-reality divergence. Source, old checkpoints and
existing live queues must remain unchanged. New bytecode caches may be
reported separately; they are not edits to the manifested source files.

Only after this fresh pass may the owner launch the formal two-worker CPU
dispatcher from this same frozen runtime, using its `fresh/readiness.json`.
It retains both meshes, all banks, both streams and independent verifiers.
The corresponding formal command is not part of the five-command witness:

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python -u studies/muon_pd_20261102/target_transport_20261117/ordinary_population.py --mode dispatch --binding /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/ordinary-population/inputs.json --readiness /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/ordinary-population/fresh/readiness.json --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/ordinary-population/formal
```
