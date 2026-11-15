# Mac mini CPU queue

Training runtime:
`/Users/mr.z/ode-workspace-prep/response-screen-20261115/repo`, b5fe7aa.
Analysis runtime:
`/Users/mr.z/ode-workspace-prep/response-screen-20261115/analysis-repo`,0e78ce7.
Their builder, evaluator, grid and pipeline are byte-identical. Readiness and
benchmark must pass before launching. Use benchmark-selected1or2workers.

```bash
cd /Users/mr.z/ode-workspace-prep/response-screen-20261115/repo
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RESPONSE_SCREEN_ASSETS=/Users/mr.z/ode-workspace-prep/response-screen-20261115/assets /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/run_queue.py --host macmini --device cpu --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --output /Users/mr.z/ode-workspace-prep/response-screen-20261115/output --workers 2 --grid studies/muon_pd_20261102/response_screen_20261115/grid.json --runner studies/muon_pd_20261102/response_screen_20261115/pipeline_screen.py --experiment-id response_screen_20261115 --detach
```

The example workers2 must be replaced by1 if the benchmark selects1.
Never rerun against an existing queue.lock. The queue records steps, child
PIDs, CPU load and errors every30seconds. Each child completes20k training,
native final sampling and selection-bank evaluation before success is recorded.
Full checkpoints and learned matrix diagnostics remain outside Git.

Run the completion watcher in a detached session under the same environment:

```bash
cd /Users/mr.z/ode-workspace-prep/response-screen-20261115/analysis-repo
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/response_screen_20261115/finish_screen.py --runtime /Users/mr.z/ode-workspace-prep/response-screen-20261115/output --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --assets /Users/mr.z/ode-workspace-prep/response-screen-20261115/assets --numpy-python /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 --output /Users/mr.z/ode-workspace-prep/response-screen-20261115/analysis-v1
```

The launch receipt records the actual NumPy Python path; the template may need
that one path changed to the locally verified Python executable. The watcher
waits for all36 successful pipelines, measures forward latency on idle experiment
workers, independently recomputes metrics, freezes a shortlist, then evaluates
only that shortlist and ordinary controls on disjoint confirmation arrays.
A failure is recorded and exits; no silent model rescue or automatic image run.
