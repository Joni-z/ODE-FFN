# Full-bank outer refinement on an idle145GPU4-7

This executes P0 of [the mechanism follow-up](sharing-symmetry-followup.md).
It changes no weights, response, training seed or optimizer. The eight primary
sharing-cohort cases have completed independent evaluation. Four focal models
are exported with their exact CPU field and sample references, configurations,
manifests and hashes. Source publication and a fresh documented CUDA witness
precede the full inference job. This document alone is not a launch receipt.

Warm environment: `/data2/zz5070/miniconda3/envs/py312/bin/python`,
Torch2.8.0+cu128. No packages installed or rebuilt. Runtime root:
`/data2/zz5070/ODE_FFN/mechanism-sharing-20261116`.

The dispatcher selects a physicalGPU4-7 with no resident compute process
and less than500MiB used, then exposes only its UUID. Require no resident
foreign compute process before launching. Physical0-3
are prohibited. Stop only our process if another task appears. Log occupancy,
power and utilization after each sampling resolution.

## Fresh documented witness

Execute exactly once from Mac mini after the source and portable files are
present on145. The dispatcher selects and binds the currently empty card:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=8 145 'cd /data2/zz5070/ODE_FFN/mechanism-sharing-20261116/repo && /data2/zz5070/miniconda3/envs/py312/bin/python -u studies/muon_pd_20261102/coefficient_20261116/sharing_dispatch.py --root /data2/zz5070/ODE_FFN/mechanism-sharing-20261116/portable --output /data2/zz5070/ODE_FFN/mechanism-sharing-20261116/doc-witness --witness'
```

Require all four cases, finite predictions, field portability at2e-5absolute
and relative tolerance, native64-source trajectory portability below1e-3,
exit0 and COMPLETE.json passed. The witness binds the portable manifest hash.
Do not replace a missing receipt with import success or partial progress.

## Full inference after readiness

Run `sharing_dispatch.py` with the same environment, `--root` set to the portable
directory, `--readiness` to the witness COMPLETE.json, and `--output` to a
new `full-refinement` directory. Pass `--detach`. Preserve frozen source;
the dispatcher records the selected UUID, actual child PID and command.
It never selects physical0-3or an occupied card. The inference child repeats
the occupancy check before and after each case/resolution.
Pass `--include-inner16` for the registered hidden-resolution check too.

Each of four models receives all32,768original confirmation sources at
Heun128/256/512 with four internal substeps, plus internal16atHeun128.
For released stages,16substeps preserves the same four quarter-interval
matrices rather than changing its nonautonomous field. This asks whether
the sharing advantage survives finer hidden integration as well as the outer
generative solve. Native internal4/Heun128 must reproduce the CPU pooled score within
1e-4, otherwise stop without a cross-backend ranking. Save every generated
array, its hash, all eight bank scores and GPU telemetry. A separate NumPy
pass recomputes the16pooled and128bank scores. Report candidate differences
at every resolution; do not select the best solver setting afterwards.

The two released-stage checkpoints also pass bounded CPU convergence checks
against separately integrated quarter-interval DOP853 references. Median
observed32->64rates are1.992and1.991, consistent with midpoint order2;
reference tolerance refinement stays below1e-8. This validates the intended
piecewise hidden-field interpretation, not its sampling advantage.
[Exact results](sharing-convergence.json). The portable bundle was exported
with source61ddc56; its unchanged data hashes are bound independently of the
later inference-runner revision.
