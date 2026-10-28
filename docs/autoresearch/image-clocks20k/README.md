# Frozen local clocks at 20,000 updates

The completed study replaces all six input-varying local clocks by their native first-64-input means at the actual 20k weights. K1 FID rises from 173.44 to 215.56; K2 rises from 180.78 to 226.23. Both lose the measured ordinary-control lead under this calibration. The two new interventions and all nine scores independently replay. No new training or architecture is selected.

Read the [analysis](analysis-report.md), [scope and checks](stats-appendix.md), [figures](figure-catalog.md) and [registered protocol](protocol.md). The early 5k collapse result does not transfer to these weights. Time adaptation, image/state feedback and calibration shift remain unresolved alternatives.

The warm environments and their spec hashes are unchanged. Full native 20k checkpoints, original banks and the sealed 5k clock helper are external inputs; Git alone does not download them. Work stays on Mac mini. Read the local provider ledger `.aris/compute/macmini.md` and the run-experiment compute contract before declaring readiness.

After the source/protocol are committed and `study.py prepare` has created the unused runtime below, a fresh agent must run this exact command once, without fixes or improvisation:

```text
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python -B /Users/mr.z/ODE-FFN-autoresearch/docs/autoresearch/image-clocks20k/study.py preflight --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-clocks20k
```

Preflight only replays native predictions and calibrates constants; it does not generate or score intervention images. It must report PASS with two native 1,024-pair replays, twelve calibration arrays, ten promoted folding witnesses and the decision-rule fixtures. If it fails, stop and preserve the runtime/log. Do not modify source, rerun it, install packages or launch the controller. Report the actual exit code, log path and any difference between this invocation and its claims. The owning agent records `documented-pass.json` only from a completed successful invocation.

After that documented pass, the registered controller command is:

```text
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python -B /Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-clocks20k/source/controller.py --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-clocks20k
```

The controller runs sampling, independent sample/calibration replay, unchanged Inception evaluation and feature/FID verification once. Its `orchestrator/status.json`, `launches.json` and five-second resource records are the monitoring entry points. Re-poll the same live process after a timeout; never infer failure from a silent sampler or restart it automatically. Expected storage is below 300 MB. No result is claimed until all four phases finish with exit 0.
