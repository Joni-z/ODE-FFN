# ODE-FFN

Development repository: [Joni-z/ODE-FFN](https://github.com/Joni-z/ODE-FFN). The `main` branch contains the active torch workspace; the previous GitHub version is preserved on `archive`. See the [development guide](docs/development.md) for data preparation after cloning and development across machines.

Primary workspace on torch: `/scratch/zz5070/ODE_FFN/jit-codebase`.

## Progress reports

- [Report 1 (PDF)](reports/report1/report1.pdf): project motivation, controlled ODELayer experiments, case studies, and hypotheses for the next two weeks. [LaTeX source and build instructions](reports/report1/README.md).

## Research status

The current research stage uses **unconditional synthetic flow matching** to study the original ODELayer and the spectrum of its learned matrix P before returning to image experiments. Existing ImageNet experiments are exploratory; their saved checkpoints do not capture complete training trajectories.

Latest progress (2026-10-20): two waves of gating, Taylor-order, and compute-budget experiments are complete, covering 30/30 training runs and 54/54 diagnostic tasks, all with seed 0. The [stage report](docs/research-2026-10-20/2026-10-20--odelayer--r00--ablation-report.md) documents the comparisons, training trajectories of P, and limitations. K=4 and K=8 perform similarly, while gains from gating and additional compute vary across settings. The next research direction remains open. The preceding 18 runs are preserved in the [SwiGLU comparison and diagnostics](docs/research-2026-10-19/results.md).

Follow-up diagnostics were submitted on 2026-10-20 as Slurm job `20405164`: 66 fixed-checkpoint evaluations test whether similar K=4/K=8 results reflect numerical saturation or training adaptation. A separate CPU-only matrix experiment is complete. This stage requests one GPU, benchmarks one versus two workers on that same GPU, and records utilization every 10 seconds. See the [prespecified protocol](docs/research-2026-10-20-followup/protocol.md) and [status and monitoring instructions](docs/research-2026-10-20-followup/status.md). Submission does not imply completion; consult the monitor before using new results.

An alternative execution directory on host 145 has passed source/data verification and CPU tests. See the [migration record](docs/research-2026-10-20-followup/migration-145.md) for its current readiness and the [storage audit](docs/research-2026-10-20-followup/storage.md). Preparation does not imply that the GPU evaluation has started.

For training waves, `scripts/submit_stage_monitored.py` includes monitoring. The fixed-checkpoint stage uses `scripts/run_frozen_order.slurm` and `scripts/monitor_diagnostics.py`. Freeze the manifest of the actual dataset on torch and use a new experiment directory for each stage. The monitors check Slurm and output artifacts every 60 seconds; see the [training-stage monitoring guide](docs/research-2026-10-20/monitoring.md). They record status and anomalies without automatically resubmitting jobs or sending external messages.

## Repository layout

| Path | Purpose |
|---|---|
| `synthetic/` | Low-dimensional and vector-valued distributions, flow-matching pairings, analytical velocity fields, data generation, and validation |
| `configs/synthetic/` | Versioned dataset parameters, separate from the legacy image trainer's configurations |
| `data/synthetic/v1/` | Local entry point for generated data, stored outside the repository at `../datasets/synthetic/v1/` |
| `../runs/synthetic/` | Synthetic training outputs, with separate directories per configuration and checkpoints at intermediate steps |
| `docs/synthetic-literature.md` | Dataset references from papers and official examples, and the rationale for the initial selection |
| `docs/synthetic.md` | Dataset descriptions, loading examples, theoretical checks, and experiment logging conventions |
| `docs/workspace.md` | Workspace locations on torch, archival procedures, and rollback instructions |
| `docs/legacy-overview.md` | Original README before reorganization; historical design notes include claims that have not yet been audited |
| `docs/checkpoints.csv` | Index of legacy checkpoint locations and file sizes |
| `docs/designs/` | Design documents previously stored in `Designs/` |
| `archive/imagenet/` | Legacy configurations and local logs; compatibility symlinks remain in the torch workspace |
| `src/`, `configs/ffn/`, `configs/Large/` | Legacy image-training implementation and configurations |
| `results/`, `wandb/`, `fid_stats/` | Legacy local checkpoints, tracking records, and FID reference statistics |
| `analysis/` | Local spectral-analysis scripts and full data for legacy P matrices; large artifacts are excluded from Git |

Datasets, checkpoints, environments, and raw logs are excluded from Git. Curated reports and analysis figures are included. Paths to local artifacts describe the torch workspace; those artifacts are not downloaded by cloning the repository.

## Using the prepared data on torch

```bash
cd /scratch/zz5070/ODE_FFN/jit-codebase
PYTHON=/scratch/zz5070/ODE_FFN/envs/synthetic/bin/python
OMP_NUM_THREADS=1 "$PYTHON" -m synthetic.validate data/synthetic/v1 --preview
"$PYTHON" -m unittest discover -s tests -p test_synthetic_data.py -v
```

To generate a new dataset version (existing output directories will not be overwritten):

```bash
"$PYTHON" -m synthetic.prepare --config configs/synthetic/data_v1.json --output ../datasets/synthetic/v2
```

`synthetic/train.py` supports SiLU FFN, SwiGLU, the professor's original ODEFFN, and spectral-norm-cap comparisons. Use this entry point for vector models. `src/main_jit.py` is the ImageNet image-training entry point.

Unconditional training means that the model receives only `x_t` and `t`, without class or mixture-component labels. The setup uses a standard Gaussian source, independent source–target pairings, and linear interpolation paths, with time running from noise at `t=0` to data at `t=1`.

## Notes on legacy image training

- `run.sh` currently appends `--debug` unconditionally and should not be used directly for full training runs. It selects configurations through a positional argument, such as `bash run.sh ffn/ode`.
- The `ode` configuration resolves to `ODEOnlyFFN`, rather than the professor's original two-stage `ODEFFN`. Some variants include RMS normalization of the update; check the configuration and implementation before interpreting P.
- Legacy `last.pt` checkpoints were repeatedly overwritten. Only the final saved states were found during the audit, so they cannot establish when P began changing during training.
- An input-normalization fix was introduced on 2026-04-18. Comparisons across periods must account for the actual training code used.

Legacy image models, training entry points, and local checkpoints are preserved. Synthetic training uses separate scripts. The historical 15-run analysis is in `docs/analysis-2026-10-13/`; additional SwiGLU experiments and mechanism diagnostics are in `docs/research-2026-10-19/`.
