# ODE-FFN

The ongoing protocol and audited experiments are in [ODE autoresearch](docs/autoresearch/README.md). The latest [independent-coefficient screen](docs/autoresearch/cycle14-independent-coefficient/analysis-report.md) trains ten new models: thin fitting improves modestly, but thick-data and ordinary-control gates fail. Its [quadratic-action precursor](docs/autoresearch/cycle13-quadratic-action/analysis-report.md) explains why term magnitude alone cannot establish useful higher-order computation. No useful new architecture has yet passed the required controls. The latest [frozen gradient diagnosis](docs/autoresearch/cycle15-gradient-routes/analysis-report.md) shows that the large fixed-readout normal loss has zero direct hidden gradient; its measured clipping-mediated effect is local and checkpoint-dependent.

Development repository: [Joni-z/ODE-FFN](https://github.com/Joni-z/ODE-FFN). The `main` branch contains the active torch workspace; the previous GitHub version is preserved on `archive`. See the [development guide](docs/development.md) for data preparation after cloning and development across machines.

Active experiment hosts are **Mac mini and Lenovo**. The historical torch checkout at `/scratch/zz5070/ODE_FFN/jit-codebase` is retained for provenance; MBP and torch are not required to run new experiments. GitHub publication should run directly from Mac mini.

Current autoresearch development worktree on Mac mini: `/Users/mr.z/ODE-FFN-autoresearch`; the synchronized main checkout remains `/Users/mr.z/ODE-FFN-report-work`. Earlier frozen experiments live under `/Users/mr.z/ode-workspace-prep/architecture-search/`; new autoresearch runtimes live under `/Users/mr.z/ode-workspace-prep/autoresearch/`. The M1 supports PyTorch MPS GPU training; current tiny-model studies use CPU because the measured updates are faster there.

Windows continuation workspace: `D:\Research\ODE-FFN\repo` on `lenovo` (RTX 4060 Laptop, 8 GB). See the [Windows workspace guide](docs/research-2026-10-24-lenovo/README.md) for the isolated environment, training command and monitoring files.

## Progress reports

- [Report 1 (PDF)](reports/report1/report1.pdf): project motivation, controlled ODELayer experiments, case studies, and hypotheses for the next two weeks. [LaTeX source and build instructions](reports/report1/README.md).

## Research status

The project tests whether a local ODE-inspired FFN can improve unconditional flow matching. The professor's original block uses two input-conditioned Taylor approximations to linear flows, separated by SiLU. Synthetic distributions provide analytic target velocities and Jacobians; these vector-model results do not establish an image-generation benefit.

**October 28:** the [paired manifold/head benchmark](docs/autoresearch/cycle07-manifold-heads/analysis.md) keeps the professor block unchanged. On a2D target embedded in128D with clean prediction, its field MSE is0.002729 versus0.002999 for parameter-matched Swi128 and0.003095 for compute-matched Swi192. The9% gain against Swi128 misses the fixed10% condition, and thick-data tangent transfer fails. No architecture is nominated. The study includes an input-rank bound, a full-dimensional control, generated examples and every P history; it does not replace older uncapped benchmarks.

**October 27:** a complete nonlinear Taylor correction, including normalization and gate derivatives, fails as a post-training upgrade to the fitted SwiGLU field on all four validation laws. This is a read-only checkpoint diagnostic, not new training or a general rejection of solver-aware architectures. See the [audited analysis and figures](docs/autoresearch/cycle01-complete-field/analysis.md). No useful final ODE architecture has yet been established.

The next [fixed-component-support control](docs/autoresearch/cycle02-full-support/analysis.md) trained five unchanged models on a half-Product/half-sign-code law. All65536 original component locations retain positive probability, but ISRU still loses to stronger SwiGLUs (field MSE0.16356 versus0.11325). Literal component count alone does not explain the prior reversal; probability concentration and posterior ambiguity remain distinct factors. The report includes the exact oracle, independent trajectory audit, P spectra and a between-branch covariance case study.

The [exact gated-shear baseline](docs/autoresearch/cycle03-gated-shear/analysis.md) represents SwiGLU by a square-zero lifted generator. Forward values, derivatives, 5000 paired optimizer updates and costs retain native behavior. This is an equivalent baseline, not a quality gain. Its generator has zero eigenvalues while conditional norm amplification changes with training; no complete-map stability guarantee follows. See [GatedShearFFN](synthetic/gated_shear.py).

The [exact gate-transport screen](docs/autoresearch/cycle04-gate-transport/analysis.md) changes the architecture and is stopped after15 audited cold trajectories. At almost matched parameters/MACs, Product16 field MSE is0.10138 for transport versus0.08465 for SwiGLU143; dependent-sign errors are0.04112 versus0.03897. The new generator is active and its solve is exact, but ordinary budget controls win. The report includes learned spectra, complete derivatives and a scoped gate-coefficient-rank analysis.

The [nonlinear feature-flow screen](docs/autoresearch/cycle05-feature-flow/analysis.md) is also stopped after10 audited trajectories. Product16 field MSE is0.09658 versus0.09709 for the paired midpoint control and0.08465 for matched-budget Swi143. The two-node approximation passes its separate32-node field/complete-Jacobian accuracy checks. Better integration of this response is not supported as a remedy; no node-count or initialization rescue is selected.

**October 26:** the ordinary ISRU observation does not transfer after fresh training on a dependent-sign target law with matched latent marginals, covariance and radial laws. Field MSE is0.085073 for ISRU and0.039790 for SwiGLU160. Its lower SW2 coexists with worse joint sign-pattern and component-distance diagnostics. The [complete transfer report](docs/research-joint-dependence-2026-10-26/analysis-report.md) includes every control, P trajectory, generation case study and a scoped mathematical analysis.

The preceding cubic-field comparison is preserved below. Its nonlinear local flow improved the old ODE on the nonlinear tasks but failed the ordinary-baseline controls; the free-ISRU Product16 observation motivated the transfer test.

| Model | Gaussian2D field MSE | Ring8 field MSE | Product16 field MSE |
|---|---:|---:|---:|
| Original ODE | 0.004449 | 0.032503 | 0.325527 |
| Cubic-field split | 0.005900 | 0.018883 | 0.092938 |
| Ordinary free ISRU | 0.005429 | 0.016694 | 0.037159 |
| SwiGLU128 | 0.003927 | 0.015296 | 0.092326 |
| SwiGLU160 | 0.004557 | 0.013921 | 0.078178 |

All rows use model seed 0 and 5000 updates. Original ODE and SwiGLU128 are explicitly reused historical controls. The [complete cubic-field report](docs/research-cubic-split-2026-10-26/analysis-report.md) includes matched controls, costs, full Jacobians, generated-sample comparisons and P trajectories. The registered ODE candidate is stopped; no useful final ODE architecture has yet been established.

Recent evidence:

- [Joint-dependence transfer control](docs/research-joint-dependence-2026-10-26/analysis-report.md): completed and independently audited. The ISRU field-risk ratio to the best SwiGLU changes from0.475 on independent signs to2.138 on dependent signs. Its [interaction analysis](docs/research-joint-dependence-2026-10-26/interaction-theory.md) distinguishes identical origin spectra from different higher-order target interactions, while showing why scalar-clock count alone is not a capacity explanation.
- [P versus direct effective matrices](docs/research-effective-map-2026-10-26/analysis-report.md): freeing the linear maps does not close the nonlinear fitting gap.
- [Hidden versus output bias](docs/research-bias-placement-2026-10-26/analysis-report.md): both placements learn, but neither materially closes that gap.
- [Local FFN teacher fitting](docs/research-local-ffn-teacher-2026-10-26/analysis-report.md) and [actual-data curvature](docs/research-data-curvature-2026-10-26/analysis-report.md): measured local behavior, with explicit limits on expressivity explanations.
- [ODE architecture literature synthesis](docs/literature-ode-synthesis-2026-10-26/review.md): continuous hidden depth, local flow maps, numerical composition and trained recurrence, based on primary sources.

Earlier comparisons and negative results remain in the [architecture evidence ledger](docs/research-architecture-search/task_plan.md), [October 20 stage report](docs/research-2026-10-20/2026-10-20--odelayer--r00--ablation-report.md) and [frozen-checkpoint follow-up](docs/research-2026-10-20-followup/analysis-report.md). Frozen study scripts retain original machine paths as provenance; use the portable training entry point below for a fresh setup.

## GitHub and machine synchronization

`main` is the shared code version. Completed, verified changes are pushed promptly so GitHub reflects current development. Check for local changes before updating another machine:

```bash
git status --short
git pull --ff-only origin main
```

Use this project identity for commits:

```bash
git config --local user.name "Zhizhe Zhang"
git config --local user.email "195848701+Joni-z@users.noreply.github.com"
```

Code and curated reports are synchronized through Git. Datasets, environments, checkpoints and raw logs remain local and are excluded by `.gitignore`. A Git checkout does not download external runtime archives.

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
