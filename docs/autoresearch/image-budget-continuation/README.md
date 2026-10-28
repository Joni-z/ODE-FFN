# Fixed image-budget comparison

The original K1 block retains the lowest image FID at 20k: 173.44 versus 195.31 for compute-matched Swi86 and 185.89 for near-parameter SiLU131. Its fitting errors are within 1% of those controls. K2 does not meet the registered follow-up margin. All seven models have worse FID at 20k than at 10k, despite better regression.

Read the [analysis](analysis-report.md), [comparison scope and checks](stats-appendix.md), [figures](figure-catalog.md) and [registered protocol](protocol.md). The result prioritizes investigation of the existing K1; it does not establish a new ODE method or identify a causal mechanism.

Runtime inputs and full checkpoints remain outside Git:
`/Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-budget-v2`.
The original 5k parents, datasets and environments are required for replay; cloning GitHub alone does not download them. Code and curated evidence are in this directory. Submitted Report1 is unchanged.

The warm image interpreter runs native checks; the metrics interpreter runs MPS Inception; the check interpreter curates figures. All training/evaluation/audit controllers have finished with exit 0. The frozen reporting source is `curation-source-v3/curate.py`, registered at `02d9aed`. With the external runtime available, its curation command is:

```text
/Users/mr.z/ode-workspace-prep/.venv-test/bin/python -B /Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-budget-v2/curation-source-v3/curate.py --root /Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-budget-v2 --output NEW_OUTPUT_DIRECTORY
```

Use a new output directory; existing outputs are preserved. Both complete independent audits are required before curation. `archive.py` verifies public copies, decisions, actual tool exits, visual review and immutable parents before sealing the archive; it does not train or evaluate another model.
