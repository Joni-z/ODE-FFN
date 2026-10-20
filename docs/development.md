# Development checkout and runtime data

Canonical repository: https://github.com/Joni-z/ODE-FFN

- `main`: current torch development tree, including the original JiT code and the synthetic experiments.
- `archive`: original GitHub main at `3e8b067a2324354d2a7219e9785244c28c04e1ee`, preserved before the 2026-10-20 migration.
- torch worktree: `/scratch/zz5070/ODE_FFN/jit-codebase`.
- MBP checkout: `/Users/zzz/Documents/NYU/Researches/ODE FFN/ODE-FFN`. Existing sibling `Paper_reading` is left intact.

## Fresh checkout

The repository contains source, configs, tests, reviewed reports, numeric summaries, figures and historical design documents. Raw datasets, checkpoints, environments, Slurm logs and local monitoring state are intentionally excluded. The legacy `Designs` and `configs_old` links point to tracked in-repository documentation/config directories. Raw log aliases on torch are machine-local.

For synthetic work use an isolated Python environment. The verified torch runtime uses PyTorch2.10.0 with CUDA12.8; install the appropriate PyTorch build for your own machine separately, then install `requirements-synthetic.txt`. No CUDA package is required for CPU-only data generation. The main `requirements.txt` is for the legacy image code.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-synthetic.txt
# Install a PyTorch build suitable for this machine before training/model tests.
python -m synthetic.prepare --config configs/synthetic/data_v1.json --output data/synthetic/v1
python -m unittest discover -s tests -p 'test_synthetic_*.py' -v
python -m unittest discover -s tests -p 'test_stage_*.py' -v
```

Dataset generation refuses to overwrite an existing directory. For exact archived run reproduction, use the actual torch data and frozen runtime; locally reconstructed arrays may agree while compressed-file/manifest hashes differ. Historic job scripts contain torch paths and belong to their recorded protocol: do not submit them unchanged as a new experiment. See `docs/research-2026-10-20/protocol.md` and `monitoring.md`.

## Source control hygiene

Before committing, inspect `git status`, `git diff --cached --stat` and the staged filenames. `.gitignore` excludes data, weights, environments, secrets and raw logs; it does not remove files already tracked by Git. Reviewed report JSON/CSV/PNG/PDF are deliberately retained as scientific evidence. Large diagnostic output should remain in the external runtime directories unless explicitly selected for a report.

Do not rewrite `archive`. Develop new work on feature branches from `main`; synchronize with GitHub before switching between MBP and torch. Keep active immutable training snapshots separate from the mutable main worktree.
