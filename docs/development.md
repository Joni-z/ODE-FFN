# Development checkout and runtime data

Canonical repository: https://github.com/Joni-z/ODE-FFN

- `main`: active development tree, including the original JiT code and the synthetic experiments.
- `archive`: original GitHub main at `3e8b067a2324354d2a7219e9785244c28c04e1ee`, preserved before the 2026-10-20 migration.
- torch worktree: `/scratch/zz5070/ODE_FFN/jit-codebase`.
- MBP checkout: `/Users/zzz/Documents/NYU/Researches/ODE FFN/ODE-FFN`. Existing sibling `Paper_reading` is left intact.

New experiment execution uses **Mac mini and Lenovo**. The torch/MBP paths above are historical checkouts, not required compute or publication services. Current Mac worktree: `/Users/mr.z/ODE-FFN-autoresearch`. Lenovo worktree: `D:\Research\ODE-FFN\repo`. Keep host-specific environments, datasets and immutable runtimes outside Git.

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

Do not rewrite `archive`. Develop new work on feature branches from `main`; synchronize Mac mini and Lenovo through GitHub. Check each checkout for local changes and use fast-forward pulls. Keep immutable training snapshots separate from mutable worktrees.

## Direct publication from Mac mini

Authenticate locally once using the official GitHub CLI browser flow; credentials belong in the CLI/OS credential store, not the repository:

```bash
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git --hostname github.com
```

After committing verified work as Zhizhe Zhang, publish from its worktree:

```bash
python scripts/publish_autoresearch.py
```

The publisher checks the canonical remote, a clean worktree, English commit identities and fast-forward ancestry, then performs a normal Git push and verifies remote main. It does not route publication through another machine or rewrite remote history. A login/network failure leaves local commits intact and must not be reported as successful publication.
