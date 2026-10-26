"""One-factor anchored-corrector screen with unchanged ODEFFN and generic controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from synthetic.anchored_corrector import build
from scripts import run_recurrent_depth as base

DOC = "docs/research-anchored-corrector-2026-10-26"
ARMS = {f"{family}_{mode}_final": dict(family=family, mode=mode, supervision="final")
        for mode in ("accumulating", "anchored")
        for family in ("ode_k2", "swiglu192", "swiglu128")}


def diagnostics(model, banks: Path, output: Path, *, rollout: bool = False) -> dict:
    metrics = base.diagnostics(model, banks, output, rollout=False)
    metrics["iteration_curve"] = [dict(updates=count, **base.field_evaluation(model, banks, updates=count))
                                  for count in (1, 2, 3, 6, 12)]
    with np.load(banks/"probe.npz") as data:
        x, t = [torch.from_numpy(data[key][:256]) for key in ("x", "t")]
    with torch.no_grad():
        states = model.hidden_states(x, t, updates=13)
        metrics["hidden_changes"] = [dict(iteration=i,
            step_rms=float((states[i+1]-states[i]).square().mean().sqrt()),
            state_rms=float(states[i].square().mean().sqrt()),
            distance_to_input_rms=float((states[i]-states[0]).square().mean().sqrt()))
            for i in range(13)]
    base.write(output/"metrics.json", metrics)
    return metrics


def train_cell(config: Path, output: Path, dataset: str, arm: str, steps: int = 5000) -> None:
    base.train_cell(config, output, dataset, arm, steps=steps, rollout=False,
                    build_model=build, arm_settings=ARMS, final_diagnostics=diagnostics)


def freeze(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(ROOT/"synthetic", output/"synthetic",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("scripts/run_anchored_corrector.py", "scripts/run_recurrent_depth.py",
                 "scripts/preflight_anchored_corrector.py", "scripts/run_prediction_bridge.py",
                 "scripts/check_gate_geometry.py", "tests/test_anchored_corrector.py",
                 "tests/test_recurrent_depth.py", DOC+"/protocol.md"):
        destination = output/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, destination)
    base.write(output/"source-sha256.json", {p.relative_to(output).as_posix(): base.sha256(p)
                                            for p in output.rglob("*") if p.is_file()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "cell", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset", choices=base.DATASETS)
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--datasets", choices=base.DATASETS, nargs="+")
    parser.add_argument("--data-v1", type=Path)
    parser.add_argument("--data-product", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.output.resolve())
    elif args.action == "cell":
        train_cell(args.config.resolve(), args.output.resolve(), args.dataset, args.arm)
    else:
        if not args.datasets or not args.data_v1 or not args.data_product:
            parser.error("run requires datasets and both cached data roots")
        config = dict(seed=0, steps=5000, workers=2, device="cpu", datasets=args.datasets, arms=ARMS,
            data_roots={d: str((args.data_product if d==base.DATASETS[2] else args.data_v1).resolve())
                        for d in args.datasets})
        base.run(args.output.resolve(), config, script_relative="scripts/run_anchored_corrector.py",
                 protocol_relative=DOC+"/protocol.md")


if __name__ == "__main__":
    main()
