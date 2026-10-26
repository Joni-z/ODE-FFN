"""Native-host preflight for the frozen recurrent-depth source."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_recurrent_depth import prepare_banks, train_cell, field_evaluation, build, write, sha256, restore_model


def run(output: Path, dataset: str, data_v1: Path, data_product: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    hashes = json.loads((ROOT/"source-sha256.json").read_text())
    for name, expected in hashes.items():
        if sha256(ROOT/name) != expected:
            raise ValueError("source mismatch")
    result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests",
                             "-p", "test_recurrent_depth.py", "-v"], cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
    (output/"unit-tests.log").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError("native unit tests failed")
    config = dict(seed=0, steps=3, datasets=[dataset],
                  data_roots={dataset: str(data_product if dataset=="rotated_product_bimodal_16d" else data_v1)})
    config_path = output/"study.json"
    write(config_path, config)
    prepare_banks(config, dataset, output)
    rows = []
    for arm in ("ode_k2_shared_deep", "ode_k2_cloned_final", "swiglu192_shared_deep"):
        train_cell(config_path, output, dataset, arm, 3, rollout=False)
        folder = output/"runs"/dataset/arm
        native = json.loads((folder/"result.json").read_text())
        state = torch.load(folder/"step_0000003.pt", map_location="cpu", weights_only=False)
        model = build(16 if dataset=="rotated_product_bimodal_16d" else 2, native["family"], native["mode"])
        restore_model(model, state)
        replay = field_evaluation(model, output/"banks"/dataset)
        for key, value in replay.items():
            if value != native[key]:
                raise AssertionError("checkpoint field replay differs: "+key)
        rows.append(dict(arm=arm, stream_digest=native["input_stream_digest"], replay="exact",
                         visits=len(json.loads((folder/"visits_0000003.json").read_text()))))
    if len({row["stream_digest"] for row in rows}) != 1:
        raise AssertionError("actual preflight training input streams differ")
    if rows[0]["visits"] != 6 or rows[1]["visits"] != 6 or rows[2]["visits"] != 0:
        raise AssertionError("per-visit captures do not match actual calls")
    for name, expected in hashes.items():
        if sha256(ROOT/name) != expected:
            raise AssertionError("source changed during preflight")
    write(output/"PASS.json", dict(status="PASS", dataset=dataset, torch=torch.__version__,
        unit_test_methods=8, actual_trainer_cells=rows, source_sha256=hashes,
        note="Three-update smoke checks only; not formal experimental results."))
    print("NATIVE PREFLIGHT PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-v1", type=Path, required=True)
    parser.add_argument("--data-product", type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve(), args.dataset, args.data_v1.resolve(), args.data_product.resolve())
