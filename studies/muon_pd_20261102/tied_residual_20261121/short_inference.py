"""Deploy an existing midpoint checkpoint with four internal matrix calls."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from tied_residual import TiedResidualLayer, build_midpoint, torch
import evaluate_screen


def run(folder, assets_root, output, solver):
    ev = evaluate_screen.ev
    ev.build_model = build_midpoint
    torch.set_num_threads(1)
    cfg = json.loads((folder / "config.json").read_text())
    dataset = cfg["cell"]["dataset"]
    assert cfg["cell"]["family"] == "state_midpoint4" and cfg["seeds"] == [0]
    manifest = Path(cfg["data_root"]) / "manifest.json"
    model, binding = ev.restore(folder, cfg, manifest, 20000, "cpu")
    law = ev.base.train.load_distribution(json.loads(manifest.read_text())["datasets"][dataset])
    arrays, fixed, bank = evaluate_screen.assets(assets_root, dataset, "confirmation", "cpu")
    assert bank["stream"] == 2026111800
    sample = ev.base.train.generated_samples
    # Same 512 sources and outer solver for the interleaved latency comparison.
    probe = torch.from_numpy(arrays["source_0"][:512])
    original = [(layer, type(layer), layer.substeps) for _, layer in model.ode_layers()]
    def configure(short):
        for layer, cls, steps in original:
            layer.__class__ = TiedResidualLayer if short and solver == "euler4" else cls
            layer.substeps = (4 if solver == "euler4" else 2) if short else steps

    timings = {"midpoint4": [], solver: []}
    for sequence in ((False, True), (True, False), (False, True), (True, False)):
        for short in sequence:
            configure(short)
            start = time.perf_counter()
            sample(model, probe, 32)
            timings[solver if short else "midpoint4"].append(time.perf_counter() - start)
    configure(True)
    _, mse = ev.field(model, fixed)
    generated, targets, scores = [], [], []
    for i in range(8):
        value = sample(model, torch.from_numpy(arrays[f"source_{i}"]), 128)
        target = torch.from_numpy(arrays[f"target_{i}"])
        scores.append(ev.base.train.sliced_w2(value.double(),target.double(),fixed["directions"].double()))
        generated.append(value)
        targets.append(target)
        print(dataset, "bank", i + 1, "of 8", flush=True)
    batch, target = torch.cat(generated), torch.cat(targets)
    pooled = ev.base.train.sliced_w2(batch.double(), target.double(), fixed["directions"].double())
    result = dict(dataset=dataset, trained="state_midpoint4", deployed=solver,
                  internal_matrix_calls_per_solve=4, heun_steps=128,
                  stream=bank["stream"], assets_sha256=bank["sha256"], **binding,
                  field_mse=mse, pooled_sw2=pooled, bank_sw2=scores,
                  wall_seconds_512_heun32=timings, **evaluate_screen.cases(batch.numpy(), law))
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("folder", "assets_root", "output"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--solver", choices=["euler4", "midpoint2"], default="euler4")
    args = parser.parse_args()
    run(args.folder, args.assets_root, args.output, args.solver)
