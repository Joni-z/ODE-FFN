"""Give the reduced-state model the historical four-step training budget."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from reduced_trial import HERE, base, build_model, torch

sys.path.insert(0, str(HERE.parent / "equal_compute_20261122"))
import budget
import continue_run


class ReducedArithmetic(budget.Arithmetic):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # A constant initializer and a layout operation add no scalar arithmetic.
        if func._schema.name in ("aten::zeros_like", "aten::permute"):
            return func(*args, **(kwargs or {}))
        return super().__torch_dispatch__(func, types, args, kwargs)


def profile(cell):
    dim = 2 if cell["dataset"] == "gmm_ring8_2d" else 16
    torch.manual_seed(0)
    model = build_model(dim, cell["family"])
    optimizer = base.ProjectionOptimizer(model, "adamw", 0.0003)
    x, y, t = torch.randn(512, dim), torch.randn(512, dim), torch.rand(512, 1)
    loss = (model((1 - t) * x + t * y, t) - (y - x)).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    with ReducedArithmetic() as forward:
        loss = (model((1 - t) * x + t * y, t) - (y - x)).square().mean()
    with ReducedArithmetic() as backward:
        loss.backward()
    with ReducedArithmetic() as update:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    return dict(forward=sum(forward.counts.values()), backward=sum(backward.counts.values()),
                optimizer=sum(update.counts.values()))


def plan():
    baseline = budget.read(budget.DOC / "budget.json")["rows"]
    result = []
    for cell in budget.read(HERE / "grid.json")["cells"]:
        ref, = [r for r in baseline if r["dataset"] == cell["dataset"] and r["family"] == "state_midpoint4"]
        phase_flops = profile(cell)
        flops = sum(phase_flops.values())
        target = ref["target_flops"]
        steps = (target + flops // 2) // flops
        result.append(dict(**cell, phase_flops=phase_flops, step_flops=flops,
                           target_flops=target, actual_flops=steps * flops, total_steps=steps))
    return result


def run(work, row):
    from time import sleep, time_ns
    import os
    cell = row["id"]
    source = work / "train/report4_reduced_state" / row["dataset"] / cell / "seed000"
    while not (source / "ADAPTER_COMPLETE").is_file():
        monitor = budget.read(work / "progress.json")["pid"]
        os.kill(monitor, 0)
        sleep(5)
    manifest = Path(budget.read(source / "config.json")["data_root"]) / "manifest.json"
    continue_run.build_model = build_model
    stage = work / "matched"
    continue_run.run(stage, dict(row, source_folder=str(source), manifest_sha256=base.digest(manifest),
                                    config_sha256=base.digest(source/"config.json"),
                                    checkpoint_sha256=base.digest(source/"checkpoints/step_0020000.pt")))
    end = budget.read(stage / "trained-endpoints" / f"{cell}.json")
    for split in ("selection","confirmation"):
        out = stage / split / cell
        if (out / "COMPLETE.json").is_file():
            continue
        if out.exists():
            out.rename(out.with_name(out.name + f"-partial-{time_ns()}"))
        cmd = [sys.executable, "-u", str(HERE/"evaluate.py"), "--folder", end["folder"],
               "--data-root", str(work.parent/"data/synthetic"),
               "--assets", str(work.parent/"coefficient-20261116/assets"),
               "--output", str(out), "--device", "cpu", "--split", split, "--steps", str(end["target"])]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("case", nargs="?")
    args = parser.parse_args()
    torch.set_num_threads(1)
    costs = args.work / "budget.json"
    if args.case:
        chosen, = [r for r in budget.read(costs) if r["id"] == args.case]
        run(args.work, chosen)
    else:
        rows = plan()
        base.dump(costs, rows)
        for row in rows:
            print(row["dataset"], row["family"], row["step_flops"], row["total_steps"])
