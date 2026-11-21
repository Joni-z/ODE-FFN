"""Time matched-compute models in interleaved order on the same CPU."""
import argparse
import copy
import json
import platform
from pathlib import Path
import statistics
import time

from reduced_trial import HERE, ReducedState, base, build_model, linear, torch
from evaluate_screen import assets, ev

FAMILIES = ("state_midpoint4", "frozen_midpoint4", "state_midpoint2",
            "reduced_midpoint4", "reduced_midpoint4_unfolded")


class UnfoldedState(ReducedState):
    def solve(self, x, steps):
        encode, decode = self.factors()

        def drift(y):
            z = linear(y, encode)
            return linear(z * self.step_size(z), decode)

        dt = 1 / steps
        for _ in range(steps):
            x = x + dt * drift(x + (dt / 2) * drift(x))
        return x


def benchmark(work):
    prep = work.parent / "coefficient-20261116"
    results = []
    cells = json.loads((HERE/"grid.json").read_text())["cells"]
    ev.build_model = build_model
    torch.set_num_threads(1)
    for dataset in dict.fromkeys(cell["dataset"] for cell in cells):
        ref, = (prep/"state-analysis/confirmation").glob(f"{dataset}__state_midpoint4__*/binding.json")
        paths = {"state_midpoint4": (Path(json.loads(ref.read_text())["source_folder"]), 20000)}
        for cell in (c for c in cells if c["dataset"] == dataset):
            endpoint = json.loads((work/"matched/trained-endpoints"/(cell["id"]+".json")).read_text())
            paths[cell["family"]] = Path(endpoint["folder"]), endpoint["target"]
        models = {}
        for family, (folder, checkpoint) in paths.items():
            cfg = json.loads((folder/"config.json").read_text())
            models[family], _ = ev.restore(folder, cfg, Path(cfg["data_root"])/"manifest.json", checkpoint, "cpu")
        unfolded = copy.deepcopy(models["reduced_midpoint4"])
        for _, layer in unfolded.ode_layers():
            layer.__class__ = UnfoldedState
        models["reduced_midpoint4_unfolded"] = unfolded
        paths["reduced_midpoint4_unfolded"] = paths["reduced_midpoint4"]
        arrays, _, bank = assets(prep/"assets",dataset,"confirmation","cpu")
        probe = torch.from_numpy(arrays["source_0"][:512])
        sample = base.train.generated_samples
        warmups = {name: sample(model, probe, 32) for name, model in models.items()}
        torch.testing.assert_close(warmups["reduced_midpoint4"], warmups["reduced_midpoint4_unfolded"],
                                   rtol=1e-4, atol=1e-4)
        times = {f: [] for f in FAMILIES}
        for offset in range(len(FAMILIES)):
            for name in FAMILIES[offset:] + FAMILIES[:offset]:
                start = time.perf_counter()
                value = sample(models[name], probe, 32)
                times[name].append(time.perf_counter()-start)
                assert torch.isfinite(value).all()
        for name, durations in times.items():
            folder, step = paths[name]
            results.append(dict(dataset=dataset, family=name, steps=step, assets_sha256=bank["sha256"],
                                checkpoint_sha256=base.digest(folder/"checkpoints"/f"step_{step:07d}.pt"),
                                seconds=durations, median_seconds=statistics.median(durations)))
    return dict(host=platform.node(), device="cpu", threads=1, outer_heun_steps=32, samples=512, rows=results)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    base.dump(a.output,benchmark(a.work))
