"""Measure the selected solver against the original two-step reference."""
import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

from efficient_trial import base, build_midpoint, build_model, torch
from evaluate_screen import assets, ev


def compare(work):
    torch.set_num_threads(1)
    ev.build_model = lambda dim, family: (build_midpoint(dim, family) if family=="state_midpoint4"
                                         else build_model(dim, family))
    rows = []
    for final in base_read(work/"results.json")["full"]:
        if final["split"] != "confirmation":
            continue
        ds, family = final["dataset"], final["family"]
        older = work.parent/"report4-reduced-state-20261122/matched/trained-endpoints"
        ref, = older.glob(f"{ds}__state_midpoint2__*.json")
        info = base_read(ref)
        ref_dir, ref_step = Path(info["folder"]), info["target"]
        loc, = (work/"train/report4_efficient_solvers"/ds).glob(f"{ds}__{family}__*/seed000")
        history = work.parent/"coefficient-20261116/state-analysis/confirmation"
        anchor, = history.glob(f"{ds}__state_midpoint4__*/binding.json")
        baseline = Path(base_read(anchor)["source_folder"])
        check = base.digest(baseline/"checkpoints/step_0020000.pt")
        options = [(family, loc, final["step"], final["checkpoint_sha256"]),
                   ("state_midpoint2",ref_dir,ref_step,info["checkpoint_sha256"]),
                   ("state_midpoint4",baseline,20000,check)]
        restored = []
        for name, folder, step, digest in options:
            cfg = base_read(folder/"config.json")
            model, anchor = ev.restore(folder,cfg,Path(cfg["data_root"])/"manifest.json",step,"cpu")
            assert anchor["checkpoint_sha256"] == digest
            restored.append((name,model,digest))
        bank, _, binding = assets(work.parent/"coefficient-20261116/assets",ds,"confirmation","cpu")
        assert binding["sha256"] == final["assets_sha256"]
        x = torch.from_numpy(bank["source_0"][:512])
        sample = base.train.generated_samples
        times = {name:[] for name,_,_ in restored}
        for name, model, _ in restored:
            sample(model,x,32)
        for i in range(6):
            shift = i % len(restored)
            for name, model, _ in restored[shift:] + restored[:shift]:
                start = perf_counter()
                value = sample(model,x,32)
                times[name].append(perf_counter()-start)
                assert torch.isfinite(value).all()
        digest = {name:sha for name,_,sha in restored}
        rows.extend(dict(dataset=ds,family=name,seconds=t,median_seconds=median(t),
                         checkpoint_sha256=digest[name]) for name,t in times.items())
    return dict(device="Mac mini CPU",threads=1,samples=512,outer_heun_steps=32,rows=rows)


def base_read(file):
    import json
    return json.loads(file.read_text())


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work",type=Path)
    parser.add_argument("output",type=Path)
    a=parser.parse_args()
    base.dump(a.output,compare(a.work))
