"""Check if averaging the two-step trajectory preserves Ring8 mode balance."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil

from cycle import DATASETS, base, evaluate, read, torch


def average(work, dataset):
    parent = work.parent/"report4-reduced-state-20261122/matched/trained-endpoints"
    file, = parent.glob(f"{dataset}__state_midpoint2__*.json")
    origin = read(file)
    last = origin["target"]
    folder = Path(origin["folder"])
    sources = [folder/"checkpoints"/f"step_{i:07d}.pt" for i in (25000,30000,last)]
    hashes = [base.digest(p) for p in sources]
    assert hashes[-1] == origin["checkpoint_sha256"]
    states = [torch.load(p, map_location="cpu",weights_only=False) for p in sources]
    tail = states[-1]
    for s in states:
        assert s["identity"] == tail["identity"] and s["identity"]["seed"] == 0
        assert s["config_sha256"] == tail["config_sha256"]
        assert s["dataset_manifest_sha256"] == tail["dataset_manifest_sha256"]
        assert all(torch.equal(s["runtime_buffers"][k],v) for k,v in tail["runtime_buffers"].items())
    model = dict(tail["model"])
    for key in model:
        v = [s["model"][key] for s in states]
        if all(torch.equal(v[0], x) for x in v[1:]):
            continue
        assert v[0].is_floating_point()
        model[key] = torch.stack([x.double() for x in v]).mean(0).to(v[0].dtype)
    staged = work/"averaged"/dataset
    staged.mkdir(parents=True, exist_ok=True)
    cp = staged/"checkpoints"/sources[-1].name
    cp.parent.mkdir(exist_ok=True)
    if not cp.exists():
        torch.save({k:tail[k] for k in ("step","runtime_buffers","identity","config_sha256",
                                      "dataset_manifest_sha256")} | {"model":model}, cp)
    shutil.copyfile(folder/"config.json", staged/"config.json")
    (staged/"ADAPTER_COMPLETE").touch()
    rows = [evaluate(work,staged,"average","state_midpoint2",last,dataset,s)
            for s in ("selection","confirmation")]
    return dict(dataset=dataset,source_steps=[25000,30000,last],source_sha256=hashes,rows=rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work",type=Path)
    a = p.parse_args()
    torch.set_num_threads(1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(lambda ds:average(a.work,ds), DATASETS))
    base.dump(a.work/"averaged-results.json", rows)
    for row in rows:
        print(row["dataset"],[(r["split"],r["pooled_sw2"]) for r in row["rows"]],flush=True)
