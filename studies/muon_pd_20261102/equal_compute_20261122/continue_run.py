"""Continue one seed-0 historical run in its own directory with the native trainer."""
import argparse
from pathlib import Path
import shutil
from types import SimpleNamespace

from budget import DOC, base, build_model, read, torch


def run(work, row):
    source = Path(row["source_folder"])
    cfg_path = source / "config.json"
    cfg = read(cfg_path)
    cell = cfg["cell"]
    base.verify_sources()
    manifest = base.verify_data(Path(cfg['data_root']).parent, cell) / 'manifest.json'
    assert base.digest(manifest) == row['manifest_sha256']
    assert base.digest(cfg_path) == row["config_sha256"]
    assert cfg["seeds"] == [0] and cfg["steps"] == 20000
    target = row["total_steps"]
    assert target > 20000 and cell["family"] != "state_midpoint4"
    original = source / "checkpoints/step_0020000.pt"
    assert base.digest(original) == row["checkpoint_sha256"]
    run_dir = work / "trained" / cfg["experiment_id"] / cell["dataset"] / cell["id"] / "seed000"
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    copy = run_dir / "config.json"
    if not copy.exists():
        shutil.copyfile(cfg_path, copy)
    assert base.digest(copy) == row["config_sha256"]
    latest = max(checkpoints.glob('step_*.pt'), default=original)
    if latest.stem != f"step_{target:07d}":
        # Native resume checks all identities and restores parameters, moments and RNG.
        base.train.VelocityNet = lambda dim, name, **opts: build_model(dim, cell["family"])
        args = SimpleNamespace(config=str(copy), dataset=cell["dataset"], model=cell["id"],
                               seed=0, device="cpu", output_root=str(work / "trained"),
                               steps=target, resume=str(latest), dry_run=False,
                               optimizer_factory=lambda model: base.ProjectionOptimizer(
                                   model, cell["optimizer"], cell["projection_lr"]))
        base.train.run(args)
    endpoint = checkpoints / f"step_{target:07d}.pt"
    ck = torch.load(endpoint, map_location="cpu", weights_only=False)
    assert ck["identity"] == dict(dataset=cell["dataset"], model=cell["id"], seed=0)
    assert ck["step"] == ck["optimizer"]["count"] == target
    assert ck["config_sha256"] == row["config_sha256"] and ck["dataset_manifest_sha256"] == row["manifest_sha256"]
    (run_dir / "ADAPTER_COMPLETE").write_text(f"Equal-compute training endpoint {target}\n")
    base.dump(work / "trained-endpoints" / (row["id"] + ".json"),
              dict(target=target, endpoint=str(endpoint), checkpoint_sha256=base.digest(endpoint),
                   parent_sha256=row["checkpoint_sha256"], folder=str(run_dir)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("case")
    args = parser.parse_args()
    row, = [r for r in read(DOC / "budget.json")["rows"] if r["id"] == args.case]
    run(args.work, row)
