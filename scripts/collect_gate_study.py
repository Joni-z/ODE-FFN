"""Watch the gate study, retrieve completed evidence, and audit saved samples."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.monitor_stage import now, write_json
from scripts.run_gate_study import select_checkpoint
from synthetic.train import sha256, sliced_w2


def remote(code: str, destination: Path | None = None) -> bytes:
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
               "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2",
               "145", shlex.join(["/data2/zz5070/miniconda3/envs/py312/bin/python", "-c", code])]
    if destination is None:
        return subprocess.check_output(command, stderr=subprocess.PIPE, timeout=30)
    with destination.open("wb") as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.PIPE, check=True, timeout=90)
    return b""


def audit(destination: Path) -> dict:
    output = destination / "output"
    frozen = json.loads((output / "frozen-inputs.json").read_text())
    manifest = frozen["manifest"]
    rows = json.loads((output / "summary.json").read_text())["results"]
    expected = {(d, e["model"]) for d in manifest["datasets"] for e in manifest["entries"]}
    seen = set()
    checks = []
    for dataset, model in sorted(expected):
        entry = next(e for e in manifest["entries"] if e["model"] == model)
        folder = output / "runs" / f"gates_20261021_{model}" / dataset / model / "seed000"
        assert (folder / "COMPLETE").read_text().strip() == "5000"
        assert sha256(folder / "config.json") == entry["sha256"]
        provenance = json.loads((folder / "provenance.json").read_text())
        assert provenance["seed"] == 0 and provenance["device"] == frozen["device"]
        for name, digest in provenance["source_sha256"].items():
            assert digest == frozen["source_and_data_sha256"][f"synthetic/{name}"]
        metrics = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
        selected = select_checkpoint(metrics, manifest["selection_steps"])
        model_rows = [r for r in rows if (r["dataset"], r["model"]) == (dataset, model)]
        assert {r["step"] for r in model_rows} == {5000, selected}
        for row in model_rows:
            identity = (dataset, model, row["step"])
            assert identity not in seen
            seen.add(identity)
            assert row["primary"] == (row["step"] == 5000)
            assert row["selected"] == (row["step"] == selected)
            path = output / "samples" / f"{dataset}-{model}-{row['step']}.npz"
            assert sha256(path) == row["samples_sha256"]
            with np.load(path, allow_pickle=False) as samples:
                score = sliced_w2(torch.from_numpy(samples["generated"]),
                                  torch.from_numpy(samples["target"]),
                                  torch.from_numpy(samples["directions"]))
            difference = abs(score-row["test_sw2"])
            assert difference < 1e-6, (identity, difference)
            checks.append(difference)
    assert len(rows) == len(seen)
    receipt = dict(time=now(), training_runs=len(expected), evaluated_endpoints=len(rows),
                   seed=0, device=frozen["device"], all_sample_hashes_verified=True,
                   max_sw2_recomputation_difference=max(checks),
                   archive_sha256=sha256(destination / "results.tar.gz"))
    write_json(destination / "audit.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.destination.mkdir(parents=True, exist_ok=True)
    base = "/data2/zz5070/ODE_FFN/gates_20261021_cpu"
    deadline = time.monotonic() + 3600
    while True:
        try:
            code = f"""import pathlib,json
b=pathlib.Path({base!r})
s=json.loads((b/'output/status.json').read_text())
m=json.loads((b/'output/monitor.json').read_text())
print(json.dumps(dict(status=s,monitor=m)))
"""
            state = json.loads(remote(code))
            write_json(args.destination / "watch-status.json", dict(time=now(), **state))
            if state["status"]["state"] == "FAILED":
                raise RuntimeError(state["status"])
            if state["status"]["state"] == "COMPLETE":
                code = f"""import pathlib,json,hashlib,tarfile,sys
b=pathlib.Path({base!r});o=b/'output'
assert json.loads((o/'status.json').read_text())['state']=='COMPLETE'
rows=json.loads((o/'summary.json').read_text())['results']
for row in rows:
 p=o/'runs'/('gates_20261021_'+row['model'])/row['dataset']/row['model']/'seed000/checkpoints'/('step_%07d.pt'%row['step'])
 assert hashlib.sha256(p.read_bytes()).hexdigest()==row['checkpoint_sha256']
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 archive.add(o,arcname='output',filter=lambda item: None if '/checkpoints' in item.name else item)
 archive.add(b/'launch.json',arcname='launch.json')
"""
                archive = args.destination / "results.tar.gz"
                remote(code, archive)
                with tarfile.open(archive) as handle:
                    handle.extractall(args.destination, filter="data")
                receipt = audit(args.destination)
                write_json(args.destination / "watch-status.json", dict(state="COLLECTED", **receipt))
                print(json.dumps(receipt), flush=True)
                return
        except (subprocess.SubprocessError, OSError) as error:
            write_json(args.destination / "watch-status.json",
                       dict(time=now(), state="CONNECTION_ERROR", error=str(error)))
            if not args.watch:
                raise
        if not args.watch:
            return
        if time.monotonic() > deadline:
            raise TimeoutError("One-hour collection window expired; remote study was not modified")
        time.sleep(45)


if __name__ == "__main__":
    main()
