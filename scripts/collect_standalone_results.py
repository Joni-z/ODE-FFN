"""Retrieve and audit completed diagnostics, retrying SSH outages when requested."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.migrate_when_idle import now, read, write


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch(cfg, destination):
    remote_code = """
import json,pathlib,sys,tarfile
cfg = json.loads(%r)
out = pathlib.Path(cfg["output"])
assert json.loads((out/"runner.json").read_text())["state"] == "SUCCEEDED"
assert json.loads((out/"summary.json").read_text())["complete"]
with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as archive:
    archive.add(out, arcname="output")
    archive.add(cfg["manifest"], arcname="manifest.json")
    archive.add(pathlib.Path(cfg["remote_script"]).parent/"gpu-preflight.json",
                arcname="gpu-preflight.json")
""" % json.dumps(cfg)
    target = [cfg["python"], "-c", remote_code]
    host = cfg["target_host"]
    options = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
               "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2"]
    if cfg.get("target_gateway"):
        target = ["ssh", *options, host, shlex.join(target)]
        host = cfg["target_gateway"]
        options += ["-o", "ControlPath=none"]
    temporary = destination/"completed-results.tar.gz.partial"
    with temporary.open("wb") as stream:
        result = subprocess.run(["ssh", *options, host, shlex.join(target)],
                                stdout=stream, stderr=subprocess.PIPE, timeout=120)
    if result.returncode:
        raise ConnectionError(result.stderr.decode(errors="replace").strip())
    with tarfile.open(temporary) as archive:
        archive.extractall(destination, filter="data")
    temporary.replace(destination/"completed-results.tar.gz")


def audit(destination):
    out = destination/"output"
    manifest = read(destination/"manifest.json")
    summary = read(out/"summary.json")
    runner = read(out/"runner.json")
    rows = summary["rows"]
    assert summary["complete"] and len(rows) == manifest["expected_cells"] == 66
    assert runner["state"] == "SUCCEEDED" and runner["exit_code"] == 0
    assert len(summary["workers"]) == 3 and all(w["exit_code"] == 0 for w in summary["workers"])
    assert sha(destination/"manifest.json") == summary["manifest_sha256"]
    entries = {(e["dataset"], e["label"]): e for e in manifest["checkpoints"]}
    assert len(entries) == 18
    native = {}
    for key, entry in entries.items():
        ds, label = key
        value = read(out/"native"/ds/(label+".json"))
        assert value["checkpoint_sha256"] == entry["checkpoint_sha256"]
        assert abs(value["sw2"] - entry["reference_sw2"]) < 5e-4
        native[key] = value
    identities = set()
    for row in rows:
        key = row["dataset"], row["label"]
        identity = (*key, str(row["eval_order"]))
        assert identity not in identities
        identities.add(identity)
        assert row["status"] == "complete" and row["seed"] == 0
        assert row["checkpoint_sha256"] == entries[key]["checkpoint_sha256"]
        assert row["manifest_sha256"] == summary["manifest_sha256"]
        for field in ("sw2", "pair_mse", "delta_sw2_from_native",
                      "endpoint_rms_from_native", "velocity_rms_from_native"):
            assert math.isfinite(row[field]), (identity, field)
        sample = out/row["samples_relative_path"]
        assert sample.resolve().is_relative_to(out.resolve())
        assert sha(sample) == row["samples_sha256"]
        if row["eval_order"] == row["trained_order"]:
            assert abs(row["sw2"] - native[key]["sw2"]) < 5e-4
    for dataset in manifest["datasets"]:
        assert read(out/f"COMPLETE-{dataset}.json")["cells"] == 22
        assert sum(row["dataset"] == dataset for row in rows) == 22
    fields = ["dataset", "label", "trained_order", "eval_order", "sw2", "oracle_mse",
              "pair_mse", "delta_sw2_from_native", "endpoint_rms_from_native",
              "velocity_rms_from_native", "elapsed_seconds"]
    with (destination/"results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    receipt = dict(checked_at=now(), complete=True, cells=66, checkpoints=18,
                   sample_hashes_verified=66, seed=0, new_training=False,
                   manifest_sha256=summary["manifest_sha256"],
                   archive_sha256=sha(destination/"completed-results.tar.gz"),
                   runner=runner, throughput=read(out/"throughput.json"),
                   max_native_reference_sw2_difference=max(
                       abs(value["sw2"] - entries[key]["reference_sw2"])
                       for key, value in native.items()))
    write(destination/"collection-audit.json", receipt)
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--destination", type=Path, required=True)
    ap.add_argument("--watch", action="store_true")
    args = ap.parse_args()
    cfg = read(args.config)
    args.destination.mkdir(parents=True, exist_ok=True)
    if (args.destination/"collection-audit.json").exists():
        receipt = audit(args.destination)
        print(json.dumps(receipt), flush=True)
        return
    while True:
        try:
            fetch(cfg, args.destination)
            receipt = audit(args.destination)
            write(args.destination/"collection-status.json",
                  dict(checked_at=now(), state="COMPLETE", cells=receipt["cells"]))
            print(json.dumps(receipt), flush=True)
            return
        except (ConnectionError, subprocess.TimeoutExpired) as error:
            write(args.destination/"collection-status.json",
                  dict(checked_at=now(), state="WAITING_FOR_CONNECTION", error=str(error)))
            print(f"{now()} WAITING_FOR_CONNECTION: {error}", flush=True)
            if not args.watch:
                raise
            time.sleep(60)


if __name__ == "__main__":
    main()
