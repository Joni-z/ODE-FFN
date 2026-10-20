"""Gate study: freeze inputs, benchmark concurrency, train and evaluate.

Run from a dedicated immutable checkout with data/synthetic/v1 linked to the
existing prepared dataset. The output directory must not already exist.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from synthetic.models import VelocityNet
from synthetic.train import (fixed_eval_data, generated_samples, load_distribution,
                             load_targets, sha256, sliced_w2)
from scripts.diagnose_synthetic import layer_diagnostics
from scripts.monitor_stage import now, write_json

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/synthetic/gates_20261021/manifest.json"


def jobs(manifest: dict) -> list[tuple[str, dict]]:
    result = []
    for index, dataset in enumerate(manifest["datasets"]):
        entries = manifest["entries"]
        result.extend((dataset, e) for e in entries[index:] + entries[:index])
    return result


def run_folder(output: Path, dataset: str, entry: dict) -> Path:
    cfg = json.loads((ROOT / entry["config"]).read_text())
    return output / "runs" / cfg["experiment_id"] / dataset / entry["model"] / "seed000"


def select_checkpoint(rows: list[dict], candidates: list[int]) -> int:
    scores = {r["step"]: r["sample"]["sliced_w2"] for r in rows
              if r["step"] in candidates and "sample" in r}
    if set(scores) != set(candidates):
        raise ValueError("Missing validation selection checkpoints")
    return min(candidates, key=lambda step: (scores[step], step))


def freeze(output: Path, manifest: dict, device: str) -> dict:
    files = list((ROOT / "synthetic").glob("*.py"))
    files += [Path(__file__), ROOT / "scripts/diagnose_synthetic.py",
              ROOT / "scripts/monitor_stage.py", MANIFEST,
              ROOT / "docs/research-2026-10-21-gates/protocol.md"]
    files += [ROOT / e["config"] for e in manifest["entries"]]
    files += [ROOT / "data/synthetic/v1/manifest.json"]
    for dataset in manifest["datasets"]:
        files += [ROOT / "data/synthetic/v1" / dataset / name
                  for name in ("train.npz", "validation.npz", "test.npz", "probe.npz")]
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in files}
    for entry in manifest["entries"]:
        if hashes[entry["config"]] != entry["sha256"]:
            raise ValueError("Config no longer matches the predeclared manifest")
    if hashes["docs/research-2026-10-21-gates/protocol.md"] != manifest["protocol_sha256"]:
        raise ValueError("Protocol changed after configuration registration")
    record = dict(created=now(), source_and_data_sha256=hashes,
                  manifest=manifest, python=sys.version, torch=torch.__version__,
                  device=device, cpu_count=os.cpu_count(),
                  gpu=torch.cuda.get_device_name() if device == "cuda" else None)
    write_json(output / "frozen-inputs.json", record)
    return hashes


def verify(hashes: dict) -> None:
    for path, expected in hashes.items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"Frozen input changed: {path}")


def benchmark_worker(output: Path, index: int, device: str) -> None:
    """Real-data training kernel with warmup excluded and a shared start barrier."""
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cfg = json.loads((ROOT / "configs/synthetic/gates_20261021/ode_original.json").read_text())
    model = VelocityNet(2, "ode_original", **cfg["model_options"]).to(device)
    data = torch.from_numpy(load_targets(ROOT / "data/synthetic/v1/gmm_ring8_2d/train.npz", 2)).to(device)
    rng = torch.Generator(device=device).manual_seed(100000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=0)

    def update() -> None:
        x1 = data[torch.randint(len(data), (512,), generator=rng, device=device)]
        x0 = torch.randn(x1.shape, generator=rng, device=device)
        t = torch.rand((512, 1), generator=rng, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = (model((1-t)*x0+t*x1, t) - (x1-x0)).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("benchmark loss")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        if not torch.isfinite(grad):
            raise FloatingPointError("benchmark gradient")
        optimizer.step()
        # Match the training loop's host synchronization.
        loss.item()
        grad.item()

    for _ in range(100):
        update()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    (output / f"ready-{index}").touch()
    deadline = time.monotonic() + 120
    while not (output / "GO").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("benchmark start barrier")
        time.sleep(.05)
    started = time.monotonic()
    for _ in range(1000):
        update()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    write_json(output / f"worker-{index}.json",
               dict(updates=1000, seconds=time.monotonic()-started))


def benchmark(output: Path, device: str) -> int:
    results = []
    for count in (1, 2):
        folder = output / f"benchmark-{count}"
        folder.mkdir()
        processes = []
        handles = []
        try:
            for i in range(count):
                handle = (folder / f"worker-{i}.log").open("w")
                handles.append(handle)
                processes.append(subprocess.Popen(
                    [sys.executable, str(Path(__file__)), "--benchmark-worker", str(i),
                     "--output", str(folder), "--device", device], stdout=handle,
                    stderr=subprocess.STDOUT, cwd=ROOT))
            deadline = time.monotonic() + 180
            while not all((folder / f"ready-{i}").exists() for i in range(count)):
                if any(p.poll() is not None for p in processes) or time.monotonic() > deadline:
                    raise RuntimeError(f"benchmark failed to warm up: {folder}")
                time.sleep(.2)
            started = time.monotonic()
            (folder / "GO").touch()
            for p in processes:
                if p.wait(timeout=180) != 0:
                    raise RuntimeError(f"benchmark failed: {folder}")
            seconds = time.monotonic() - started
            results.append(dict(workers=count, updates=1000*count, seconds=seconds,
                                updates_per_second=1000*count/seconds))
        finally:
            for p in processes:
                if p.poll() is None:
                    p.terminate()
                    p.wait(timeout=10)
            for handle in handles:
                handle.close()
    ratio = results[1]["updates_per_second"] / results[0]["updates_per_second"]
    selected = 2 if ratio >= 1.1 else 1
    write_json(output / "benchmark.json", dict(results=results, ratio=ratio, workers=selected))
    return selected


def monitor(output: Path, manifest: dict, stop: threading.Event, device: str) -> None:
    while not stop.is_set():
        rows = []
        for dataset, entry in jobs(manifest):
            folder = run_folder(output, dataset, entry)
            metrics = folder / "metrics.jsonl"
            step = None
            if metrics.exists():
                for line in reversed(metrics.read_text().splitlines()):
                    try:
                        step = json.loads(line)["step"]
                        break
                    except (json.JSONDecodeError, KeyError):
                        continue
            rows.append(dict(dataset=dataset, model=entry["model"], step=step,
                             complete=(folder / "COMPLETE").exists()))
        gpu = None
        if device.startswith("cuda"):
            try:
                command = ["nvidia-smi", "--query-gpu=uuid,utilization.gpu,memory.used,power.draw",
                           "--format=csv,noheader,nounits"]
                if os.environ.get("CUDA_VISIBLE_DEVICES"):
                    command += ["--id", os.environ["CUDA_VISIBLE_DEVICES"]]
                probe = subprocess.run(command, capture_output=True, text=True, timeout=5)
                gpu = dict(returncode=probe.returncode, data=probe.stdout.strip(),
                           error=probe.stderr.strip())
            except (OSError, subprocess.TimeoutExpired) as error:
                gpu = dict(error=str(error))
        record = dict(time=now(), progress=rows, gpu=gpu)
        write_json(output / "monitor.json", record)
        with (output / "telemetry.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        stop.wait(10)


def train_one(output: Path, dataset: str, entry: dict, device: str) -> None:
    log = output / "logs" / f"{dataset}-{entry['model']}.log"
    with log.open("w") as handle:
        result = subprocess.run(
            [sys.executable, "-m", "synthetic.train", "--config", entry["config"],
             "--dataset", dataset, "--model", entry["model"], "--seed", "0",
             "--device", device, "--output-root", str(output / "runs")],
            cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Training failed; inspect {log}")


def evaluate_all(output: Path, manifest: dict, device: str) -> None:
    # Enforce the training/test phase boundary for the whole study.
    for dataset, entry in jobs(manifest):
        folder = run_folder(output, dataset, entry)
        if (folder / "COMPLETE").read_text().strip() != "5000":
            raise ValueError(f"Incomplete training: {folder}")
    data_root = ROOT / "data/synthetic/v1"
    data_manifest = json.loads((data_root / "manifest.json").read_text())
    summary = []
    for dataset, entry in jobs(manifest):
        folder = run_folder(output, dataset, entry)
        cfg = json.loads((folder / "config.json").read_text())
        if sha256(folder / "config.json") != entry["sha256"] or cfg["sampling_split"] != "validation":
            raise ValueError("Training configuration differs")
        rows = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
        selected = select_checkpoint(rows, manifest["selection_steps"])
        spec = data_manifest["datasets"][dataset]
        fixed = fixed_eval_data(data_root / dataset, load_distribution(spec), 2, 0,
                                device, cfg["sample_count"], cfg["sliced_directions"])
        for step in sorted({manifest["primary_step"], selected}):
            path = folder / "checkpoints" / f"step_{step:07d}.pt"
            state = torch.load(path, map_location=device, weights_only=False)
            if (state["config_sha256"] != entry["sha256"] or state["step"] != step
                    or state["dataset_manifest_sha256"] != sha256(data_root / "manifest.json")
                    or state["identity"] != dict(dataset=dataset, model=entry["model"], seed=0)):
                raise ValueError("Checkpoint identity differs")
            model = VelocityNet(2, entry["model"], **cfg["model_options"]).to(device)
            model.load_state_dict(state["model"], strict=True)
            model.eval()
            samples = generated_samples(model, fixed["sample_source"], cfg["heun_steps"])
            sample_path = output / "samples" / f"{dataset}-{entry['model']}-{step}.npz"
            sample_path.parent.mkdir(exist_ok=True)
            np.savez_compressed(sample_path, generated=samples.cpu().numpy(),
                                source=fixed["sample_source"].cpu().numpy(),
                                target=fixed["sample_target"].cpu().numpy(),
                                directions=fixed["directions"].cpu().numpy())
            metric = next(r for r in rows if r["step"] == step)
            record = dict(dataset=dataset, model=entry["model"], step=step,
                          primary=step == manifest["primary_step"], selected=step == selected,
                          validation_sw2=metric["sample"]["sliced_w2"],
                          test_sw2=sliced_w2(samples, fixed["sample_target"], fixed["directions"]),
                          validation_pair_mse=metric["validation_pair_mse"],
                          validation_oracle_mse=metric["validation_oracle_mse"],
                          parameters=entry["parameters"], forward_macs=entry["forward_macs"],
                          checkpoint_sha256=sha256(path),
                          samples_sha256=sha256(sample_path),
                          layers=layer_diagnostics(model, fixed["probe_x"], fixed["probe_t"]))
            write_json(output / "evaluations" / f"{dataset}-{entry['model']}-{step}.json", record)
            summary.append(record)
    write_json(output / "summary.json", dict(results=summary))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--benchmark-worker", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.benchmark_worker is not None:
        benchmark_worker(args.output, args.benchmark_worker, args.device)
        return
    if args.device == "cpu":
        # Also prevent checkpoint RNG capture from initializing CUDA in children.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    elif torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one GPU using CUDA_VISIBLE_DEVICES")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    manifest = json.loads(MANIFEST.read_text())
    hashes = freeze(output, manifest, args.device)
    stop = threading.Event()
    watcher = threading.Thread(target=monitor, args=(output, manifest, stop, args.device), daemon=True)
    watcher.start()
    try:
        write_json(output / "status.json", dict(state="BENCHMARK", time=now(), pid=os.getpid()))
        workers = benchmark(output, args.device)
        write_json(output / "status.json", dict(state="TRAINING", time=now(), workers=workers, pid=os.getpid()))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(train_one, output, dataset, entry, args.device)
                       for dataset, entry in jobs(manifest)]
            for future in futures:
                future.result()
        verify(hashes)
        write_json(output / "status.json", dict(state="EVALUATING", time=now(), pid=os.getpid()))
        evaluate_all(output, manifest, args.device)
        verify(hashes)
        write_json(output / "status.json", dict(state="COMPLETE", time=now()))
    except BaseException as error:
        write_json(output / "status.json", dict(state="FAILED", time=now(), error=repr(error)))
        raise
    finally:
        stop.set()
        watcher.join(timeout=15)


if __name__ == "__main__":
    main()
