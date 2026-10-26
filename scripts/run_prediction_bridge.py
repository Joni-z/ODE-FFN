"""A frozen, monitored prediction-space comparison; training seed 0 only."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import numpy as np
import psutil
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from synthetic.models import VelocityNet
from synthetic.population import fingerprint_batch
from synthetic.train import load_distribution, load_targets, generated_samples, sliced_w2, sha256
from scripts.check_gate_geometry import jacobian, oracle_geometry

DOC = "docs/research-prediction-bridge-2026-10-25"
DATASETS = {
    "gaussian_aniso_2d": "/Users/mr.z/ode-workspace-prep/data/synthetic/v1",
    "gmm_ring8_2d": "/Users/mr.z/ode-workspace-prep/data/synthetic/v1",
    "rotated_product_bimodal_16d": "/Users/mr.z/ode-workspace-prep/data/synthetic/product16_v1",
}
BACKBONES = {
    "ode_k1": ("ode_original", {"orders": 1}),
    "ode_k2": ("ode_original", {"orders": 2}),
    "swiglu128": ("swiglu", {"ffn_hidden": 128}),
    "swiglu192": ("swiglu", {"ffn_hidden": 192}),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append(path: Path, value: object) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")


class PredictionBridge(nn.Module):
    def __init__(self, raw: VelocityNet, prediction: str, denominator_floor: float = .05):
        super().__init__()
        if prediction not in ("x", "v") or denominator_floor <= 0:
            raise ValueError("unsupported prediction conversion")
        self.raw = raw
        self.prediction = prediction
        self.denominator_floor = denominator_floor

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        value = self.raw(x, t)
        if self.prediction == "x":
            value = (value - x) / (1-t).clamp_min(self.denominator_floor)
        return value


def target_velocity(target: torch.Tensor, state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (target-state)/(1-t).clamp_min(.05)


def make_raw(dim: int, label: str) -> VelocityNet:
    kind, options = BACKBONES[label]
    return VelocityNet(dim, kind, width=64, depth=3, gate_hidden=64,
                       **options)


def copy_backbone(model: VelocityNet, reference: VelocityNet) -> None:
    """Copy all surrounding parameters and actual buffers, leaving FFNs alone."""
    with torch.no_grad():
        for getter in ("named_parameters", "named_buffers"):
            destination = dict(getattr(model, getter)())
            source = dict(getattr(reference, getter)())
            names = {name for name in destination if ".ffn." not in name}
            if names != {name for name in source if ".ffn." not in name}:
                raise ValueError("common backbone names differ")
            for name in names:
                if destination[name].shape != source[name].shape:
                    raise ValueError("common backbone shape differs: " + name)
                destination[name].copy_(source[name])


def build(dim: int, label: str, prediction: str) -> PredictionBridge:
    torch.manual_seed(0)
    reference = make_raw(dim, "ode_k2")
    torch.manual_seed(0)
    raw = make_raw(dim, label)
    copy_backbone(raw, reference)
    return PredictionBridge(raw, prediction)


def budget(model: PredictionBridge) -> dict:
    count = [0]
    def hook(module, args, result):
        count[0] += args[0].numel()*module.out_features
    hooks = [layer.register_forward_hook(hook) for layer in model.modules()
             if isinstance(layer, nn.Linear)]
    try:
        with torch.no_grad():
            model(torch.zeros(1, model.raw.dim), torch.full((1, 1), .5))
    finally:
        for handle in hooks:
            handle.remove()
    count[0] += sum(layer.orders*layer.proj.weight.numel()
                    for _, layer in model.raw.ode_layers())
    return {"parameters": sum(p.numel() for p in model.parameters()),
            "dense_forward_macs": count[0]}


def spectrum(model: PredictionBridge) -> dict:
    records = {}
    for name, layer in model.raw.ode_layers():
        matrix = layer.proj.weight.detach().double()
        singular = torch.linalg.svdvals(matrix)
        eigen = torch.linalg.eigvals(matrix)
        records[name] = {"singular_values": singular.tolist(),
                         "eigenvalue_real": eigen.real.tolist(),
                         "eigenvalue_imag": eigen.imag.tolist()}
    return records


def load_data(dataset: str):
    root = Path(DATASETS[dataset])
    manifest = json.loads((root/"manifest.json").read_text())
    spec = manifest["datasets"][dataset]
    if spec["source"] != "standard_normal" or spec["coupling"] != "independent":
        raise ValueError("unexpected source or coupling")
    folder = root/dataset
    for name in ("train.npz", "validation.npz", "test.npz", "probe.npz"):
        if sha256(folder/name) != manifest["files"][dataset+"/"+name]["sha256"]:
            raise ValueError("dataset checksum mismatch")
    return root, spec, load_distribution(spec), load_targets(folder/"train.npz", spec["dim"])


def prepare_banks(dataset: str, output: Path) -> None:
    root, spec, dist, _ = load_data(dataset)
    validation = load_targets(root/dataset/"validation.npz", spec["dim"])
    if len(validation) < 4096:
        raise ValueError("need at least 4096 validation targets")
    rng = np.random.default_rng(7761)
    target = validation[rng.integers(len(validation), size=20000)]
    source = rng.standard_normal(target.shape).astype(np.float32)
    u = rng.random((len(target), 1)).astype(np.float32)
    eta = rng.standard_normal((len(target), 1)).astype(np.float32)
    times = {"interior": .05+.9*u,
             "uniform_clipped": u,
             "native": 1/(1+np.exp(.8-.8*eta))}
    folder = output/"banks"/dataset
    folder.mkdir(parents=True, exist_ok=False)
    for name, t in times.items():
        state = (t*target+(1-t)*source).astype(np.float32)
        oracle = ((1-t)/np.maximum(1-t, .05))*dist.velocity(state, t)
        np.savez_compressed(folder/(name+".npz"), x=state, t=t, oracle=oracle)
    rng = np.random.default_rng(4427)
    sample_source = rng.standard_normal((2048, spec["dim"])).astype(np.float32)
    directions = rng.standard_normal((128, spec["dim"])).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    np.savez_compressed(folder/"sampling.npz", source=sample_source,
                        target=validation[:2048], target2=validation[2048:4096], directions=directions)
    for time_value in (.1, .3, .5, .7, .9):
        t = np.full((512, 1), time_value, dtype=np.float32)
        state = t*validation[:512]+(1-t)*rng.standard_normal((512, spec["dim"])).astype(np.float32)
        truth = oracle_geometry(dist, state, float(t[0, 0]))
        np.savez_compressed(folder/f"panel_{time_value:.1f}.npz", x=state, t=t,
                            oracle=truth["oracle"], oracle_jacobian=truth["oracle_jacobian"])


@torch.no_grad()
def field_evaluation(model: PredictionBridge, banks: Path, save: Path | None = None) -> dict:
    model.eval()
    metrics = {}
    for name in ("interior", "native", "uniform_clipped"):
        with np.load(banks/(name+".npz")) as data:
            x, t, oracle = (data[key] for key in ("x", "t", "oracle"))
        predictions = []
        for index in range(0, len(x), 1024):
            predictions.append(model(torch.from_numpy(x[index:index+1024]),
                                     torch.from_numpy(t[index:index+1024])).numpy())
        prediction = np.concatenate(predictions)
        if not np.isfinite(prediction).all():
            raise FloatingPointError("nonfinite field predictions")
        metrics[name+"_oracle_mse"] = float(np.mean((prediction-oracle)**2))
        if save is not None:
            np.savez_compressed(save/(name+".npz"), prediction=prediction)
    model.train()
    return metrics


def final_diagnostics(model: PredictionBridge, banks: Path, output: Path) -> dict:
    output.mkdir(exist_ok=False)
    metrics = field_evaluation(model, banks, output)
    model.eval()
    panel_rows = []
    for time_value in (.1, .3, .5, .7, .9):
        with np.load(banks/f"panel_{time_value:.1f}.npz") as data:
            x, t, oracle, truth = [data[key] for key in ("x", "t", "oracle", "oracle_jacobian")]
        pred, derivative = jacobian(model, torch.from_numpy(x), torch.from_numpy(t))
        pred, derivative = pred.numpy(), derivative.numpy()
        np.savez_compressed(output/f"panel_{time_value:.1f}.npz", prediction=pred, jacobian=derivative)
        panel_rows.append({"time": time_value, "field_mse": float(np.mean((pred-oracle)**2)),
                           "jacobian_error": float(np.mean(np.sum((derivative-truth)**2, axis=(1, 2))))})
    metrics["panels"] = panel_rows
    metrics["jacobian_error"] = float(np.mean([r["jacobian_error"] for r in panel_rows]))
    with np.load(banks/"sampling.npz") as data:
        source, target, target2, directions = [torch.from_numpy(data[key])
                                              for key in ("source", "target", "target2", "directions")]
    started = time.monotonic()
    generated = generated_samples(model, source, 32)
    if not torch.isfinite(generated).all():
        raise FloatingPointError("nonfinite generated samples")
    metrics.update(sw2=sliced_w2(generated, target, directions),
                   sw2_reference_floor=sliced_w2(target, target2, directions),
                   sampling_seconds=time.monotonic()-started)
    np.savez_compressed(output/"samples.npz", generated=generated.numpy())
    write(output/"metrics.json", metrics)
    return metrics


def checkpoint(model, optimizer, generator, step, digest, folder):
    path = folder/f"step_{step:07d}.pt"
    if path.exists():
        raise FileExistsError(path)
    torch.save({"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "runtime_buffers": {name: value.detach().clone() for name, value in model.named_buffers()},
                "data_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
                "input_stream_digest": digest}, path)
    write(folder/f"spectrum_{step:07d}.json", spectrum(model))


def cell(output: Path, dataset: str, backbone: str, prediction: str, steps: int) -> None:
    torch.set_num_threads(1)
    run = output/"runs"/dataset/(backbone+"_"+prediction)
    run.mkdir(parents=True, exist_ok=False)
    write(run/"status.json", {"state": "RUNNING", "step": 0, "pid": os.getpid(), "time": now()})
    try:
        _, _, _, targets = load_data(dataset)
        targets = torch.from_numpy(targets)
        model = build(targets.shape[1], backbone, prediction)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.)
        generator = torch.Generator().manual_seed(100000)
        digest = hashlib.sha256(b"").hexdigest()
        banks = output/"banks"/dataset
        write(run/"budget.json", budget(model))
        checkpoint(model, optimizer, generator, 0, digest, run)
        append(run/"metrics.jsonl", {"step": 0, **field_evaluation(model, banks)})
        aggregate_loss = aggregate_grad = 0.
        clipped = 0
        started = time.monotonic()
        for step in range(1, steps+1):
            index = torch.randint(len(targets), (512,), generator=generator)
            target = targets[index]
            source = torch.randn(target.shape, generator=generator)
            t = torch.sigmoid(-.8+.8*torch.randn((len(target), 1), generator=generator))
            state = t*target+(1-t)*source
            digest = fingerprint_batch(digest, target, source, t)
            optimizer.zero_grad(set_to_none=True)
            loss = (model(state, t)-target_velocity(target, state, t)).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError("nonfinite gradient")
            optimizer.step()
            aggregate_loss += loss.item()
            aggregate_grad += norm.item()
            clipped += int(norm > 1)
            if step % 500 == 0 or step == steps:
                interval = 500 if step % 500 == 0 else step % 500
                row = {"step": step, "train_loss": aggregate_loss/interval,
                       "grad_norm": aggregate_grad/interval, "clip_fraction": clipped/interval,
                       "wall_seconds": time.monotonic()-started, "time": now()}
                if step in (1000, 2500, steps):
                    row.update(field_evaluation(model, banks))
                    checkpoint(model, optimizer, generator, step, digest, run)
                append(run/"metrics.jsonl", row)
                write(run/"status.json", {"state": "RUNNING", "step": step,
                                          "pid": os.getpid(), "time": now()})
                print(json.dumps(row), flush=True)
                aggregate_loss = aggregate_grad = 0.
                clipped = 0
        result = final_diagnostics(model, banks, run/"diagnostics")
        write(run/"result.json", {"dataset": dataset, "backbone": backbone,
                                  "prediction": prediction, "step": steps,
                                  "input_stream_digest": digest, **budget(model), **result})
        write(run/"status.json", {"state": "COMPLETE", "step": steps,
                                  "pid": os.getpid(), "time": now()})
    except BaseException as error:
        write(run/"status.json", {"state": "FAILED", "pid": os.getpid(),
                                  "time": now(), "error": repr(error)})
        raise


def monitor(output: Path, stop: threading.Event) -> None:
    while not stop.is_set():
        rows = []
        for process in psutil.Process(os.getpid()).children(recursive=True):
            try:
                rows.append({"pid": process.pid, "rss_bytes": process.memory_info().rss,
                             "cpu_seconds": sum(process.cpu_times()[:2]),
                             "command": process.cmdline()})
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        row = {"time": now(), "available_memory_bytes": psutil.virtual_memory().available,
               "processes": rows}
        append(output/"monitor.jsonl", row)
        write(output/"monitor.json", row)
        stop.wait(10)


def freeze(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(ROOT/"synthetic", destination/"synthetic",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for relative in ["scripts/run_prediction_bridge.py", "scripts/check_gate_geometry.py",
                     "tests/test_prediction_bridge.py", DOC+"/protocol.md"]:
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/relative, target)
    files = {str(path.relative_to(destination)): sha256(path)
             for path in destination.rglob("*") if path.is_file()}
    write(destination/"source-sha256.json", files)


def run(output: Path, steps: int = 5000) -> None:
    if steps != 5000:
        raise ValueError("the registered screen is 5000 updates")
    output.mkdir(parents=True, exist_ok=False)
    for dataset in DATASETS:
        prepare_banks(dataset, output)
    source = output.parent/"source"
    if source != ROOT:
        raise ValueError("formal execution must use the frozen source beside output")
    hashes = json.loads((ROOT/"source-sha256.json").read_text())
    for name, expected in hashes.items():
        if sha256(ROOT/name) != expected:
            raise ValueError("source hash mismatch: "+name)
    write(output/"provenance.json", {"time": now(), "python": sys.version,
                                     "torch": torch.__version__, "seed": 0, "device": "cpu",
                                     "workers": 2, "steps": steps, "source_sha256": hashes,
                                     "protocol_sha256": sha256(ROOT/DOC/"protocol.md")})
    stop = threading.Event()
    watcher = threading.Thread(target=monitor, args=(output, stop), daemon=True)
    watcher.start()
    progress = {"state": "RUNNING", "total": 24, "completed": 0, "time": now()}
    write(output/"status.json", progress)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
    processes: set[subprocess.Popen] = set()
    lock = threading.Lock()
    aborted = threading.Event()

    def launch(job):
        if aborted.is_set():
            raise RuntimeError("screen cancelled after a failed cell")
        dataset, backbone, prediction = job
        log = output/f"{dataset}-{backbone}-{prediction}.log"
        command = [sys.executable, str(ROOT/"scripts/run_prediction_bridge.py"), "cell",
                   "--output", str(output), "--dataset", dataset, "--backbone", backbone,
                   "--prediction", prediction, "--steps", str(steps)]
        with log.open("w") as handle:
            with lock:
                if aborted.is_set():
                    raise RuntimeError("screen cancelled before launch")
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
                processes.add(process)
            returncode = process.wait()
            with lock:
                processes.remove(process)
        if returncode:
            raise RuntimeError("failed cell: "+"/".join(job))
        return job

    # Alternate prediction spaces and architecture families within each dataset.
    jobs = [(dataset, backbone, prediction) for dataset in DATASETS
            for backbone in BACKBONES for prediction in ("v", "x")]
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(launch, job) for job in jobs]
            try:
                for future in as_completed(futures):
                    job = future.result()
                    progress.update(completed=progress["completed"]+1, latest=list(job), time=now())
                    write(output/"status.json", progress)
            except BaseException:
                aborted.set()
                for future in futures:
                    future.cancel()
                with lock:
                    for process in processes:
                        process.terminate()
                raise
        for name, expected in hashes.items():
            if sha256(ROOT/name) != expected:
                raise ValueError("source changed during training: "+name)
        progress.update(state="COMPLETE", time=now())
        write(output/"status.json", progress)
    except BaseException as error:
        with lock:
            for process in processes:
                process.terminate()
        progress.update(state="FAILED", time=now(), error=repr(error))
        write(output/"status.json", progress)
        raise
    finally:
        stop.set()
        watcher.join(timeout=15)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "cell"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--dataset", choices=tuple(DATASETS))
    parser.add_argument("--backbone", choices=tuple(BACKBONES))
    parser.add_argument("--prediction", choices=("x", "v"))
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.output.resolve())
    elif args.action == "cell":
        cell(args.output.resolve(), args.dataset, args.backbone, args.prediction, args.steps)
    else:
        run(args.output.resolve(), args.steps)


if __name__ == "__main__":
    main()
