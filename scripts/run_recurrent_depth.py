"""Portable, monitored recurrent-depth ablation. All formal training uses seed0."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from synthetic.recurrent_depth import build, training_loss, compute_budget
from synthetic.population import fingerprint_batch
from synthetic.train import load_distribution, load_targets, fixed_eval_data, generated_samples, sliced_w2, sha256, restore_model
from scripts.check_gate_geometry import jacobian, oracle_geometry
from scripts.run_prediction_bridge import now, write, append, monitor, checkpoint

DOC = "docs/research-recurrent-depth-2026-10-26"
DATASETS = ("gaussian_aniso_2d", "gmm_ring8_2d", "rotated_product_bimodal_16d")
ARMS = {}
for layout in ("native", "cloned", "shared"):
    for supervision in ("final", "deep"):
        for family in ("ode_k2", "swiglu192"):
            ARMS[f"{family}_{layout}_{supervision}"] = dict(family=family, mode=layout, supervision=supervision)
for supervision in ("final", "deep"):
    ARMS[f"swiglu128_native_{supervision}"] = dict(family="swiglu128", mode="native", supervision=supervision)


def load_data(config: dict, dataset: str):
    root = Path(config["data_roots"][dataset])
    manifest = json.loads((root/"manifest.json").read_text())
    spec = manifest["datasets"][dataset]
    if spec["source"] != "standard_normal" or spec["coupling"] != "independent":
        raise ValueError("unexpected dataset coupling")
    for name in ("train.npz", "validation.npz", "test.npz", "probe.npz"):
        if sha256(root/dataset/name) != manifest["files"][dataset+"/"+name]["sha256"]:
            raise ValueError("dataset checksum mismatch")
    return root, spec, load_distribution(spec)


def prepare_banks(config: dict, dataset: str, output: Path) -> None:
    root, spec, distribution = load_data(config, dataset)
    fixed = fixed_eval_data(root/dataset, distribution, spec["dim"], 0, "cpu", 2048, 128, "validation")
    folder = output/"banks"/dataset
    folder.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(folder/"field.npz", x=fixed["x"].numpy(), t=fixed["t"].numpy(),
                        oracle=fixed["oracle"].numpy(), pair=fixed["y"].numpy())
    np.savez_compressed(folder/"probe.npz", x=fixed["probe_x"].numpy(), t=fixed["probe_t"].numpy())
    np.savez_compressed(folder/"sampling.npz", source=fixed["sample_source"].numpy(),
                        target=fixed["sample_target"].numpy(), target2=fixed["sample_target2"].numpy(),
                        directions=fixed["directions"].numpy())
    validation = load_targets(root/dataset/"validation.npz", spec["dim"])
    rng = np.random.default_rng(4427)
    for value in (.1, .3, .5, .7, .9):
        t = np.full((512, 1), value, dtype=np.float32)
        x = t*validation[:512]+(1-t)*rng.standard_normal((512, spec["dim"])).astype(np.float32)
        truth = oracle_geometry(distribution, x, float(t[0, 0]))
        np.savez_compressed(folder/f"panel_{value:.1f}.npz", x=x, t=t,
                            oracle=truth["oracle"], oracle_jacobian=truth["oracle_jacobian"])


@torch.no_grad()
def field_evaluation(model, banks: Path, *, updates: int = 3, step_size=None,
                     save: Path | None = None) -> dict:
    model.eval()
    with np.load(banks/"field.npz") as data:
        x, t, oracle, pair = [data[key] for key in ("x", "t", "oracle", "pair")]
    predictions = []
    for start in range(0, len(x), 1024):
        predictions.append(model(torch.from_numpy(x[start:start+1024]),
                                 torch.from_numpy(t[start:start+1024]),
                                 updates=updates, step_size=step_size).numpy())
    prediction = np.concatenate(predictions)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("nonfinite evaluation")
    if save is not None:
        np.savez_compressed(save, prediction=prediction)
    result = dict(oracle_mse=float(np.mean((prediction-oracle)**2)),
                  pair_mse=float(np.mean((prediction-pair)**2)),
                  prediction_rms=float(np.sqrt(np.mean(prediction**2))))
    model.train()
    return result


@torch.no_grad()
def visit_diagnostics(model, banks: Path) -> list:
    visits, hooks = [], []
    for name, layer in model.ode_layers():
        def capture(module, args, result, name=name):
            visits.append((name, module, args[0].detach().clone(),
                           args[1].detach().clone() if len(args)>1 and args[1] is not None else None))
        hooks.append(layer.register_forward_hook(capture))
    try:
        with np.load(banks/"probe.npz") as data:
            model(torch.from_numpy(data["x"][:32]), torch.from_numpy(data["t"][:32]))
    finally:
        for hook in hooks:
            hook.remove()
    counts, rows = {}, []
    for name, layer, hidden, context in visits:
        counts[name] = counts.get(name, 0)+1
        rows.append(dict(layer=name, visit=counts[name], activation=layer.activation_stats(hidden, context)))
    return rows


def diagnostics(model, banks: Path, output: Path, *, rollout: bool = True) -> dict:
    output.mkdir(exist_ok=False)
    metrics = field_evaluation(model, banks, save=output/"field.npz")
    model.eval()
    rows = []
    for value in (.1, .3, .5, .7, .9):
        with np.load(banks/f"panel_{value:.1f}.npz") as data:
            x, t, oracle, truth = [data[key] for key in ("x", "t", "oracle", "oracle_jacobian")]
        prediction, derivative = jacobian(model, torch.from_numpy(x), torch.from_numpy(t))
        prediction, derivative = prediction.numpy(), derivative.numpy()
        rows.append(dict(time=value, field_mse=float(np.mean((prediction-oracle)**2)),
                         jacobian_error=float(np.mean(np.sum((derivative-truth)**2, axis=(1, 2))))))
        np.savez_compressed(output/f"panel_{value:.1f}.npz", prediction=prediction, jacobian=derivative)
    metrics.update(panels=rows, jacobian_error=float(np.mean([row["jacobian_error"] for row in rows])))
    with np.load(banks/"sampling.npz") as data:
        source, target, target2, directions = [torch.from_numpy(data[key])
                                              for key in ("source", "target", "target2", "directions")]
    started = time.monotonic()
    samples = generated_samples(model, source, 32)
    if not torch.isfinite(samples).all():
        raise FloatingPointError("nonfinite samples")
    metrics.update(sw2=sliced_w2(samples, target, directions),
                   sw2_reference_floor=sliced_w2(target, target2, directions),
                   sampling_seconds=time.monotonic()-started)
    np.savez_compressed(output/"samples.npz", generated=samples.numpy())
    if rollout and model.mode == "shared":
        curves = []
        for count in (1, 2, 3, 6, 12):
            curves.append(dict(updates=count, convention="extended_horizon",
                **field_evaluation(model, banks, updates=count)))
        for count in (6, 12):
            curves.append(dict(updates=count, convention="fixed_horizon",
                **field_evaluation(model, banks, updates=count, step_size=math.sqrt(3)/count)))
        metrics["depth_curve"] = curves
    write(output/"metrics.json", metrics)
    return metrics


def train_cell(config_path: Path, output: Path, dataset: str, arm: str,
               steps: int = 5000, *, rollout: bool = True, build_model=None,
               arm_settings: dict | None = None, final_diagnostics=None) -> None:
    torch.set_num_threads(1)
    config = json.loads(config_path.read_text())
    setting = (ARMS if arm_settings is None else arm_settings)[arm]
    folder = output/"runs"/dataset/arm
    folder.mkdir(parents=True, exist_ok=False)
    write(folder/"status.json", dict(state="RUNNING", step=0, pid=os.getpid(), time=now()))
    try:
        root, spec, _ = load_data(config, dataset)
        target_bank = torch.from_numpy(load_targets(root/dataset/"train.npz", spec["dim"]))
        model = (build if build_model is None else build_model)(
            spec["dim"], setting["family"], setting["mode"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.)
        rng = torch.Generator().manual_seed(100000)
        digest = hashlib.sha256(b"").hexdigest()
        banks = output/"banks"/dataset
        costs = dict(inference=compute_budget(model),
                     training_forward=compute_budget(model, supervision=setting["supervision"]))
        write(folder/"provenance.json", dict(arm=arm, setting=setting, dataset=dataset, seed=0,
            torch=torch.__version__, actual_device="cpu", costs=costs,
            config_sha256=sha256(config_path), manifest_sha256=sha256(root/"manifest.json")))
        checkpoint(model, optimizer, rng, 0, digest, folder)
        write(folder/"visits_0000000.json", visit_diagnostics(model, banks))
        append(folder/"metrics.jsonl", dict(step=0, **field_evaluation(model, banks)))
        started = time.monotonic()
        loss_sum = gradient_sum = 0.
        clipped = count = 0
        for step in range(1, steps+1):
            target = target_bank[torch.randint(len(target_bank), (512,), generator=rng)]
            source = torch.randn(target.shape, generator=rng)
            t = torch.rand((512, 1), generator=rng)
            state = (1-t)*source+t*target
            digest = fingerprint_batch(digest, target, source, t)
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(model, state, t, target-source, setting["supervision"])
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(gradient):
                raise FloatingPointError("nonfinite gradient")
            optimizer.step()
            loss_sum += loss.item()
            gradient_sum += gradient.item()
            clipped += int(gradient > 1)
            count += 1
            if step % 500 == 0 or step == steps:
                record = dict(step=step, train_loss=loss_sum/count, grad_norm=gradient_sum/count,
                    clip_fraction=clipped/count, wall_seconds=time.monotonic()-started, time=now())
                if step in (1000, 2500, steps):
                    record.update(field_evaluation(model, banks))
                    checkpoint(model, optimizer, rng, step, digest, folder)
                    write(folder/f"visits_{step:07d}.json", visit_diagnostics(model, banks))
                append(folder/"metrics.jsonl", record)
                write(folder/"status.json", dict(state="RUNNING", step=step, pid=os.getpid(), time=now()))
                print(json.dumps(record), flush=True)
                loss_sum = gradient_sum = 0.
                clipped = count = 0
        metrics = (diagnostics if final_diagnostics is None else final_diagnostics)(
            model, banks, folder/"diagnostics", rollout=rollout)
        write(folder/"result.json", dict(dataset=dataset, arm=arm, **setting, step=steps,
            input_stream_digest=digest, **costs["inference"], **metrics))
        write(folder/"status.json", dict(state="COMPLETE", step=steps, pid=os.getpid(), time=now()))
    except BaseException as error:
        write(folder/"status.json", dict(state="FAILED", error=repr(error), pid=os.getpid(), time=now()))
        raise


def freeze(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(ROOT/"synthetic", destination/"synthetic",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for relative in ("scripts/run_recurrent_depth.py", "scripts/preflight_recurrent_depth.py",
                     "scripts/run_prediction_bridge.py", "scripts/check_gate_geometry.py",
                     "scripts/start_recurrent_depth.ps1", "tests/test_recurrent_depth.py", DOC+"/protocol.md"):
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/relative, target)
    hashes = {path.relative_to(destination).as_posix(): sha256(path)
              for path in destination.rglob("*") if path.is_file()}
    write(destination/"source-sha256.json", hashes)


def run(output: Path, config: dict, *, script_relative: str = "scripts/run_recurrent_depth.py",
        protocol_relative: str | None = None) -> None:
    hashes = json.loads((ROOT/"source-sha256.json").read_text())
    for name, expected in hashes.items():
        if sha256(ROOT/name) != expected:
            raise ValueError("source mismatch: "+name)
    output.mkdir(parents=True, exist_ok=False)
    write(output/"study.json", config)
    write(output/"provenance.json", dict(time=now(), python=sys.version, torch=torch.__version__,
        source_root=str(ROOT), source_sha256=hashes,
        protocol_sha256=sha256(ROOT/(protocol_relative or DOC+"/protocol.md"))))
    arms = config["arms"]
    progress = dict(state="RUNNING", phase="prepare", total=len(config["datasets"])*len(arms), completed=0, time=now())
    write(output/"status.json", progress)
    stop, aborted, lock = threading.Event(), threading.Event(), threading.Lock()
    watcher = threading.Thread(target=monitor, args=(output, stop), daemon=True)
    watcher.start()
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    processes = set()
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYTHONUNBUFFERED="1", PYTHONUTF8="1", CUDA_VISIBLE_DEVICES="")
    def launch(job):
        dataset, arm = job
        command = [sys.executable, str(ROOT/script_relative), "cell",
                   "--output", str(output), "--config", str(output/"study.json"),
                   "--dataset", dataset, "--arm", arm]
        with (output/f"{dataset}-{arm}.log").open("w") as handle:
            with lock:
                if aborted.is_set():
                    raise RuntimeError("queue aborted")
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
                processes.add(process)
            code = process.wait()
            with lock:
                processes.remove(process)
        if code:
            raise RuntimeError("failed cell: "+dataset+"/"+arm)
        return job
    try:
        for dataset in config["datasets"]:
            prepare_banks(config, dataset, output)
        progress.update(phase="training", time=now())
        write(output/"status.json", progress)
        jobs = [(dataset, arm) for dataset in config["datasets"] for arm in arms]
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
                raise ValueError("source changed during training")
        progress.update(state="COMPLETE", phase="complete", time=now())
    except BaseException as error:
        progress.update(state="FAILED", error=repr(error), time=now())
        raise
    finally:
        write(output/"status.json", progress)
        stop.set()
        watcher.join(timeout=15)
        if sys.platform == "win32":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "cell"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-v1", type=Path)
    parser.add_argument("--data-product", type=Path)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--arm", choices=tuple(ARMS))
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.output.resolve())
    elif args.action == "cell":
        train_cell(args.config.resolve(), args.output.resolve(), args.dataset, args.arm)
    else:
        if not args.datasets or not args.data_v1 or not args.data_product:
            parser.error("run needs datasets and both dataset roots")
        config = dict(seed=0, steps=5000, workers=2, device="cpu", datasets=args.datasets,
            arms=ARMS, data_roots={dataset: str((args.data_product if dataset==DATASETS[2] else args.data_v1).resolve())
                                  for dataset in args.datasets})
        run(args.output.resolve(), config)


if __name__ == "__main__":
    main()
