"""Fixed-checkpoint order intervention on one GPU, with measured worker concurrency."""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synthetic.interventions import action, evaluation_order
from synthetic.models import VelocityNet
from synthetic.train import fixed_eval_data, generated_samples, load_distribution, sliced_w2
from scripts.monitor_stage import now, write_json

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify(manifest):
    for name, expected in manifest["source_sha256"].items():
        assert digest(ROOT/name) == expected, f"Changed frozen source: {name}"
    data = Path(manifest["data_root"])
    assert digest(data/"manifest.json") == manifest["data_manifest_sha256"]
    mf = read(data/"manifest.json")
    for ds in manifest["datasets"]:
        for name, info in mf["files"].items():
            if name.startswith(ds+"/"):
                assert digest(data/name) == info["sha256"], f"Data changed: {name}"
    for entry in manifest["checkpoints"]:
        assert digest(entry["checkpoint"]) == entry["checkpoint_sha256"]
        assert digest(Path(entry["run_dir"])/"config.json") == entry["config_sha256"]


def load_model(entry, manifest, device):
    cfg = read(Path(entry["run_dir"])/"config.json")
    state = torch.load(entry["checkpoint"], map_location=device, weights_only=False)
    assert state["identity"] == dict(dataset=entry["dataset"], model=entry["model"], seed=0)
    assert state["step"] == 5000 and state["config_sha256"] == entry["config_sha256"]
    assert state["dataset_manifest_sha256"] == manifest["data_manifest_sha256"]
    spec = read(Path(manifest["data_root"])/"manifest.json")["datasets"][entry["dataset"]]
    model = VelocityNet(spec["dim"], entry["model"], **cfg["model_options"]).to(device).eval()
    model.load_state_dict(state["model"], strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    fixed = fixed_eval_data(Path(manifest["data_root"])/entry["dataset"],
                            load_distribution(spec), spec["dim"], 0, device, 2048, 128)
    return model, fixed


def weights_digest(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def local_probes(model, fixed):
    captured = {}
    hooks = [layer.register_forward_pre_hook(
        lambda m, args, key=name: captured.__setitem__(key, args[0][:16].detach().clone()))
        for name, layer in model.ode_layers()]
    try:
        model(fixed["probe_x"], fixed["probe_t"])
    finally:
        for hook in hooks:
            hook.remove()
    rows = []
    for name, layer in model.ode_layers():
        h = captured[name]
        s = layer.step_size(h).double()
        p = layer.operator().double()
        exact = action(h.double(), p, s, "exact")
        for k in [1,2,4,8]:
            y = action(h.double(), p, s, k)
            relative = (y-exact).norm(dim=1)/exact.norm(dim=1).clamp_min(1e-12)
            rows.append(dict(layer=name, eval_order=k, count=len(h),
                             mean_relative_action_error=relative.mean().item(),
                             max_relative_action_error=relative.max().item()))
    return rows


@torch.no_grad()
def evaluate(model, fixed):
    predictions = torch.cat([model(x, t) for x,t in zip(
        fixed["x"].split(512), fixed["t"].split(512))])
    if not torch.isfinite(predictions).all():
        raise FloatingPointError("Nonfinite validation velocity")
    paired = (predictions-fixed["y"]).square().mean().item()
    oracle = ((predictions-fixed["oracle"]).square().mean().item()
              if fixed["oracle"] is not None else None)
    samples = generated_samples(model, fixed["sample_source"], 32)
    if not torch.isfinite(samples).all():
        raise FloatingPointError("Nonfinite generated endpoint")
    bins = []
    for i in range(10):
        mask=(fixed["t"][:,0]>=i/10)&(fixed["t"][:,0]<(i+1)/10)
        bins.append(dict(t_left=i/10, t_right=(i+1)/10, count=int(mask.sum()),
            oracle_mse=((predictions[mask]-fixed["oracle"][mask]).square().mean().item()
                        if fixed["oracle"] is not None else None)))
    return dict(sw2=sliced_w2(samples, fixed["sample_target"], fixed["directions"]),
                pair_mse=paired, oracle_mse=oracle, time_bins=bins), samples, predictions


def worker(manifest_path, output, dataset, device="cuda"):
    manifest = read(manifest_path)
    verify(manifest)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    output=Path(output)
    entries=[e for e in manifest["checkpoints"] if e["dataset"] == dataset]
    for entry in entries:
        model, fixed=load_model(entry, manifest, device)
        before=weights_digest(model)
        start=time.monotonic()
        with torch.no_grad():
            native, native_samples, native_predictions=evaluate(model, fixed)
        if abs(native["sw2"]-entry["reference_sw2"]) >= 5e-4:
            raise RuntimeError(f"Native checkpoint reproduction failed: {entry['label']}: {native['sw2']}")
        native.update(dataset=dataset, label=entry["label"], trained_order=entry["orders"],
                      eval_order="native", checkpoint_sha256=entry["checkpoint_sha256"],
                      elapsed_seconds=time.monotonic()-start, status="complete")
        write_json(output/"native"/dataset/(entry["label"]+".json"), native)
        is_ode=entry["model"] == "ode_original"
        probes=local_probes(model, fixed) if is_ode else []
        write_json(output/"probes"/dataset/(entry["label"]+".json"), probes)
        modes=manifest["evaluation_orders"] if is_ode else ["native"]
        # Balanced rotation affects execution order only, never comparisons or weights.
        if is_ode:
            offset=manifest["datasets"].index(dataset)%len(modes)
            modes=modes[offset:]+modes[:offset]
        for mode in modes:
            target=output/"cells"/dataset/f"{entry['label']}--{mode}.json"
            if target.exists():
                old=read(target)
                assert old["checkpoint_sha256"] == entry["checkpoint_sha256"]
                assert old["manifest_sha256"] == digest(manifest_path)
                continue
            start=time.monotonic()
            result=dict(dataset=dataset, label=entry["label"], trained_order=entry["orders"],
                        eval_order=mode, checkpoint_sha256=entry["checkpoint_sha256"],
                        manifest_sha256=digest(manifest_path), seed=0)
            try:
                if mode == "native":
                    metrics, samples, predictions=native, native_samples, native_predictions
                else:
                    with torch.no_grad(), evaluation_order(model, mode, chunk=manifest["exact_chunk"]):
                        metrics, samples, predictions=evaluate(model, fixed)
                result.update({k:v for k,v in metrics.items() if k in (
                    "sw2","pair_mse","oracle_mse","time_bins")})
                result.update(status="complete", delta_sw2_from_native=result["sw2"]-native["sw2"],
                    endpoint_rms_from_native=(samples-native_samples).square().mean().sqrt().item(),
                    velocity_rms_from_native=(predictions-native_predictions).square().mean().sqrt().item())
                if mode == entry["orders"] and abs(result["sw2"]-native["sw2"]) >= 5e-4:
                    raise RuntimeError("Float64-action precision control failed")
                sample_path=output/"samples"/dataset/f"{entry['label']}--{mode}.npz"
                sample_path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(sample_path, generated=samples.detach().cpu().numpy(),
                                    validation_velocity=predictions.detach().cpu().numpy())
                result["samples_relative_path"]=str(sample_path.relative_to(output))
                result["samples_sha256"]=digest(sample_path)
            except FloatingPointError as error:
                result.update(status="nonfinite", reason=str(error))
            result["elapsed_seconds"]=time.monotonic()-start
            write_json(target,result)
            print(json.dumps(result),flush=True)
        assert before == weights_digest(model), "Intervention modified model weights"
        assert digest(entry["checkpoint"]) == entry["checkpoint_sha256"], "Checkpoint was modified"
    expected=22
    rows=[read(p) for p in (output/"cells"/dataset).glob("*.json")]
    assert len(rows) == expected and all(r["status"] == "complete" for r in rows)
    write_json(output/f"COMPLETE-{dataset}.json",dict(dataset=dataset, cells=len(rows),status="complete"))


def benchmark_child(manifest_path, barrier, queue, repeats):
    """Timed region starts after process/model initialization and GPU warm-up."""
    try:
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False
        manifest=read(manifest_path)
        entry=next(e for e in manifest["checkpoints"] if e["label"] == "gate_k2_input")
        model,fixed=load_model(entry,manifest,"cuda")
        x,t=fixed["x"][:512],fixed["t"][:512]
        with torch.no_grad(),evaluation_order(model,"exact",manifest["exact_chunk"]):
            model(x,t);torch.cuda.synchronize()
            barrier.wait(timeout=180)
            start=time.monotonic()
            for _ in range(repeats):
                model(x,t)
            torch.cuda.synchronize()
            queue.put(dict(seconds=time.monotonic()-start, examples=512*repeats,
                           peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    except BaseException as error:
        queue.put(dict(error=repr(error)))
        try: barrier.abort()
        except Exception: pass
        raise


def benchmark(manifest_path, workers):
    ctx=mp.get_context("spawn")
    barrier=ctx.Barrier(workers+1);queue=ctx.Queue()
    procs=[ctx.Process(target=benchmark_child,args=(manifest_path,barrier,queue,8)) for _ in range(workers)]
    for p in procs:p.start()
    try:
        barrier.wait(timeout=180)
        results=[queue.get(timeout=180) for _ in procs]
        for p in procs:p.join(timeout=30)
        if any("error" in r for r in results) or any(p.exitcode != 0 for p in procs):
            raise RuntimeError(f"GPU benchmark failed: {results}")
        # GPU synchronization ends each child's work. Exclude process-exit overhead.
        seconds=max(r["seconds"] for r in results)
        return dict(workers=workers, seconds=seconds, examples=sum(r["examples"] for r in results),
                    examples_per_second=sum(r["examples"] for r in results)/seconds,
                    per_worker=results)
    finally:
        for p in procs:
            if p.is_alive():p.terminate()
            p.join(timeout=10)


def telemetry(path, stop):
    path.parent.mkdir(parents=True,exist_ok=True)
    gpu=os.environ.get("CUDA_VISIBLE_DEVICES","0")
    command=["nvidia-smi","-i",gpu,"--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw",
             "--format=csv,noheader,nounits"]
    with path.open("a") as f:
        if path.stat().st_size == 0:f.write("timestamp,index,gpu_util_percent,memory_mib,power_w\n")
        while not stop.is_set():
            try:
                result=subprocess.run(command,capture_output=True,text=True,timeout=8,check=True)
                f.write(result.stdout);f.flush()
            except Exception as error:
                (path.parent/"telemetry-error.txt").write_text(str(error))
            stop.wait(10)


def launch(manifest_path, output):
    manifest=read(manifest_path);verify(manifest)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    lock=output/"launch.json"
    with lock.open("x") as f:
        json.dump(dict(started_at=now(),job_id=os.environ.get("SLURM_JOB_ID"),
                       manifest_sha256=digest(manifest_path)),f)
    stop=threading.Event()
    thread=threading.Thread(target=telemetry,args=(output/"monitoring/gpu_usage.csv",stop),daemon=True)
    thread.start()
    active=[];remaining=list(manifest["datasets"]);finished=[]
    try:
        single,dual=benchmark(manifest_path,1),benchmark(manifest_path,2)
        concurrency=2 if dual["examples_per_second"] >= 1.10*single["examples_per_second"] else 1
        write_json(output/"throughput.json",dict(single=single,dual=dual,selected_workers=concurrency,
            rule="Choose two only for >=10% aggregate throughput gain; same single allocated GPU."))
        while remaining or active:
            while remaining and len(active)<concurrency:
                ds=remaining.pop(0)
                log=(output/f"worker-{ds}.log").open("a")
                command=[sys.executable,__file__,"worker","--manifest",str(manifest_path),
                         "--output",str(output),"--dataset",ds]
                p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                active.append((ds,p,log))
            for item in list(active):
                ds,p,log=item
                if p.poll() is not None:
                    finished.append(dict(dataset=ds,exit_code=p.returncode))
                    log.close();active.remove(item)
            cells=list((output/"cells").glob("*/*.json"))
            write_json(output/"status.json",dict(checked_at=now(),job_id=os.environ.get("SLURM_JOB_ID"),
                selected_workers=concurrency,active=[ds for ds,_,_ in active],
                finished=finished,cells=len(cells),expected_cells=66))
            if active:time.sleep(5)
        rows=[read(p) for p in sorted((output/"cells").glob("*/*.json"))]
        complete=len(rows)==66 and all(r["status"]=="complete" for r in rows) and all(
            r["exit_code"]==0 for r in finished)
        if complete:
            complete=all(digest(output/r["samples_relative_path"]) == r["samples_sha256"] for r in rows)
        verify(manifest)
        write_json(output/"summary.json",dict(complete=complete,rows=rows,workers=finished,
            manifest_sha256=digest(manifest_path),completed_at=now(),new_training=False))
        if not complete:raise RuntimeError(f"Diagnostics incomplete: {finished}; cells={len(rows)}")
    finally:
        stop.set();thread.join(timeout=12)
        for ds,p,log in active:
            if p.poll() is None:p.terminate()
            log.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode",choices=["worker","launch","verify"])
    ap.add_argument("--manifest",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--dataset")
    ap.add_argument("--device",default="cuda")
    args=ap.parse_args()
    if args.mode == "verify": verify(read(args.manifest));print("Frozen code, inputs, and checkpoints verified.")
    elif args.mode == "worker":worker(args.manifest,args.output,args.dataset,args.device)
    else:launch(args.manifest,args.output)


if __name__ == "__main__":
    main()
