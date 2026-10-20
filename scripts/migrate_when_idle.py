"""Run the frozen diagnostics on an idle standalone GPU, or retain the Slurm run.

The coordinator runs on the Mac mini. Remote probe/start/run commands use only
the standard library until the existing scientific runner is launched.
"""
import argparse
import csv
import datetime
import io
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def command(args, timeout=45):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{shlex.join(args)} exited {result.returncode}: {result.stderr.strip()}")
    return result.stdout.strip()


def ssh(host, args):
    return command(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                    host, shlex.join(args)], timeout=60)


def gpu_inventory():
    rows = csv.reader(io.StringIO(command([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"])))
    gpus = {}
    for row in rows:
        index, uuid, name, memory, util = [x.strip() for x in row]
        gpus[uuid] = dict(index=int(index), uuid=uuid, name=name,
                          memory_mib=float(memory), utilization=float(util), pids=[])
    processes = command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                         "--format=csv,noheader,nounits"])
    for row in csv.reader(io.StringIO(processes)):
        if len(row) == 2 and row[0].strip() in gpus:
            gpus[row[0].strip()]["pids"].append(int(row[1].strip()))
    return list(gpus.values())


def idle(gpu):
    # Low instantaneous utilization alone never qualifies an occupied GPU.
    return not gpu["pids"] and gpu["memory_mib"] < 256 and gpu["utilization"] <= 1


def probe(cfg):
    output = Path(cfg["output"])
    runner = read(output/"runner.json") if (output/"runner.json").exists() else None
    progress = read(output/"status.json") if (output/"status.json").exists() else None
    summary = read(output/"summary.json") if (output/"summary.json").exists() else None
    return dict(checked_at=now(), gpus=gpu_inventory(), runner=runner, progress=progress,
                cells=len(list((output/"cells").glob("*/*.json"))),
                summary_complete=bool(summary and summary.get("complete")))


def start(cfg, gpu):
    # Recheck immediately before dispatch, after the coordinator's idle samples.
    assert any(g["uuid"] == gpu and idle(g) for g in gpu_inventory()), "GPU no longer idle"
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    with (output/"dispatch.json").open("x") as f:
        json.dump(dict(created_at=now(), gpu_uuid=gpu, config=cfg), f, indent=2)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    with (output/"runner.log").open("a") as log:
        p = subprocess.Popen([cfg["python"], cfg["remote_script"], "run",
                              "--config", cfg["remote_config"], "--gpu", gpu],
                             stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                             start_new_session=True, env=env)
    receipt = dict(pid=p.pid, started_at=now(), gpu_uuid=gpu)
    write(output/"pid.json", receipt)
    return receipt


def run(cfg, gpu):
    output = Path(cfg["output"])
    state = dict(pid=os.getpid(), started_at=now(), gpu_uuid=gpu,
                 python=cfg["python"], state="VERIFYING")
    write(output/"runner.json", state)
    p = None
    def interrupted(signum, frame):
        raise RuntimeError(f"Standalone runner received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        args = [cfg["python"], str(Path(cfg["runtime"])/"scripts/run_frozen_order.py"),
                "verify", "--manifest", cfg["manifest"], "--output", cfg["output"]]
        subprocess.run(args, check=True, timeout=120, cwd=cfg["runtime"])
        assert any(g["uuid"] == gpu and idle(g) for g in gpu_inventory()), \
            "GPU became occupied during preflight; no GPU evaluation launched"
        args[2] = "launch"
        p = subprocess.Popen(args, cwd=cfg["runtime"], start_new_session=True)
        state.update(state="RUNNING", child_pid=p.pid)
        write(output/"runner.json", state)
        code = p.wait(timeout=cfg["time_limit_seconds"])
        summary = read(output/"summary.json") if (output/"summary.json").exists() else {}
        assert code == 0 and summary.get("complete") and len(summary.get("rows", [])) == 66, \
            f"Diagnostics incomplete; exit={code}"
        state.update(state="SUCCEEDED", exit_code=code, completed_at=now())
    except BaseException as error:
        state.update(state="FAILED", error=repr(error), completed_at=now())
    finally:
        if p is not None and p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        write(output/"runner.json", state)
    return 0 if state["state"] == "SUCCEEDED" else 2


def remote(cfg, mode, gpu=None):
    args = [cfg["python"], cfg["remote_script"], mode, "--config", cfg["remote_config"]]
    if gpu:
        args += ["--gpu", gpu]
    if cfg.get("target_gateway"):
        # Authenticate on the gateway using its existing target SSH setup.
        # ProxyJump alone would still use credentials from the local machine.
        inner = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                 cfg["target_host"], shlex.join(args)]
        text = command(["ssh", "-o", "BatchMode=yes", "-o", "ControlPath=none",
                        "-o", "ConnectTimeout=15", cfg["target_gateway"],
                        shlex.join(inner)], timeout=60)
    else:
        text = ssh(cfg["target_host"], args)
    return json.loads(text)


def origin_status(cfg):
    text = ssh(cfg["torch_host"], ["squeue", "-h", "-j", cfg["origin_job_id"],
                                   "-o", "%T|%r"])
    if not text:
        return "LEFT_QUEUE", ""
    state, reason = text.splitlines()[0].split("|", 1)
    return state.strip(), reason.strip()


def record(cfg, phase, **values):
    value = dict(checked_at=now(), phase=phase, origin_job_id=cfg["origin_job_id"], **values)
    write(Path(cfg["local_dir"])/"latest.json", value)
    print(json.dumps(value), flush=True)


def migrate(cfg, gpu):
    job = cfg["origin_job_id"]
    # Holding a still-pending job prevents it from racing the standalone launch.
    try:
        ssh(cfg["torch_host"], ["scontrol", "hold", job])
    except Exception as error:
        record(cfg, "HOLD_FAILED_NO_TARGET_LAUNCHED", error=str(error))
        return 2
    state, reason = origin_status(cfg)
    if state != "PENDING" or reason != "JobHeldUser":
        ssh(cfg["torch_host"], ["scontrol", "release", job])
        record(cfg, "ORIGIN_CHANGED_BEFORE_MIGRATION", state=state, reason=reason)
        return 2
    record(cfg, "ORIGIN_HELD", gpu_uuid=gpu)
    try:
        receipt = remote(cfg, "start", gpu)
        write(Path(cfg["local_dir"])/"target-dispatch.json", receipt)
    except Exception as error:
        # A lost SSH reply does not establish that remote launch failed.
        record(cfg, "RECONCILING_TARGET_DISPATCH", error=str(error))
    migrated = False
    absent_checks = 0
    while True:
        try:
            target = remote(cfg, "probe")
            runner = target["runner"]
            absent_checks = absent_checks + 1 if runner is None else 0
            if not migrated and ((runner and runner["state"] == "FAILED") or absent_checks >= 3):
                ssh(cfg["torch_host"], ["scontrol", "release", job])
                record(cfg, "TARGET_FAILED_OR_NOT_STARTED_TORCH_RELEASED", target=target)
                return 2
            # A completed first cell proves startup beyond merely spawning a PID.
            if not migrated and target["cells"] >= 1 and runner and runner["state"] in (
                    "RUNNING", "SUCCEEDED"):
                ssh(cfg["torch_host"], ["scancel", "--state=PENDING", job])
                migrated = True
                write(Path(cfg["local_dir"])/"migration.json", dict(
                    migrated_at=now(), origin_job_id=job, gpu_uuid=gpu,
                    target_host=cfg["target_host"], target_output=cfg["output"],
                    source_action="Held pending job, verified target's first cell, then cancelled pending job"))
            record(cfg, "RUNNING_ON_145" if migrated else "VERIFYING_145_STARTUP",
                   target=target)
            if runner and runner["state"] in ("SUCCEEDED", "FAILED"):
                ok = runner["state"] == "SUCCEEDED" and target["summary_complete"]
                record(cfg, "COMPLETE_ON_145" if ok else "FAILED_ON_145", target=target)
                return 0 if ok else 2
        except Exception as error:
            # Keep the Slurm job held until the standalone state is known.
            record(cfg, "TARGET_MONITOR_DISCONNECTED", migrated=migrated, error=str(error))
        time.sleep(10 if not migrated else cfg["interval_seconds"])


def watch(cfg):
    import fcntl
    folder = Path(cfg["local_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/"watch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # A previous dispatch requires reconciliation, not another launch.
        if (folder/"target-dispatch.json").exists() or (folder/"migration.json").exists():
            raise RuntimeError("Previous migration exists; inspect its state before restarting")
        write(folder/"watcher.json", dict(pid=os.getpid(), started_at=now()))
        counts = {}
        while True:
            try:
                state, reason = origin_status(cfg)
                if state != "PENDING" or reason.startswith("JobHeld"):
                    record(cfg, "ORIGIN_SELECTED_OR_CHANGED", state=state, reason=reason)
                    return 0
                target = remote(cfg, "probe")
                if target["runner"] is not None:
                    record(cfg, "EXISTING_TARGET_RUN_REQUIRES_RECONCILIATION", target=target)
                    return 2
                counts = {g["uuid"]: counts.get(g["uuid"], 0) + 1 if idle(g) else 0
                          for g in target["gpus"]}
                record(cfg, "WAITING_FOR_IDLE_GPU", origin_reason=reason,
                       idle_checks=counts, target=target)
                candidates = [g for g in target["gpus"] if counts[g["uuid"]] >= cfg["idle_checks"]]
                if candidates:
                    return migrate(cfg, min(candidates, key=lambda g: g["index"])["uuid"])
            except Exception as error:
                counts = {}
                record(cfg, "WAIT_MONITOR_DISCONNECTED", error=str(error))
            time.sleep(cfg["interval_seconds"])


def monitor(cfg):
    """Monitor an explicitly launched standalone job without changing Slurm."""
    import fcntl
    folder = Path(cfg["local_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/"watch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(folder/"watcher.json", dict(pid=os.getpid(), started_at=now(), mode="monitor"))
        while True:
            try:
                target = remote(cfg, "probe")
                runner = target["runner"]
                record(cfg, "MONITORING_145", target=target)
                if runner and runner["state"] in ("SUCCEEDED", "FAILED"):
                    ok = runner["state"] == "SUCCEEDED" and target["summary_complete"]
                    record(cfg, "COMPLETE_ON_145" if ok else "FAILED_ON_145", target=target)
                    return 0 if ok else 2
            except Exception as error:
                record(cfg, "TARGET_MONITOR_DISCONNECTED", error=str(error))
            time.sleep(cfg["interval_seconds"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["probe", "start", "run", "watch", "monitor"])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--gpu")
    args = ap.parse_args()
    cfg = read(args.config)
    assert cfg["origin_job_id"].isdigit()
    if args.mode == "probe":
        print(json.dumps(probe(cfg)))
    elif args.mode == "start":
        print(json.dumps(start(cfg, args.gpu)))
    elif args.mode == "run":
        raise SystemExit(run(cfg, args.gpu))
    elif args.mode == "monitor":
        raise SystemExit(monitor(cfg))
    else:
        raise SystemExit(watch(cfg))


if __name__ == "__main__":
    main()
