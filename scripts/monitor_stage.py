"""Monitor Slurm stage arrays and result artifacts; never mutate training runs.

probe runs on torch; watch runs locally and survives the interactive SSH session.
Only stdout/files are used: no email, messaging service or automatic resubmission.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
            "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def read_json(path):
    return json.loads(path.read_text())


def state_name(value):
    return value.split()[0].rstrip("+") if value else "UNKNOWN"


def parse_accounting(text):
    rows = {}
    for line in text.splitlines():
        fields = line.split("|")
        if len(fields) >= 4:
            rows[fields[0]] = dict(state=state_name(fields[1]),
                                   exit_code=fields[2], elapsed=fields[3])
    return rows


def outcome(jobs, artifact_complete):
    if not jobs or any(j["state"] not in TERMINAL for j in jobs):
        return "RUNNING_OR_PENDING", False
    if any(j["state"] != "COMPLETED" or j["exit_code"] != "0:0" for j in jobs):
        return "FAILED", True
    return ("SUCCESS" if artifact_complete else "INCOMPLETE"), True


def probe(args):
    root = Path(args.runtime)
    report = Path(args.report_root)
    manifest = read_json(root / args.manifest)
    ids = [f"{args.train_job}_{i}" for i in range(len(manifest["datasets"]))]
    ids.append(args.report_job)
    command = ["sacct", "-n", "-P", "-X", "-j",
               f"{args.train_job},{args.report_job}",
               "--format=JobID,State,ExitCode,Elapsed"]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=25)
    states = parse_accounting(result.stdout)
    jobs = [{"id": job, **states.get(job, dict(state="UNKNOWN", exit_code="", elapsed=""))}
            for job in ids]
    queue = subprocess.run(["squeue", "-h", "-r", "-j",
                            f"{args.train_job},{args.report_job}", "-o", "%i|%T|%M|%R"],
                           capture_output=True, text=True, timeout=25)
    progress, issues, failures = [], [], []
    diagnostics = 0
    for ds in manifest["datasets"]:
        for entry in manifest["entries"]:
            cfg = read_json(root / entry["config"])
            folder = Path(args.output_root) / cfg["experiment_id"] / ds / entry["model"] / "seed000"
            last = None
            rows = folder / "metrics.jsonl"
            if rows.exists():
                # Ignore a partially written last record.
                data = rows.read_text()
                lines = data.splitlines()
                if data and not data.endswith("\n"):
                    lines = lines[:-1]
                if lines:
                    last = json.loads(lines[-1])
            marker = folder / "COMPLETE"
            complete = (marker.exists() and marker.read_text().strip() == str(entry["steps"])
                        and last is not None and last["step"] == entry["steps"]
                        and (folder / "checkpoints" / f"step_{entry['steps']:07d}.pt").exists())
            if complete and read_json(folder / "config.json") != cfg:
                complete = False
                issues.append(f"{ds}/{entry['label']}: config mismatch")
            progress.append(dict(dataset=ds, label=entry["label"], complete=complete,
                                 step=last["step"] if last else None, target_steps=entry["steps"]))
            failure = report / "failures" / ds / (entry["label"] + ".json")
            if failure.exists():
                failures.append(read_json(failure))
            for step in entry["evaluate_steps"]:
                path = report / "diagnostics" / ds / f"{entry['label']}-step{step:07d}.json"
                if path.exists():
                    r = read_json(path)
                    if (r["dataset"], r["label"], r["step"], r["seed"], r["config_sha256"]) == (
                            ds, entry["label"], step, 0, entry["sha256"]):
                        diagnostics += 1
                    else:
                        issues.append(f"diagnostic identity mismatch: {path}")
    summary = report / "stage-summary.json"
    report_complete = summary.exists() and read_json(summary).get("complete") is True
    done = sum(p["complete"] for p in progress)
    artifacts_complete = (done == manifest["run_count"]
                          and diagnostics == manifest["eval_checkpoint_count"]
                          and report_complete and not issues and not failures)
    status, terminal = outcome(jobs, artifacts_complete)
    snapshot = dict(checked_at_utc=now(), status=status, terminal=terminal, jobs=jobs,
                    queue=queue.stdout.strip(), queue_error=queue.stderr.strip(),
                    completed_runs=done, expected_runs=manifest["run_count"],
                    completed_diagnostics=diagnostics,
                    expected_diagnostics=manifest["eval_checkpoint_count"],
                    report_complete=report_complete, issues=issues, failures=failures,
                    progress=progress, report_path=str(report / "stage-report.md"))
    monitor_dir = report / "monitoring"
    write_json(monitor_dir / "latest.json", snapshot)
    text = [f"# Experiment monitor: {status}", "", f"Checked UTC: {snapshot['checked_at_utc']}",
            f"Training: {done}/{manifest['run_count']}; diagnostics: "
            f"{diagnostics}/{manifest['eval_checkpoint_count']}; report complete: {report_complete}", ""]
    text += [f"- {j['id']}: {j['state']}, exit={j['exit_code']}, elapsed={j['elapsed']}" for j in jobs]
    text += ["", "Monitor validates identities/completion markers; full checkpoint hash audit is separate."]
    (monitor_dir / "status.md").write_text("\n".join(text) + "\n")
    return snapshot


def remote_probe_command(args):
    values = [args.remote_python, args.remote_script, "probe",
              "--runtime", args.runtime, "--manifest", args.manifest,
              "--report-root", args.report_root, "--output-root", args.output_root,
              "--train-job", args.train_job, "--report-job", args.report_job]
    return shlex.join(values)


def watch(args):
    local = Path(args.local_dir)
    local.mkdir(parents=True, exist_ok=True)
    lock = local / "watch.lock"
    # POSIX advisory lock is automatically released on termination.
    import fcntl
    with lock.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A monitor is already watching this submission.")
        return watch_locked(args, local)


def watch_locked(args, local):
    write_json(local / "watcher.json", dict(pid=os.getpid(), started_at_utc=now(),
                                          host=args.host, train_job=args.train_job,
                                          report_job=args.report_job, interval_seconds=args.interval))
    start = time.monotonic()
    last_signature, last_progress_change, consecutive_errors = None, time.monotonic(), 0
    while True:
        try:
            result = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                 "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
                 args.host, remote_probe_command(args)],
                capture_output=True, text=True, timeout=90, check=True)
            value = json.loads(result.stdout)
            consecutive_errors = 0
            # Elapsed time alone does not count as experiment progress.
            signature = json.dumps([[(j["id"], j["state"]) for j in value["jobs"]],
                                    value["progress"], value["completed_diagnostics"],
                                    value["report_complete"]], sort_keys=True)
            if signature != last_signature:
                last_signature, last_progress_change = signature, time.monotonic()
            value["seconds_without_progress"] = round(time.monotonic() - last_progress_change)
            active = any(j["state"] == "RUNNING" for j in value["jobs"])
            value["stall_warning"] = active and value["seconds_without_progress"] >= args.stall_minutes * 60
            value["failure_warning"] = bool(value["failures"] or value["issues"]) or any(
                j["state"] in TERMINAL and j["state"] != "COMPLETED" for j in value["jobs"])
            write_json(local / "latest.json", value)
            print(f"{now()} {value['status']} runs={value['completed_runs']}/{value['expected_runs']} "
                  f"diagnostics={value['completed_diagnostics']}/{value['expected_diagnostics']} "
                  f"stall_warning={value['stall_warning']} failure_warning={value['failure_warning']}", flush=True)
            with (local / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(value, allow_nan=False) + "\n")
            if value["terminal"]:
                return 0 if value["status"] == "SUCCESS" else 2
        except (subprocess.SubprocessError, OSError, ValueError) as error:
            consecutive_errors += 1
            message = str(error)
            if isinstance(error, subprocess.CalledProcessError):
                message = error.stderr.strip() or message
            value = dict(checked_at_utc=now(), status="MONITOR_DISCONNECTED",
                         terminal=False, error=message[:2000],
                         consecutive_errors=consecutive_errors)
            write_json(local / "latest.json", value)
            with (local / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(value) + "\n")
            print(f"{now()} MONITOR_DISCONNECTED ({consecutive_errors}): {message[:500]}", flush=True)
        if args.once:
            return 0
        if time.monotonic() - start >= args.max_hours * 3600:
            value = dict(checked_at_utc=now(), status="MONITOR_EXPIRED", terminal=False)
            write_json(local / "latest.json", value)
            print(f"{now()} MONITOR_EXPIRED; job success has NOT been established", flush=True)
            return 3
        time.sleep(args.interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["probe", "watch"])
    parser.add_argument("--train-job", required=True)
    parser.add_argument("--report-job", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--manifest", default="configs/synthetic/stage_20261020/manifest.json")
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--output-root", default="/scratch/zz5070/ODE_FFN/runs/synthetic")
    parser.add_argument("--host", default="torch")
    parser.add_argument("--remote-python", default="/scratch/zz5070/ODE_FFN/envs/synthetic/bin/python")
    parser.add_argument("--remote-script", default="/scratch/zz5070/ODE_FFN/jit-codebase/scripts/monitor_stage.py")
    parser.add_argument("--local-dir", default="monitoring")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--stall-minutes", type=int, default=15)
    parser.add_argument("--max-hours", type=float, default=24)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not args.train_job.isdigit() or not args.report_job.isdigit():
        parser.error("job IDs must be numeric")
    if args.interval < 30 or args.stall_minutes <= 0 or args.max_hours <= 0:
        parser.error("interval >=30 seconds; stall-minutes and max-hours must be positive")
    if args.mode == "probe":
        print(json.dumps(probe(args), allow_nan=False))
    else:
        raise SystemExit(watch(args))


if __name__ == "__main__":
    main()
