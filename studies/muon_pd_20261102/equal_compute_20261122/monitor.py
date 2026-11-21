"""Supervise all four continuations, then run the existing confirmation evaluator."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BUDGET = REPO / "docs/report4/equal-compute/budget.json"
EVAL = HERE.parent / "response_screen_20261115/evaluate_screen.py"


def read(p):
    return json.loads(p.read_text())


def save(p, obj):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, p)


def last_step(folder):
    try:
        with (folder / "metrics.jsonl").open("rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 131072))
            lines = f.read().splitlines()
        return json.loads(lines[-1])["step"]
    except (OSError, IndexError, KeyError, ValueError):
        return 20000


def supervise(work, assets):
    work.mkdir(parents=True, exist_ok=True)
    rows = [r for r in read(BUDGET)["rows"] if r["total_steps"] > 20000]
    assert len(rows) == 4
    children, status, retries = {}, {}, {}
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
    for row in rows:
        status[row['id']] = dict(phase='waiting', step=20000, target=row['total_steps'])
    while True:
        for row in rows:
            key = row["id"]
            state = status[key]
            source = Path(row["source_folder"])
            cfg = read(source / "config.json")
            run_dir = work / "trained" / cfg["experiment_id"] / row["dataset"] / key / "seed000"
            state["step"] = max(state["step"], last_step(run_dir))
            if state["phase"] in ("complete", "failed"):
                continue
            child = children.get(key)
            if child:
                code = child.poll()
                if code is None:
                    if state["phase"] == "evaluation":
                        progress = Path(state["output"]) / "progress.json"
                        if progress.exists():
                            state["banks_completed"] = read(progress)["banks_completed"]
                    continue
                del children[key]
                if code:
                    token = (key, state["phase"])
                    retries[token] = retries.get(token, 0) + 1
                    if retries[token] > 1:
                        state.update(phase="failed", exit_code=code)
                        continue
            if (work / "trained-endpoints" / (key + ".json")).is_file():
                ready = sorted((work / "confirmation").glob(key + "-*/COMPLETE.json"))
                if ready:
                    assert read(ready[-1])["steps"] == [row["total_steps"]]
                    state.update(phase="complete", step=row["total_steps"], banks_completed=8,
                                 output=str(ready[-1].parent))
                    continue
                parent = work / "confirmation"
                parent.mkdir(exist_ok=True)
                out = parent / (key + "-" + str(time.time_ns()))
                cmd = [sys.executable, "-u", str(EVAL), "--folder", str(run_dir),
                       "--data-root", str(Path(cfg["data_root"]).parent), "--assets", str(assets),
                       "--output", str(out), "--device", "cpu", "--split", "confirmation",
                       "--steps", str(row["total_steps"])]
                state.update(phase="evaluation", step=row["total_steps"], output=str(out))
            else:
                cmd = [sys.executable, "-u", str(HERE / "continue_run.py"), str(work), key]
                state["phase"] = "training"
            with (work / (key + ".log")).open("a") as log:
                p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
            state["pid"] = p.pid
            children[key] = p
        snapshot = dict(monitor_pid=os.getpid(), unix_time=time.time(), rows=status)
        save(work / "progress.json", snapshot)
        if all(s["phase"] in ("complete", "failed") for s in status.values()):
            return 0 if all(s["phase"] == "complete" for s in status.values()) else 1
        time.sleep(15)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("assets", type=Path)
    a = p.parse_args()
    import fcntl
    a.work.mkdir(parents=True, exist_ok=True)
    with (a.work / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sys.exit(supervise(a.work, a.assets))
