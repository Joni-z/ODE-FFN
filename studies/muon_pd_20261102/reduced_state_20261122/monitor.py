"""Run and monitor the Mac mini study through both held-out sample streams."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent

def read(path):
    return json.loads(path.read_text())

def launch(output, prep, cell):
    key, dataset = cell["id"], cell["dataset"]
    folder = output / "train/report4_reduced_state" / dataset / key / "seed000"
    data = prep / "data/synthetic"
    assets = prep / "coefficient-20261116/assets"
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    def execute(script, args, log, anchors):
        with (output / log).open("a") as handle:
            proc = subprocess.Popen([sys.executable, "-u", str(HERE / script), *map(str, args)],
                                    stdout=handle, stderr=subprocess.STDOUT, env=env)
            anchors["pid"] = proc.pid
            return proc.wait()

    state = dict(id=key, phase="training", target=20000, folder=str(folder))
    live[key] = state
    if not (folder / "ADAPTER_COMPLETE").exists():
        command = ["--cell", key, "--data-root", data, "--output", output / "train", "--device", "cpu"]
        snapshots = sorted((folder / "checkpoints").glob("step_*.pt"))
        if snapshots:
            command += ["--resume", snapshots[-1]]
        code = execute("reduced_trial.py", command, f"{key}.log", state)
        if code:
            raise RuntimeError(f"{key} training exit {code}")
    for split in ("selection", "confirmation"):
        dest = output / split / key
        state.update(phase=split, evaluation=str(dest))
        if (dest / "COMPLETE.json").exists():
            continue
        if dest.exists():
            dest.rename(dest.with_name(dest.name + f"-partial-{time.time_ns()}"))
        args = ["--folder", folder, "--data-root", data, "--assets", assets, "--output", dest,
                "--device", "cpu", "--split", split, "--steps", "20000"]
        code = execute("evaluate.py", args, f"{key}-{split}.log", state)
        if code:
            raise RuntimeError(f"{key} {split} exit {code}")
    state.update(phase="complete")

def snapshot(output):
    rows = []
    for state in tuple(live.values()):
        row = dict(state)
        metrics = Path(row["folder"]) / "metrics.jsonl"
        if metrics.is_file():
            with metrics.open("rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 65536))
                for line in reversed(f.read().splitlines()):
                    try:
                        row["step"] = json.loads(line)["step"]
                        break
                    except (ValueError, KeyError):
                        continue
        if row["phase"] in ("selection", "confirmation"):
            progress = Path(row["evaluation"]) / "progress.json"
            if progress.is_file():
                row["banks_completed"] = read(progress)["banks_completed"]
        rows.append(row)
    tmp = output / "progress.tmp"
    tmp.write_text(json.dumps(dict(pid=os.getpid(), at=time.time(), cases=rows),indent=2)+"\n")
    tmp.replace(output / "progress.json")


live = {}
if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--workers", type=int, default=6)
    options = p.parse_args()
    out = options.output
    out.mkdir(exist_ok=True, parents=True)
    import fcntl
    with (out / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with ThreadPoolExecutor(max_workers=options.workers) as pool:
            cases = read(HERE / "grid.json")["cells"]
            futures = [pool.submit(launch, out, out.parent, case) for case in cases]
            while not all(f.done() for f in futures):
                snapshot(out)
                time.sleep(10)
            snapshot(out)
            for task in futures:
                task.result()
