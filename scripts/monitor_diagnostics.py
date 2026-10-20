"""Detached Slurm/result monitor for the single-GPU frozen-order experiment."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.monitor_stage import TERMINAL, now, parse_accounting, write_json


def probe(job, output):
    output=Path(output)
    r=subprocess.run(["sacct","-n","-P","-X","-j",job,
        "--format=JobID,State,ExitCode,Elapsed"],capture_output=True,text=True,check=True,timeout=20)
    state=parse_accounting(r.stdout).get(job,dict(state="UNKNOWN",exit_code="",elapsed=""))
    cells=list((output/"cells").glob("*/*.json"))
    summary=json.loads((output/"summary.json").read_text()) if (output/"summary.json").exists() else None
    progress=json.loads((output/"status.json").read_text()) if (output/"status.json").exists() else None
    utilization=output/"monitoring/gpu_usage.csv"
    gpu_last=utilization.read_text().splitlines()[-1] if utilization.exists() else None
    terminal=state["state"] in TERMINAL
    complete=bool(summary and summary.get("complete"))
    result=dict(checked_at=now(),job_id=job,**state,terminal=terminal,
        cells=len(cells),expected_cells=66,complete=complete,progress=progress,
        last_gpu_sample=gpu_last,
        successful=terminal and state["state"]=="COMPLETED" and state["exit_code"]=="0:0" and complete)
    print(json.dumps(result))


def watch(args):
    args.local_dir.mkdir(parents=True,exist_ok=True)
    import fcntl,os
    with (args.local_dir/"watch.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(args.local_dir/"watcher.json",dict(pid=os.getpid(),started_at=now(),job_id=args.job))
        signature=None;last_change=time.monotonic()
        while True:
            command=shlex.join([args.remote_python,args.remote_script,"probe","--job",args.job,
                                "--output",args.output])
            try:
                p=subprocess.run(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=15",
                    args.host,command],capture_output=True,text=True,check=True,timeout=45)
                value=json.loads(p.stdout)
                current=(value["state"],value["cells"])
                if current != signature:signature=current;last_change=time.monotonic()
                value["stall_warning"]=value["state"]=="RUNNING" and time.monotonic()-last_change>900
                write_json(args.local_dir/"latest.json",value)
                print(json.dumps(value),flush=True)
                if value["terminal"]:return 0 if value["successful"] else 2
            except Exception as error:
                write_json(args.local_dir/"latest.json",dict(checked_at=now(),state="MONITOR_DISCONNECTED",
                    job_id=args.job,error=str(error),terminal=False))
                print(f"{now()} MONITOR_DISCONNECTED: {error}",flush=True)
            time.sleep(60)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode",choices=["probe","watch"])
    ap.add_argument("--job",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--host",default="torch")
    ap.add_argument("--remote-python",default="/scratch/zz5070/ODE_FFN/envs/synthetic/bin/python")
    ap.add_argument("--remote-script",default="/scratch/zz5070/ODE_FFN/jit-codebase/scripts/monitor_diagnostics.py")
    ap.add_argument("--local-dir",type=Path,default=Path("monitoring"))
    args=ap.parse_args()
    if not args.job.isdigit():ap.error("numeric Slurm job ID required")
    if args.mode=="probe":probe(args.job,args.output)
    else:raise SystemExit(watch(args))


if __name__=="__main__":main()
