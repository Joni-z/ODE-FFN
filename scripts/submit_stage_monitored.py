"""Submit frozen stage scripts and automatically detach a local monitor.

Each submission has a unique local receipt. A partial receipt is never retried
automatically, preventing duplicate GPU jobs after interrupted SSH responses.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from monitor_stage import now, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--host", default="torch")
    parser.add_argument("--remote-python", default="/scratch/zz5070/ODE_FFN/envs/synthetic/bin/python")
    parser.add_argument("--remote-script", default="/scratch/zz5070/ODE_FFN/jit-codebase/scripts/monitor_stage.py")
    parser.add_argument("--output-root", default="/scratch/zz5070/ODE_FFN/runs/synthetic")
    parser.add_argument("--train-script", default="scripts/run_stage_waves.slurm")
    parser.add_argument("--report-script", default="scripts/finalize_stage_report.slurm")
    args = parser.parse_args()
    args.local_dir.mkdir(parents=True, exist_ok=True)
    receipt = args.local_dir / "submission.json"
    data = dict(created_at_utc=now(), runtime=args.runtime, report_root=args.report_root,
                state="PREPARING", host=args.host)
    # Exclusive creation must precede any remote mutation.
    with receipt.open("x") as stream:
        json.dump(data, stream, indent=2)

    def remote(argv):
        command = "cd " + shlex.quote(args.runtime) + " && " + shlex.join(argv)
        return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                               args.host, command], capture_output=True, text=True,
                              check=True, timeout=60).stdout.strip()

    def submit(argv):
        reply = remote(argv)
        job = reply.split(";")[0]
        if not job.isdigit():
            raise RuntimeError(f"Unexpected sbatch reply: {reply!r}; inspect Slurm before retrying")
        return job

    try:
        remote([args.remote_python, "scripts/verify_stage_sources.py"])
        # Also protect submissions whose older frozen verifier checked only code.
        remote([args.remote_python, "-c",
                "import hashlib,json,pathlib; p=pathlib.Path('.'); "
                "m=json.loads((p/'source-manifest.json').read_text()); "
                "a=hashlib.sha256((p/'data/synthetic/v1/manifest.json').read_bytes()).hexdigest(); "
                "assert a==m['data_manifest_sha256'], 'Execution-host data manifest differs from snapshot annotation'"])
        remote([args.remote_python, args.remote_script, "--help"])
        data["state"] = "SUBMITTING_TRAIN"
        write_json(receipt, data)
        data["training_array_job_id"] = submit(["sbatch", "--parsable", args.train_script])
        data["state"] = "SUBMITTING_REPORT"
        write_json(receipt, data)
        data["report_job_id"] = submit(["sbatch", "--parsable",
            "--dependency=afterany:" + data["training_array_job_id"], args.report_script])
        data["state"] = "SUBMITTED"
        write_json(receipt, data)
        command = [sys.executable, str(Path(__file__).with_name("monitor_stage.py")),
                   "watch", "--host", args.host, "--runtime", args.runtime,
                   "--manifest", args.manifest, "--report-root", args.report_root,
                   "--output-root", args.output_root, "--remote-python", args.remote_python,
                   "--remote-script", args.remote_script,
                   "--train-job", data["training_array_job_id"],
                   "--report-job", data["report_job_id"],
                   "--local-dir", str(args.local_dir.resolve())]
        with (args.local_dir / "monitor.log").open("a") as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        data["monitor_pid"] = child.pid
        data["state"] = "SUBMITTED_MONITOR_LAUNCHED"
        write_json(receipt, data)
        print(json.dumps(data, indent=2))
    except Exception as error:
        data["error"] = str(error)
        data["requires_reconciliation"] = True
        write_json(receipt, data)
        raise


if __name__ == "__main__":
    main()
