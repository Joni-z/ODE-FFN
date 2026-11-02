"""One persistent, monitored queue per host; failures never trigger auto-restarts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', required=True, choices=['145', 'lenovo'])
    p.add_argument('--device', required=True, choices=['cpu', 'cuda'])
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--detach', action='store_true')
    args = p.parse_args()
    if args.workers not in (1, 2):
        raise ValueError('registered host concurrency is 1 or 2')
    output = Path(args.output).resolve()
    monitor = output/'monitor'
    monitor.mkdir(parents=True, exist_ok=True)
    if (monitor/'queue.lock').exists():
        raise FileExistsError('queue lock exists: inspect it, do not duplicate the queue')
    if args.detach:
        command = [sys.executable, '-u', str(Path(__file__).resolve())] + [x for x in sys.argv[1:] if x != '--detach']
        # Windows OpenSSH owns a job object that otherwise kills descendants
        # when the SSH command exits. A new console/session alone is insufficient.
        flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP |
                 subprocess.CREATE_BREAKAWAY_FROM_JOB) if os.name == 'nt' else 0
        with (monitor/'queue-console.log').open('a') as console:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=console, stderr=console,
                                     creationflags=flags, start_new_session=os.name != 'nt')
        print(json.dumps(dict(queue_pid=child.pid, output=str(output))), flush=True)
        return
    descriptor = os.open(monitor/'queue.lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, 'w') as f:
        f.write(str(os.getpid()))
    grid = json.loads((HERE/'grid.json').read_text())
    pending = [c for c in grid['cells'] if c['host'] == args.host]
    launched, active, done = [], {}, []
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONUNBUFFERED='1')
    started = time.time()
    failed = False
    try:
        while pending or active:
            while pending and len(active) < args.workers and not failed:
                cell = pending.pop(0)
                command = [sys.executable, '-u', str(HERE/'run.py'), '--cell', cell['id'],
                           '--data-root', args.data_root, '--output', str(output), '--device', args.device]
                console = (monitor/f"{cell['id']}.log").open('x')
                proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=console, stderr=console, env=env)
                row = dict(cell=cell['id'], dataset=cell['dataset'], pid=proc.pid, started=time.time(), command=command)
                active[proc.pid] = (proc, console, row)
                launched.append(row)
            for pid, (proc, console, row) in list(active.items()):
                code = proc.poll()
                if code is not None:
                    console.close()
                    complete = output/'muon_pd_20261102'/row['dataset']/row['cell']/'seed000'/'ADAPTER_COMPLETE'
                    row.update(exit_code=code, finished=time.time(), complete=complete.exists())
                    failed |= code != 0 or not complete.exists()
                    done.append(row)
                    del active[pid]
            running = []
            for _, _, row in active.values():
                folder = output/'muon_pd_20261102'/row['dataset']/row['cell']/'seed000'
                last = None
                if (folder/'metrics.jsonl').exists():
                    lines = (folder/'metrics.jsonl').read_text().splitlines()
                    if lines:
                        try:
                            metric = json.loads(lines[-1])
                            last = {k: metric.get(k) for k in ('step', 'validation_oracle_mse', 'wall_seconds')}
                            last['sample'] = metric.get('sample')
                        except json.JSONDecodeError:
                            pass
                running.append(dict(row, latest=last))
            resources = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,power.draw',
                                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
            status = dict(unix_time=time.time(), started=started, queue_pid=os.getpid(),
                          host=args.host, device=args.device, workers=args.workers,
                          cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                          running=running, completed=done, pending=[c['id'] for c in pending],
                          failed=failed, gpu=resources.stdout.strip(), gpu_query_exit=resources.returncode)
            write(monitor/'status.json', status)
            with (monitor/'history.jsonl').open('a') as f:
                f.write(json.dumps(status) + '\n')
            print(json.dumps(dict(time=time.time(), running=[(r['cell'], r['latest']) for r in running],
                                  completed=len(done), pending=len(pending), failed=failed)), flush=True)
            if failed and not active:
                break
            if pending or active:
                time.sleep(30)
        write(monitor/'queue-finished.json', dict(unix_time=time.time(), failed=failed,
                                                 completed=done, unstarted=pending))
    finally:
        # Leave the lock as an execution record. Relaunch requires explicit inspection.
        for proc, console, row in active.values():
            print(f'Queue interrupted; child {proc.pid} must be inspected before recovery', flush=True)
            console.close()
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
