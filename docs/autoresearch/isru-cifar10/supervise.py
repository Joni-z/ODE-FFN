"""Owned local controller: monitor one process, record its actual exit, never restart."""
from __future__ import annotations
import argparse
import os
import subprocess
import time
from pathlib import Path
import psutil
from common import append, load, verify, write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    root = args.root.resolve()
    _, paths = verify(root)
    command = [paths['training_python'], '-B', str(root/'source/train.py'), '--root', str(root)]
    if args.resume:
        command.append('--resume')
    with (root/'output/training.log').open('a') as handle:
        child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
    write(root/'output/process.json', dict(controller_pid=os.getpid(), training_pid=child.pid,
          started=time.time(), command=command, policy='one child; monitor only; no automatic restart'))
    inhibitor = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(child.pid)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    process = psutil.Process(child.pid)
    previous_cpu = 0.
    previous_time = time.monotonic()
    while child.poll() is None:
        now = time.monotonic()
        try:
            cpu = process.cpu_times()
            total = cpu.user+cpu.system
            status_path = root/'output/status.json'
            status = load(status_path) if status_path.exists() else dict(phase='INITIALIZING', step=0)
            row = dict(timestamp=time.time(), pid=child.pid, running=process.is_running(),
                CPU_percent=100*(total-previous_cpu)/max(now-previous_time, 1e-6),
                RSS_bytes=process.memory_info().rss, system_available_bytes=psutil.virtual_memory().available,
                free_disk_bytes=psutil.disk_usage(root).free,
                status_age_seconds=time.time()-status.get('timestamp', time.time()), status=status,
                GPU_utilization_percent=None, GPU_utilization_note='no portable utilization counter; MPS memory/actual throughput in status')
            append(root/'output/monitor.jsonl', row)
            write(root/'output/monitor-latest.json', row)
            previous_cpu, previous_time = total, now
        except psutil.NoSuchProcess:
            break
        time.sleep(30)
    code = child.wait()
    if inhibitor.poll() is None:
        inhibitor.terminate()
    write(root/'output/process-exit.json', dict(pid=child.pid, exit_code=code, timestamp=time.time(),
          status=load(root/'output/status.json') if (root/'output/status.json').exists() else None))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
