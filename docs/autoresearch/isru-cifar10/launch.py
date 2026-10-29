"""Detach one monitored run after the documented readiness receipt exists."""
from __future__ import annotations
import argparse
import os
import subprocess
import time
from pathlib import Path
import psutil
from common import load, sha, verify, write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    root = args.root.resolve()
    _, paths = verify(root)
    receipt = load(root/'documented-pass.json')
    assert receipt['status'] == 'PASS' and receipt['bindings_sha256'] == sha(root/'bindings.json')
    previous = root/'output/launch.json'
    if previous.exists():
        old = load(previous)
        if psutil.pid_exists(old['controller_pid']):
            raise RuntimeError('controller PID exists; inspect it, do not duplicate launch')
        assert args.resume, 'existing launch requires explicit resume'
    else:
        assert not args.resume
    command = [paths['training_python'], '-B', str(root/'source/supervise.py'), '--root', str(root)]
    if args.resume:
        command.append('--resume')
    with (root/'output/controller.log').open('a') as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True, cwd=root)
    time.sleep(2)
    assert process.poll() is None, 'controller exited early; inspect controller.log'
    value = dict(controller_pid=process.pid, timestamp=time.time(), command=command,
                 bindings_sha256=sha(root/'bindings.json'), root=str(root), resumed=args.resume)
    write(previous, value)
    print(value, flush=True)


if __name__ == '__main__':
    main()
