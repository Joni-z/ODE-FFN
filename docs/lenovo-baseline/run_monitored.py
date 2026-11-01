"""Run the upstream CLI once, recording process/GPU status outside the repo."""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    a = parser.parse_args()
    a.config = a.config.resolve()
    config = json.loads(a.config.read_text())
    if config['status'] != 'REGISTERED':
        raise RuntimeError('Training configuration has not been finalized')
    # Refuse an existing run. Recovery must be an explicit upstream --resume.
    a.run_dir.mkdir(parents=True, exist_ok=False)
    os.environ.update(config['environment'])
    executable = Path(sys.executable).resolve()
    command = [str(executable), '-u', str(ROOT / 'main_jit.py'),
               *config['arguments'], '--output_dir', str(a.run_dir / 'output')]
    git = config['git_executable']
    commit = subprocess.check_output([git, '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output([git, '-C', str(ROOT), 'status', '--porcelain'], text=True).strip()
    if dirty:
        raise RuntimeError('Refusing an uncommitted source checkout: ' + dirty)
    paths = [*ROOT.glob('*.py'), *ROOT.joinpath('util').glob('*.py'), a.config,
             Path(__file__).resolve()]
    binding = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    write_json(a.run_dir / 'launch.json', dict(commit=commit, command=command,
               configuration=config, source_sha256=binding, supervisor_pid=os.getpid()))
    # Request that Windows not enter idle sleep while this process is active.
    awake = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001) if os.name == 'nt' else None
    try:
        with (a.run_dir / 'train.log').open('x', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            while True:
                exit_code = process.poll()
                try:
                    gpu = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                                          '--format=csv,noheader,nounits'], capture_output=True,
                                         text=True, timeout=10)
                    telemetry = gpu.stdout.strip() if gpu.returncode == 0 else gpu.stderr.strip()
                except (OSError, subprocess.TimeoutExpired) as exc:
                    telemetry = str(exc)
                with (a.run_dir / 'train.log').open('rb') as read_log:
                    read_log.seek(max(0, read_log.seek(0, 2) - 8192))
                    recent = read_log.read().decode('utf-8', errors='replace').splitlines()[-6:]
                status = dict(utc=datetime.now(timezone.utc).isoformat(), pid=process.pid,
                              supervisor_pid=os.getpid(), state='RUNNING' if exit_code is None else 'EXITED',
                              exit_code=exit_code, gpu_csv=telemetry, log_tail=recent,
                              idle_sleep_request=bool(awake))
                write_json(a.run_dir / 'status.json', status)
                with (a.run_dir / 'monitor.jsonl').open('a') as monitor:
                    monitor.write(json.dumps(status) + '\n')
                if exit_code is not None:
                    write_json(a.run_dir / 'process-exit.json', status)
                    break
                time.sleep(30)
        raise SystemExit(exit_code)
    finally:
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()
