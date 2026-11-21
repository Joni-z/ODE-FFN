"""Submit dependent Torch allocations for a single midpoint trajectory and attach its monitor."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/scratch/zz5070/ODE_FFN')
CODE = Path(__file__).resolve().parents[1]
OPS = ROOT / 'operations/celeba128-midpoint-20261122'
OUT = ROOT / 'runs/celeba/20261122-midpoint4-w1200-s0'
BASE = ROOT / 'runs/celeba/20261114-b16-swiglu-s0-to160'
LAUNCH = OPS / 'launch.json'


def save(path, record):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(record, indent=2) + '\n')
    temp.replace(path)


def submit():
    commit = subprocess.check_output(['git', '-C', str(CODE), 'rev-parse', 'HEAD'], text=True).strip()
    record = json.loads(LAUNCH.read_text()) if LAUNCH.exists() else dict(source_commit=commit, jobs=[])
    assert commit == record['source_commit']
    assert all((BASE / f'checkpoint-{e:04d}.pth').is_file() for e in (100,120,140,160))
    if not record['jobs'] and OUT.exists():
        raise RuntimeError(f'Inspect existing run before a new submission: {OUT}')
    (ROOT/'runs/celeba/slurm').mkdir(exist_ok=True, parents=True)
    for target in (40,80,120,160):
        if any(row['target'] == target for row in record['jobs']):
            continue
        cmd = ['sbatch','--parsable',f'--job-name=midpt-s0-e{target}', '--time=12:00:00',
               '--mem=40G','--kill-on-invalid-dep=yes',
               f'--export=ALL,ODE_FFN_CODE={CODE},ODE_FFN_EPOCHS={target},ODE_FFN_FRESH_START={int(target==40)},ODE_FFN_MAX_TRAIN_SECONDS=0']
        if record['jobs']:
            cmd.append('--dependency=afterok:' + record['jobs'][-1]['job_id'])
        cmd += [str(CODE/'slurm/celeba.slurm'), 'midpoint', 'celeba128-midpoint']
        job = subprocess.check_output(cmd, text=True).strip().split(';')[0]
        assert job.isdigit()
        record['jobs'].append(dict(job_id=job, target=target, output=str(OUT), command=cmd))
        save(LAUNCH, record)
        print('SUBMITTED', job, target, flush=True)
    # Idempotent across repeated successful invocations.
    pid = record.get('monitor_pid')
    if pid and Path(f'/proc/{pid}/cmdline').exists() and b'monitor_slurm.py' in Path(f'/proc/{pid}/cmdline').read_bytes():
        return
    with (OPS/'monitor.log').open('a') as log:
        proc = subprocess.Popen([sys.executable,'-u',str(CODE/'scripts/monitor_slurm.py'),
                                 '--launch',str(LAUNCH),'--interval','60'],
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    record['monitor_pid'] = proc.pid
    save(LAUNCH, record)
    print('MONITOR', proc.pid, flush=True)


if __name__ == '__main__':
    OPS.mkdir(parents=True, exist_ok=True)
    with (OPS/'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        submit()
