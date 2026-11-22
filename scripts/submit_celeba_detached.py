"""Schedule the remaining training stages and their independent paired evaluations."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys

from submit_celeba_midpoint import save

ROOT = Path('/scratch/zz5070/ODE_FFN')
OPS = ROOT / 'operations/celeba-midpoint-detached'
RUN = ROOT / 'runs/celeba/20261123-midpoint4-detached-s0'
CODE = Path(__file__).resolve().parents[1]


def submit():
    proof = json.loads((OPS / 'cuda-check.json').read_text())
    assert proof['status'] == 'PASS'
    cutoff = json.loads((OPS / 'handoff.json').read_text())
    epoch = cutoff['epoch']
    assert (RUN / 'checkpoint-last.pth').is_file()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE, text=True).strip()
    assert (RUN / 'source-commit.txt').read_text().strip() == commit
    path = OPS / 'launch.json'
    record = json.loads(path.read_text()) if path.exists() else dict(source_commit=commit, jobs=[])
    assert record['source_commit'] == commit
    old_run = Path(cutoff['original_run'])
    previous_train = None
    for target in range(20, 161, 20):
        if target <= epoch and (old_run / 'paired' / f'epoch-{target:04d}/comparison.json').exists():
            continue
        phases = ['train', 'fid'] if target > epoch else ['fid']
        if target <= epoch:
            assert (RUN / f'checkpoint-{target:04d}.pth').is_file()
        for phase in phases:
            cached = next((j for j in record['jobs'] if (j['phase'], j['target']) == (phase, target)), None)
            if cached:
                if phase == 'train':
                    previous_train = cached['job_id']
                continue
            name = f'mid-{phase}-{target}'
            command = ['sbatch', '--parsable', '--kill-on-invalid-dep=yes', f'--job-name={name}',
                       f'--export=ALL,ODE_FFN_CODE={CODE}', '--time=04:00:00']
            if previous_train:
                command.append('--dependency=afterok:' + previous_train)
            command += [str(CODE / 'slurm/celeba_midpoint_stage.slurm'), phase, str(target)]
            job = subprocess.check_output(command, text=True).strip().split(';')[0]
            assert job.isdigit()
            output = RUN if phase == 'train' else RUN / 'paired' / f'epoch-{target:04d}'
            record['jobs'].append(dict(phase=phase, target=target, output=str(output), job_id=job, command=command))
            save(path, record)
            if phase == 'train':
                previous_train = job
    pid = record.get('monitor_pid')
    cmdline = Path(f'/proc/{pid}/cmdline')
    if not (pid and cmdline.exists() and str(path).encode() in cmdline.read_bytes()):
        with (OPS / 'monitor.log').open('a') as log:
            monitor = subprocess.Popen([sys.executable, '-u', str(CODE / 'scripts/monitor_slurm.py'),
                                        '--launch', str(path), '--interval', '60'],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        record['monitor_pid'] = monitor.pid
        save(path, record)
    print([(j['phase'], j['target'], j['job_id']) for j in record['jobs']], flush=True)


if __name__ == '__main__':
    with (OPS / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        submit()
