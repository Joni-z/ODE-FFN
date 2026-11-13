"""Lightweight queue/progress monitor; never restarts or submits jobs."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=60)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    launch = json.loads(args.launch.read_text())
    expected = {job['job_id']: job for job in launch['jobs']}
    previous = None
    terminal = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
                'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE', 'PREEMPTED'}
    while True:
        query = subprocess.run(['squeue', '-h', '-u', 'zz5070', '-o', '%i|%T|%r|%N'],
                               text=True, capture_output=True, check=True)
        queued = {}
        for line in query.stdout.splitlines():
            fields = line.split('|', 3)
            if len(fields) == 4 and fields[0] in expected:
                queued[fields[0]] = dict(state=fields[1], reason=fields[2], nodes=fields[3])
        states = {}
        if len(queued) < len(expected):
            accounting = subprocess.run([
                'sacct', '-nP', '-j', ','.join(expected), '-o', 'JobIDRaw,State,ExitCode,Elapsed,NodeList'
            ], text=True, capture_output=True, check=True)
            for line in accounting.stdout.splitlines():
                fields = line.split('|')
                if len(fields) >= 5 and fields[0] in expected:
                    states[fields[0]] = dict(state=fields[1].split()[0].rstrip('+'),
                                             exit_code=fields[2], elapsed=fields[3], nodes=fields[4])
        states.update(queued)
        for job_id, job in expected.items():
            output = job.get('output')
            progress = Path(output) / 'progress.json' if output else None
            if progress and progress.exists():
                states.setdefault(job_id, {})['progress'] = json.loads(progress.read_text())
            if output and Path(output).exists():
                from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
                events = EventAccumulator(output, size_guidance={'scalars': 0}).Reload()
                fid = {tag: [{'epoch': e.step, 'value': e.value} for e in events.Scalars(tag)]
                       for tag in events.Tags()['scalars'] if tag.startswith('fid')}
                states.setdefault(job_id, {})['FID'] = fid
                states[job_id]['numbered_checkpoints'] = sorted(
                    p.name for p in Path(output).glob('checkpoint-[0-9]*.pth'))
        snapshot = dict(updated_at=time.time(), source_commit=launch['source_commit'], jobs=states)
        destination = args.launch.parent / 'monitor.json'
        temporary = destination.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(snapshot, indent=2) + '\n')
        temporary.replace(destination)
        signature = {key: (item.get('state'), item.get('reason')) for key, item in states.items()}
        if signature != previous:
            print(json.dumps(snapshot), flush=True)
            previous = signature
        if args.once or all(states.get(key, {}).get('state') in terminal for key in expected):
            return
        time.sleep(max(30.0, args.interval))


if __name__ == '__main__':
    main()
