"""Four one-thread CPU workers for the bounded density-resolution pilot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def dump(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    for name in ['root', 'data', 'output', 'readiness']:
        p.add_argument('--'+name, required=True)
    p.add_argument('--workers', type=int, choices=[1, 2, 3, 4], default=4)
    p.add_argument('--detach', action='store_true')
    a = p.parse_args()
    root, data, out = map(Path, [a.root, a.data, a.output])
    meta = json.loads((data/'density-data.json').read_text())
    ready = json.loads(Path(a.readiness).read_text())
    digest = hashlib.sha256((data/'density-data.json').read_bytes()).hexdigest()
    assert meta['passed'] and len(meta['references']) == 8
    assert ready['passed'] and ready['device'] == 'cpu' and ready['density_data_sha256'] == digest
    assert len(ready['models']) == 8 and len(ready['entry_witnesses']) == 2
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='-1')
    commit = subprocess.check_output(['git', '-C', str(HERE.parents[2]), 'rev-parse', 'HEAD'], text=True).strip()
    if a.detach:
        assert not (out/'queue-start.json').exists() and not (out/'queue.lock').exists()
        command = [sys.executable, '-u', str(Path(__file__).resolve())]+[v for v in sys.argv[1:] if v != '--detach']
        with (out/'queue-console.log').open('x') as log:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=env, start_new_session=True)
        row = dict(queue_pid=proc.pid, command=command, frozen_commit=commit, started=time.time())
        dump(out/'queue-start.json', row)
        print(json.dumps(row))
        return
    with (out/'queue.lock').open('x') as lock:
        lock.write(str(os.getpid()))
    pending, active, completed = list(enumerate(meta['references'])), {}, []
    failed = False
    while pending or active:
        for pid, (proc, log, row) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close()
            file = Path(row['output'])/'COMPLETE.json'
            done = json.loads(file.read_text()) if file.exists() else {}
            passed = done.get('passed') and done.get('pilot') and not done.get('witness') and len(done.get('rows', [])) == 6
            row.update(exit_code=code, complete=bool(passed), finished=time.time())
            completed.append(row)
            failed |= code != 0 or not passed
            del active[pid]
        while pending and len(active) < a.workers and not failed:
            index, reference = pending.pop(0)
            target = out/'cases'/reference['id']
            command = [sys.executable, '-u', str(HERE/'density_probe.py'), '--root', str(root), '--data', str(data),
                       '--output', str(target), '--readiness', a.readiness, '--index', str(index), '--phase', 'pilot', '--device', 'cpu']
            log = (out/(reference['id']+'.log')).open('x')
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=env)
            row = dict(index=index, id=reference['id'], pid=proc.pid, output=str(target), command=command, started=time.time())
            active[proc.pid] = (proc, log, row)
        live = []
        for proc, log, row in active.values():
            path = Path(row['output'])/'progress.json'
            try:
                latest = json.loads(path.read_text()) if path.exists() else None
            except json.JSONDecodeError:
                latest = None
            live.append(dict(row, progress=latest))
        dump(out/'status.json', dict(queue_pid=os.getpid(), frozen_commit=commit, running=live, completed=completed,
                                    pending=[v['id'] for _, v in pending], failed=bool(failed), unix_time=time.time()))
        if failed and not active:
            break
        if pending or active:
            time.sleep(5)
    passed = not failed and not pending and len(completed) == 8
    dump(out/'queue-finished.json', dict(passed=bool(passed), completed=completed, unstarted=[v['id'] for _, v in pending],
                                        frozen_commit=commit, density_data_sha256=digest))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
