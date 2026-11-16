"""Two monitored precision-probe CPU workers; preserve failures, never retry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, data: dict) -> None:
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def main() -> None:
    p = argparse.ArgumentParser()
    for key in ['study', 'pilot', 'data-root', 'output', 'readiness']:
        p.add_argument('--'+key, required=True)
    p.add_argument('--refined-native', help='Verified native-qualification directory for sensitivity refinement')
    p.add_argument('--detach', action='store_true')
    a = p.parse_args()
    out, study = Path(a.output), Path(a.study)
    ready = read(Path(a.readiness))
    assert ready.get('mode', 'native') == ('refined' if a.refined_native else 'native')
    assert ready['passed'] and ready['native_kernel_cases'] == 12 and ready['native_cli_witnesses'] == 2
    for file, digest in ready['bound_artifacts'].items():
        assert sha(Path(file)) == digest
    for name, digest in ready['runtime_source_sha256'].items():
        assert sha(HERE/name) == digest
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='-1')
    out.mkdir(parents=True, exist_ok=True)
    if a.detach:
        assert not (out/'queue-start.json').exists() and not (out/'queue.lock').exists()
        command = [sys.executable, '-u', str(Path(__file__).resolve())]+[v for v in sys.argv[1:] if v != '--detach']
        with (out/'queue-console.log').open('x') as log:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=env, start_new_session=True)
        dump(out/'queue-start.json', dict(pid=proc.pid, command=command, source_freeze=ready['source_freeze'], started=time.time()))
        print(json.dumps(dict(queue_pid=proc.pid, output=str(out))), flush=True)
        return
    with (out/'queue.lock').open('x') as f:
        f.write(str(os.getpid()))
    anchors = read(study/'anchors.json')
    pending = [r['id'] for r in anchors['references']+anchors['new_cases']]
    assert len(pending) == len(set(pending)) == 12
    active, completed, failed = {}, [], False
    while pending or active:
        for pid, (proc, log, row) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close()
            file = Path(row['output'])/'COMPLETE.json'
            meta = read(file) if file.exists() else {}
            valid = meta.get('passed') and not meta.get('witness') and meta.get('sources') == 256 and meta.get('id') == row['id']
            row.update(exit_code=code, complete=bool(valid), finished=time.time())
            completed.append(row)
            failed |= code != 0 or not valid
            del active[pid]
        while pending and len(active) < 2 and not failed:
            case = pending.pop(0)
            target = out/'cases'/case
            probe = 'refined_trajectory_probe.py' if a.refined_native else 'native_time_probe.py'
            command = [sys.executable, '-u', str(HERE/probe), '--study', a.study,
                '--pilot', a.pilot, '--data-root', a.data_root, '--case', case, '--output', str(target)]
            if a.refined_native:
                command += ['--native', a.refined_native]
            log = (out/(case+'.log')).open('x')
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=env)
            active[proc.pid] = (proc, log, dict(id=case, pid=proc.pid, output=str(target), command=command, started=time.time()))
        live = []
        for proc, log, row in active.values():
            file = Path(row['output'])/'progress.json'
            try:
                progress = read(file) if file.exists() else None
            except json.JSONDecodeError:
                progress = None
            live.append(dict(row, progress=progress))
        dump(out/'status.json', dict(queue_pid=os.getpid(), running=live, completed=completed,
            pending=pending, failed=bool(failed), unix_time=time.time(), cpu_load=list(os.getloadavg())))
        if failed and not active:
            break
        if pending or active:
            time.sleep(5)
    passed = not failed and not pending and len(completed) == 12
    dump(out/'queue-finished.json', dict(passed=bool(passed), completed=completed, unstarted=pending,
        source_freeze=ready['source_freeze'], readiness_sha256=sha(Path(a.readiness))))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
