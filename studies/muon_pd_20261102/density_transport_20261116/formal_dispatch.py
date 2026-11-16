"""Monitored native-density jobs: one CPU thread or one idle permitted GPU."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('symmetry_dispatch_helpers', HERE.parent/'sharing_symmetry_20261116/run_symmetry_queue.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def dump(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    for name in ['root', 'data', 'output']:
        p.add_argument('--'+name, required=True)
    p.add_argument('--phase', choices=['witness', 'benchmark', 'formal'], required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    p.add_argument('--indices', default='0,1,2,3,4,5,6,7')
    p.add_argument('--readiness')
    p.add_argument('--derivative-backend', choices=['coordinate', 'batched'], default='coordinate')
    p.add_argument('--workers', type=int, choices=[1, 2, 3, 4], default=4)
    p.add_argument('--detach', action='store_true')
    a = p.parse_args()
    root, data, out = map(Path, [a.root, a.data, a.output])
    meta = json.loads((data/'density-data.json').read_text())
    assert meta['passed'] and len(meta['references']) == 8
    indices = [int(v) for v in a.indices.split(',')]
    assert indices and len(set(indices)) == len(indices) and all(0 <= i < 8 for i in indices)
    digest = hashlib.sha256((data/'density-data.json').read_bytes()).hexdigest()
    commit = subprocess.check_output(['git', '-C', str(HERE.parents[2]), 'rev-parse', 'HEAD'], text=True).strip()
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if a.device == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = '-1'
    if a.phase == 'witness':
        assert not a.detach and not out.exists()
        index, uuid = None, None
        if a.device == 'cuda':
            free, inventory = helpers.inventory()
            if not free:
                print(json.dumps(dict(started=False, reason='No idle physical GPU4-7')))
                raise SystemExit(75)
            index, uuid = free[0]
            env['CUDA_VISIBLE_DEVICES'] = uuid
        command = [sys.executable, '-u', str(HERE/'check_formal.py'), '--root', str(root), '--data', str(data),
                   '--output', str(out), '--device', a.device, '--derivative-backend', a.derivative_backend]
        if uuid:
            command += ['--gpu-uuid', uuid]
        with out.with_name(out.name+'-console.log').open('x') as log:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=env)
            row = dict(pid=proc.pid, command=command, physical_gpu=index, gpu_uuid=uuid, frozen_commit=commit, started=time.time())
            receipt = out.with_name(out.name+'-launch.json')
            dump(receipt, row)
            print(json.dumps(row), flush=True)
            row['exit_code'] = proc.wait()
            dump(receipt, row)
        raise SystemExit(row['exit_code'])
    assert a.readiness
    ready = json.loads(Path(a.readiness).read_text())
    assert ready['passed'] and ready['device'] == a.device and ready['model_cases'] == 8
    assert ready['derivative_backend'] == a.derivative_backend
    assert ready['density_data_sha256'] == digest and len(ready['formal_witnesses']) == 2
    for name in ['density_ops.py', 'formal_density.py', 'jacobian_backend.py']:
        assert ready['source_sha256'][name] == hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    if a.phase == 'benchmark':
        assert a.workers == 1, 'Time cases sequentially on each host'
    out.mkdir(parents=True, exist_ok=True)
    if a.detach:
        assert not (out/'queue-start.json').exists() and not (out/'queue.lock').exists()
        command = [sys.executable, '-u', str(Path(__file__).resolve())]+[v for v in sys.argv[1:] if v != '--detach']
        with (out/'queue-console.log').open('x') as log:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=env, start_new_session=True)
        row = dict(queue_pid=proc.pid, frozen_commit=commit, command=command, started=time.time(), phase=a.phase, device=a.device)
        dump(out/'queue-start.json', row)
        print(json.dumps(row))
        return
    with (out/'queue.lock').open('x') as lock:
        lock.write(str(os.getpid()))
    pending, active, completed, failed = indices.copy(), {}, [], False
    while pending or active:
        for pid, (proc, log, row) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close()
            target = Path(row['output'])/'COMPLETE.json'
            done = json.loads(target.read_text()) if target.exists() else {}
            complete = done.get('passed') and done.get('phase') == a.phase and done.get('parameters_unchanged')
            complete = complete and len(done.get('rows', [])) == (16 if a.phase == 'formal' else 2)
            row.update(exit_code=code, complete=bool(complete), finished=time.time())
            completed.append(row)
            failed |= code != 0 or not complete
            del active[pid]
        if a.device == 'cuda':
            try:
                slots, resource = helpers.inventory()
                error = None
            except (subprocess.SubprocessError, OSError) as exc:
                slots, resource, error = [], [], repr(exc)
        else:
            slots, resource, error = [(None, None)]*a.workers, [], None
        reserved = {row['gpu_uuid'] for proc, log, row in active.values()}
        for gpu_index, uuid in slots:
            if not pending or failed or len(active) >= a.workers:
                break
            if uuid and uuid in reserved:
                continue
            index = pending.pop(0)
            case = meta['references'][index]['id']
            target = out/'cases'/case
            command = [sys.executable, '-u', str(HERE/'formal_density.py'), '--root', str(root), '--data', str(data),
                       '--output', str(target), '--index', str(index), '--phase', a.phase, '--device', a.device,
                       '--readiness', a.readiness, '--derivative-backend', a.derivative_backend]
            if uuid:
                command += ['--gpu-uuid', uuid]
            log = (out/(case+'.log')).open('x')
            child_env = dict(env, CUDA_VISIBLE_DEVICES=uuid) if uuid else env
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=child_env)
            row = dict(index=index, id=case, pid=proc.pid, physical_gpu=gpu_index, gpu_uuid=uuid,
                       output=str(target), command=command, started=time.time())
            active[proc.pid] = (proc, log, row)
            if uuid:
                reserved.add(uuid)
        running = []
        for proc, log, row in active.values():
            file = Path(row['output'])/'progress.json'
            try:
                latest = json.loads(file.read_text()) if file.exists() else None
            except json.JSONDecodeError:
                latest = None
            running.append(dict(row, progress=latest))
        status = dict(queue_pid=os.getpid(), frozen_commit=commit, device=a.device, phase=a.phase, unix_time=time.time(),
                      completed=completed, running=running, pending=pending, failed=bool(failed), gpus4to7=resource, inventory_error=error)
        dump(out/'status.json', status)
        if failed and not active:
            break
        if pending or active:
            time.sleep(5)
    passed = not failed and not pending and len(completed) == len(indices)
    dump(out/'queue-finished.json', dict(passed=bool(passed), phase=a.phase, device=a.device,
        completed=completed, unstarted=pending, selected_indices=indices, frozen_commit=commit, density_data_sha256=digest))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
