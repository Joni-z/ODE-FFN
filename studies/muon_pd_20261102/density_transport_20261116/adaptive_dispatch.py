"""Use the existing resource selector for the bounded full-source refinement."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from formal_dispatch import HERE, dump, helpers


def main():
    p = argparse.ArgumentParser()
    for name in ['root', 'data', 'output']:
        p.add_argument('--'+name, required=True)
    p.add_argument('--runtimes', nargs='+', required=True)
    p.add_argument('--indices', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    p.add_argument('--derivative-backend', choices=['coordinate', 'batched'], required=True)
    p.add_argument('--workers', type=int, choices=[1, 2, 3, 4], default=2)
    p.add_argument('--readiness')
    p.add_argument('--witness', action='store_true')
    p.add_argument('--detach', action='store_true')
    a = p.parse_args()
    data, out = Path(a.data), Path(a.output)
    refs = json.loads((data/'density-data.json').read_text())['references']
    indices = [int(v) for v in a.indices.split(',')]
    assert indices and len(set(indices)) == len(indices) and all(0 <= i < 8 for i in indices)
    parents = {}
    for name in a.runtimes:
        runtime = Path(name)
        start = json.loads((runtime/'queue-start.json').read_text())
        state_file = runtime/('queue-finished.json' if (runtime/'queue-finished.json').exists() else 'status.json')
        state = json.loads(state_file.read_text())
        for row in state['completed']:
            if row['index'] not in indices:
                continue
            assert row['exit_code'] == 0 and row['complete'] and row['index'] not in parents
            parents[row['index']] = (runtime/'cases'/row['id'], start['frozen_commit'])
    assert set(parents) == set(indices), 'Require successful parent worker completion, not just a file'
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if a.device == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = '-1'
    commit = subprocess.check_output(['git', '-C', str(HERE.parents[2]), 'rev-parse', 'HEAD'], text=True).strip()
    def command(index, target, uuid=None):
        parent, parent_commit = parents[index]
        cmd = [sys.executable, '-u', str(HERE/'adaptive_density.py'), '--root', a.root, '--data', a.data,
            '--case-output', str(parent), '--parent-commit', parent_commit, '--output', str(target),
            '--device', a.device, '--derivative-backend', a.derivative_backend]
        if uuid:
            cmd += ['--gpu-uuid', uuid]
        cmd += ['--witness'] if a.witness else ['--readiness', a.readiness]
        return cmd
    if a.witness:
        assert len(indices) == 1 and not a.detach and not out.exists()
        physical, uuid = None, None
        if a.device == 'cuda':
            slots, _ = helpers.inventory()
            if not slots:
                print(json.dumps(dict(started=False, reason='No idle physical GPU4-7')))
                raise SystemExit(75)
            physical, uuid = slots[0]
            env['CUDA_VISIBLE_DEVICES'] = uuid
        cmd = command(indices[0], out, uuid)
        with out.with_name(out.name+'-console.log').open('x') as log:
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=env)
            row = dict(pid=proc.pid, command=cmd, physical_gpu=physical, gpu_uuid=uuid,
                       frozen_commit=commit, started=time.time())
            receipt = out.with_name(out.name+'-launch.json')
            dump(receipt, row)
            print(json.dumps(row), flush=True)
            row['exit_code'] = proc.wait()
            dump(receipt, row)
        raise SystemExit(row['exit_code'])
    assert a.readiness
    out.mkdir(parents=True, exist_ok=True)
    if a.detach:
        assert not (out/'queue-start.json').exists() and not (out/'queue.lock').exists()
        cmd = [sys.executable, '-u', str(Path(__file__).resolve())]+[v for v in sys.argv[1:] if v != '--detach']
        with (out/'queue-console.log').open('x') as log:
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                    env=env, start_new_session=True)
        row = dict(queue_pid=proc.pid, frozen_commit=commit, command=cmd, started=time.time(),
                   device=a.device, indices=indices, adaptive=True)
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
            file = Path(row['output'])/'COMPLETE.json'
            done = json.loads(file.read_text()) if file.exists() else {}
            valid = done.get('passed') and done.get('adaptive') and not done.get('witness', True)
            valid = valid and len(done.get('cells', [])) == 2 and all(c['samples'] == 1024 for c in done['cells'])
            row.update(exit_code=code, complete=bool(valid), finished=time.time(),
                numerical_resolved=bool(valid and all(c['numerical_qualified'] for c in done['cells'])))
            completed.append(row)
            failed |= code != 0 or not valid
            del active[pid]
        if a.device == 'cuda':
            try:
                slots, inventory = helpers.inventory()
                error = None
            except (subprocess.SubprocessError, OSError) as exc:
                slots, inventory, error = [], [], repr(exc)
        else:
            slots, inventory, error = [(None, None)]*a.workers, [], None
        reserved = {row['gpu_uuid'] for proc, log, row in active.values()}
        for physical, uuid in slots:
            if not pending or failed or len(active) >= a.workers:
                break
            if uuid and uuid in reserved:
                continue
            index = pending.pop(0)
            target = out/'cases'/refs[index]['id']
            cmd = command(index, target, uuid)
            log = (out/(refs[index]['id']+'.log')).open('x')
            child_env = dict(env, CUDA_VISIBLE_DEVICES=uuid) if uuid else env
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=child_env)
            row = dict(index=index, id=refs[index]['id'], pid=proc.pid, physical_gpu=physical,
                gpu_uuid=uuid, output=str(target), command=cmd, started=time.time())
            active[proc.pid] = (proc, log, row)
            if uuid:
                reserved.add(uuid)
        running = []
        for proc, log, row in active.values():
            file = Path(row['output'])/'progress.json'
            running.append(dict(row, progress=json.loads(file.read_text()) if file.exists() else None))
        dump(out/'status.json', dict(queue_pid=os.getpid(), frozen_commit=commit, device=a.device,
            unix_time=time.time(), completed=completed, running=running, pending=pending,
            failed=bool(failed), gpus4to7=inventory, inventory_error=error))
        if failed and not active:
            break
        if pending or active:
            time.sleep(5)
    passed = not failed and not pending and len(completed) == len(indices)
    dump(out/'queue-finished.json', dict(passed=bool(passed), adaptive=True, completed=completed,
        unstarted=pending, selected_indices=indices, frozen_commit=commit,
        numerical_resolved=all(r['numerical_resolved'] for r in completed)))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
