"""Monitored bank queue, using only newly empty physical145 GPUs4-7."""
import argparse
import csv
import io
import os
from pathlib import Path
import subprocess
import sys
import time
from population_resolution import read, dump, sha


def main():
    p = argparse.ArgumentParser()
    for name in ('repo', 'root', 'reference', 'output', 'cache', 'execution-proof', 'readiness', 'python'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    ready = read(a.readiness)
    script = a.repo/'studies/muon_pd_20261102/target_transport_20261117'
    assert ready['passed'] and ready['source_sha256'] == sha(script/'evaluate_analytic_target.py')
    assert ready['portable_sha256'] == sha(a.root/'portable.json')
    freeze = read(a.repo.parent/'source-freeze.json')
    assert all(sha(a.repo/f) == h for f,h in freeze['files'].items())
    sys.path.insert(0, str(script.parent/'coefficient_20261116'))
    from sharing_dispatch import eligible
    a.output.mkdir(parents=True, exist_ok=False)
    primary = 'product_rk4_trained'
    others = [r['id'] for r in read(a.root/'portable.json')['cases'] if r['id'] != primary]
    pending = [dict(case=case, split=split, bank=bank, steps=steps)
               for cases in ([primary], others) for steps in (128, 2048)
               for case in cases for split in ('selection', 'confirmation') for bank in range(8)]
    assert len(pending) == 128
    active, completed, failed = {}, [], False
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    query = lambda fields: subprocess.check_output(['nvidia-smi', '--query-'+fields,
        '--format=csv,noheader,nounits'], text=True)
    began = time.monotonic()
    def event(row):
        import json
        with (a.output/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(unix_time=time.time(), **row))+'\n')
    while pending or active:
        inventory = query('gpu=index,uuid,memory.used,utilization.gpu,power.draw')
        apps = list(csv.reader(io.StringIO(query('compute-apps=gpu_uuid,pid'))))
        occupied = {r[0].strip() for r in apps if len(r)>1 and r[1].strip().isdigit()}
        for pid, (proc, log, row) in list(active.items()):
            if proc.poll() is None and row['stage'] == 'evaluate':
                foreign = [int(r[1]) for r in apps if len(r)>1 and r[0].strip() == row['gpu_uuid']
                           and r[1].strip().isdigit() and int(r[1]) != pid]
                if foreign:
                    row['foreign_context_detected'] = foreign
                    row['termination_scope'] = 'our worker only'
                    proc.terminate()
            code = proc.poll()
            if code is None:
                continue
            log.close()
            row.update(actual_terminal_exit=code, finished_unix=time.time())
            completed.append(row)
            event(dict(event='terminal', **row))
            del active[pid]
            dump(a.output/'completed.json', completed)
            if code != 0:
                failed = True
                continue
            if row['stage'] == 'evaluate':
                assert read(Path(row['output'])/'COMPLETE.json')['passed']
                command = [str(a.python), '-u', str(script/'verify_analytic_target.py'),
                           '--input', row['output'], '--output', str(Path(row['output'])/'independent.json')]
                verify_log = (a.output/(row['key']+'-independent.log')).open('x')
                child = subprocess.Popen(command, stdout=verify_log, stderr=verify_log,
                                         stdin=subprocess.DEVNULL, env=env)
                verification = dict(key=row['key'], stage='verify', pid=child.pid,
                    output=row['output'], command=command, started_unix=time.time())
                active[child.pid] = child, verify_log, verification
                event(dict(event='started', **verification))
            else:
                assert read(Path(row['output'])/'independent.json')['passed']
        used = {r['gpu_uuid'] for _,_,r in active.values() if r['stage'] == 'evaluate'}
        idle = eligible(list(csv.reader(io.StringIO(inventory))), occupied | used)
        while pending and idle and not failed:
            physical, uuid = idle.pop(0)
            task = pending.pop(0)
            key = f"{task['case']}-{task['split']}-bank{task['bank']}-heun{task['steps']}"
            out = a.output/key
            command = [str(a.python), '-u', str(script/'evaluate_analytic_target.py'),
                '--root', str(a.root), '--reference', str(a.reference), '--output', str(out),
                '--cache', str(a.cache), '--execution-proof', str(a.execution_proof),
                '--readiness', str(a.readiness), '--gpu-uuid', uuid]
            for name, value in task.items():
                command += ['--'+name, str(value)]
            log = (a.output/(key+'-native.log')).open('x')
            child = subprocess.Popen(command, stdout=log, stderr=log, stdin=subprocess.DEVNULL, env=env)
            row = dict(key=key, **task, stage='evaluate', pid=child.pid, output=str(out),
                       command=command, physical_gpu=physical, gpu_uuid=uuid, started_unix=time.time())
            active[child.pid] = child, log, row
            event(dict(event='started', **row))
        status = dict(pid=os.getpid(), unix_time=time.time(), seconds=time.monotonic()-began,
            running=[r for _,_,r in active.values()], completed=len(completed),
            pending=len(pending), failed=failed, gpu_inventory=inventory)
        dump(a.output/'progress.json', status)
        import json
        with (a.output/'history.jsonl').open('a') as f:
            f.write(json.dumps(dict(unix_time=status['unix_time'], completed=len(completed),
                pending=len(pending), running=[dict(pid=pid,key=row['key'],stage=row['stage'])
                    for pid,(_,_,row) in active.items()], gpu_inventory=inventory))+'\n')
        if failed and not active:
            break
        if pending or active:
            time.sleep(5)
    assert all(sha(a.repo/f) == h for f,h in freeze['files'].items())
    result = dict(passed=not failed and not pending and len(completed)==256,
        completed=completed, unstarted=pending, source_commit=freeze['commit'],
        source_freeze_sha256=sha(a.repo.parent/'source-freeze.json'),
        readiness_sha256=sha(a.readiness), source_sha256=sha(__file__))
    dump(a.output/'TERMINAL.json', result)
    print('ANALYTIC_TARGET_QUEUE_TERMINAL', result['passed'], flush=True)
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
