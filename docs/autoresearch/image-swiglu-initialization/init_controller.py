"""One owned child at a time; fixed phase budgets and fail-closed execution."""
from __future__ import annotations
import json
import subprocess
import time
import psutil
import init_study as study


def gate(root):
    study.verify(root)
    passed = study.load(root/'documented-pass.json')
    assert passed['status'] == 'PASS' and passed['doc_vs_reality_mismatches'] == []
    assert len(passed['records']) == 2
    for name, sentinel, interpreter in (
        ('preflight', 'SWIGLU_INIT_NATIVE_PREFLIGHT_PASS', study.TRAIN),
        ('metric-preflight', 'SWIGLU_INIT_METRIC_PREFLIGHT_PASS', study.METRICS)):
        record = next(v for v in passed['records'] if v['action'] == name)
        command = f"{interpreter} -B {root/'source/init_study.py'} {name} --root {root}"
        assert record['documented_command'] == record['executed_command'] == command
        assert record['invocation_count'] == 1 and record['exit_code'] == 0
        assert record['terminal_tool_result']['exit_code'] == 0
        assert record['doc_vs_reality_mismatches'] == [] and record['expected_sentinel_observed']
        assert record['expected_sentinel'] == sentinel
        path = root/(name+'.json')
        assert study.load(path)['status'] == 'PASS'
        assert passed['preflight_hashes'][name] == study.sha(path)


def run(root):
    gate(root)
    out = root/'orchestrator'
    out.mkdir()
    me = psutil.Process()
    study.write(out/'identity.json', dict(pid=me.pid, create_time=me.create_time()))
    phases = [('train', study.TRAIN, 7200), ('audit-training', study.TRAIN, 1200),
              ('sample', study.TRAIN, 2700), ('audit-sampling', study.TRAIN, 2700),
              ('evaluate', study.METRICS, 1200), ('audit-metrics', study.METRICS, 1200)]
    records = []
    child = None
    try:
        for action, interpreter, ceiling in phases:
            command = [interpreter, '-B', str(root/'source/init_study.py'), action, '--root', str(root)]
            with (out/f'{action}.log').open('w') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            monitor = psutil.Process(child.pid)
            monitor.cpu_percent()
            tick = time.monotonic()
            record = dict(action=action, command=command, pid=child.pid,
                          create_time=monitor.create_time(), ceiling_seconds=ceiling)
            records.append(record)
            study.write(out/'launches.json', records)
            print(json.dumps(dict(state='STARTED', **record)), flush=True)
            while child.poll() is None:
                assert time.monotonic()-tick < ceiling, (action, 'wall-time ceiling')
                sample = dict(action=action, seconds=time.monotonic()-tick)
                try:
                    sample.update(pid=child.pid, CPU_percent=monitor.cpu_percent(),
                                  RSS=monitor.memory_info().rss)
                except psutil.NoSuchProcess:
                    pass
                progress = root/'output/runs'/study.ARM/'status.json'
                if progress.exists() and action == 'train':
                    sample['progress'] = study.load(progress)
                with (out/'resources.jsonl').open('a') as log:
                    log.write(json.dumps(sample)+'\n')
                study.write(out/'status.json', dict(state='RUNNING', completed=len(records)-1,
                            total=6, **sample))
                time.sleep(5)
            record.update(exit_code=child.wait(), seconds=time.monotonic()-tick)
            study.write(out/'launches.json', records)
            assert record['exit_code'] == 0, record
            print(json.dumps(dict(state='EXITED', **record)), flush=True)
        study.verify(root)
        study.write(out/'status.json', dict(state='COMPLETE', phases=6, new_trajectories=1,
                    quality_conclusion_pending=True))
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if child is not None and records:
            records[-1]['exit_code'] = child.poll()
            study.write(out/'launches.json', records)
        study.write(out/'status.json', dict(state='FAILED', error=repr(error), records=records))
        raise
