"""Monitored three-arm queue and bounded audit/feature phases; no retries."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import subprocess,time,traceback
from pathlib import Path
import psutil
import study
root=Path('/Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle20');out=root/'orchestrator';out.mkdir()
assert study.load(root/'ready.json')['preflight']==study.sha(root/'preflight.json');study.verify(root)
records=[];live={};started=time.monotonic()
def execute_batch(specs):
    for name,command,ceiling in specs:
        with (out/f'{name}.log').open('w') as log:proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        record=dict(phase=name,command=command,pid=proc.pid,create_time=psutil.Process(proc.pid).create_time(),start=time.time());records.append(record)
        live[name]=(proc,time.monotonic(),ceiling,record);study.write(out/'launches.json',records)
    while live:
        resources=[]
        for name,(proc,t0,ceiling,record) in list(live.items()):
            code=proc.poll()
            if code is not None:
                record.update(exit_code=code,seconds=time.monotonic()-t0);del live[name];study.write(out/'launches.json',records);assert code==0,record
                print(__import__('json').dumps(record),flush=True)
            else:
                assert time.monotonic()-t0<ceiling,(name,'registered ceiling')
                try:
                    p=psutil.Process(proc.pid);resources.append(dict(phase=name,pid=p.pid,cpu_seconds=sum(p.cpu_times()[:2]),rss=p.memory_info().rss))
                except psutil.NoSuchProcess:pass
        study.append(out/'resources.jsonl',dict(elapsed=time.monotonic()-started,processes=resources))
        study.write(out/'status.json',dict(state='RUNNING',active=[dict(phase=k,pid=v[0].pid) for k,v in live.items()],completed=sum('exit_code' in r for r in records),total=4,seconds=time.monotonic()-started))
        if live:time.sleep(5)
try:
    execute_batch([('training',[study.TRAIN,str(root/'source/study.py'),'run','--root',str(root)],3700)])
    execute_batch([('audit-training',[study.TRAIN,str(root/'source/audit.py'),'--root',str(root),'--phase','training'],1800),
                   ('metrics',[study.METRICS,str(root/'source/study.py'),'evaluate','--root',str(root)],600)])
    execute_batch([('audit-features',[study.METRICS,str(root/'source/audit.py'),'--root',str(root),'--phase','features'],600)])
    study.verify(root);study.write(out/'status.json',dict(state='COMPLETE',phases=4,seconds=time.monotonic()-started,no_retries=True))
except BaseException:
    for proc,_,_,_ in live.values():
        if proc.poll() is None:
            try:
                for child in psutil.Process(proc.pid).children(recursive=True):
                    try:child.terminate()
                    except psutil.NoSuchProcess:pass
            except psutil.NoSuchProcess:pass
            proc.terminate()
    study.write(out/'status.json',dict(state='FAILED',phases=records));traceback.print_exc();raise
