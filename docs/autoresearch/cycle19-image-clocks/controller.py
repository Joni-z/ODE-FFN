"""One monitored pass, no automatic retries or generation restarts."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import json,subprocess,time,traceback
from pathlib import Path
import psutil
import study
root=Path('/Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle19');out=root/'orchestrator';out.mkdir()
assert study.load(root/'preflight.json')['status']=='PASS';study.verify(root,True)
phases=[('sampling',[study.TRAIN,str(root/'source/study.py'),'run','--root',str(root)],1300),
('audit-sampling',[study.TRAIN,str(root/'source/audit.py'),'--root',str(root),'--phase','sampling'],1200),
('metrics',[study.METRICS,str(root/'source/study.py'),'evaluate','--root',str(root)],600),
('audit-features',[study.METRICS,str(root/'source/audit.py'),'--root',str(root),'--phase','features'],600)]
records=[];p=None
try:
    for name,command,ceiling in phases:
        with (out/f'{name}.log').open('w') as log:p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        started=time.monotonic();record=dict(phase=name,command=command,pid=p.pid,create_time=psutil.Process(p.pid).create_time());records.append(record);study.write(out/'launches.json',records)
        while p.poll() is None:
            assert time.monotonic()-started<ceiling,(name,'registered ceiling')
            try:
                job=psutil.Process(p.pid);study.append(out/'resources.jsonl',dict(phase=name,pid=p.pid,seconds=time.monotonic()-started,cpu_seconds=sum(job.cpu_times()[:2]),rss=job.memory_info().rss))
            except psutil.NoSuchProcess:pass
            study.write(out/'status.json',dict(state='RUNNING',phase=name,pid=p.pid,completed=len(records)-1,total=4,seconds=time.monotonic()-started));time.sleep(5)
        record.update(exit_code=p.returncode,seconds=time.monotonic()-started);study.write(out/'launches.json',records);assert p.returncode==0,record
        print(json.dumps(record),flush=True)
    study.verify(root,True);study.write(out/'status.json',dict(state='COMPLETE',phases=4,no_training=True))
except BaseException:
    if p and p.poll() is None:
        try:
            children=psutil.Process(p.pid).children(recursive=True)
            for child in children:
                try:child.terminate()
                except psutil.NoSuchProcess:pass
        except psutil.NoSuchProcess:pass
        p.terminate()
    study.write(out/'status.json',dict(state='FAILED',phases=records));traceback.print_exc();raise
