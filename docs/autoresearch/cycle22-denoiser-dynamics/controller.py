"""One monitored numerical run followed by its independent finite differences."""
import json,subprocess,time,traceback
from pathlib import Path
import psutil
import study
root=Path('/Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle22');out=root/'orchestrator';out.mkdir()
study.verify(root);records=[];started=time.monotonic();proc=None
try:
    for phase in ('run','audit'):
        command=[study.CHECK,str(root/'source/study.py'),phase,'--root',str(root)]
        with (out/f'{phase}.log').open('w') as log:proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        record=dict(phase=phase,command=command,pid=proc.pid,create_time=psutil.Process(proc.pid).create_time());records.append(record);study.write(out/'launches.json',records);t0=time.monotonic()
        while True:
            code=proc.poll()
            if code is not None:
                record.update(exit_code=code,seconds=time.monotonic()-t0);study.write(out/'launches.json',records);assert code==0,record;break
            assert time.monotonic()-t0<300,(phase,'registered ceiling')
            try:
                p=psutil.Process(proc.pid);entry=dict(phase=phase,pid=p.pid,cpu_seconds=sum(p.cpu_times()[:2]),rss=p.memory_info().rss)
                with (out/'resources.jsonl').open('a') as f:f.write(json.dumps(dict(seconds=time.monotonic()-started,**entry))+'\n')
            except psutil.NoSuchProcess:pass
            study.write(out/'status.json',dict(state='RUNNING',phase=phase,pid=proc.pid,seconds=time.monotonic()-started));time.sleep(2)
    study.verify(root);study.write(out/'status.json',dict(state='COMPLETE',phases=2,seconds=time.monotonic()-started,no_retries=True))
except BaseException:
    if proc is not None and proc.poll() is None:proc.terminate()
    study.write(out/'status.json',dict(state='FAILED',records=records));traceback.print_exc();raise
