"""Bounded training/control cohort with monitored terminal phase exits."""
import argparse,subprocess,time
from pathlib import Path
import psutil
import training_controls as study

def main(root):
    study.verify(root);assert study.load(root/'documented-pass.json')['status']=='PASS'
    assert study.load(root/'preflight.json')['status']==study.load(root/'metric-preflight.json')['status']=='PASS'
    out=root/'orchestrator';out.mkdir();me=psutil.Process();study.write(out/'identity.json',dict(pid=me.pid,create_time=me.create_time()))
    source=root/'source'
    phases=[('training',[study.TRAIN,'-B',str(source/'training_controls.py'),'run','--root',str(root)],7500),
        ('audit-training',[study.TRAIN,'-B',str(source/'audit_training.py'),'--root',str(root),'--phase','training'],1200),
        ('sampling',[study.TRAIN,'-B',str(source/'training_controls.py'),'sample','--root',str(root)],3600),
        ('audit-sampling',[study.TRAIN,'-B',str(source/'audit_training.py'),'--root',str(root),'--phase','sampling'],3600),
        ('metrics',[study.METRICS,'-B',str(source/'training_controls.py'),'evaluate','--root',str(root)],1200),
        ('audit-features',[study.METRICS,'-B',str(source/'audit_training.py'),'--root',str(root),'--phase','features'],1200)]
    records=[];process=None
    try:
        for name,command,ceiling in phases:
            with (out/f'{name}.log').open('w') as f:process=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
            monitor=psutil.Process(process.pid);tracked={};tick=time.monotonic()
            record=dict(phase=name,command=command,pid=process.pid,create_time=monitor.create_time());records.append(record);study.write(out/'launches.json',records)
            while process.poll() is None:
                assert time.monotonic()-tick<ceiling;resources=[]
                try:children=[monitor,*monitor.children(recursive=True)]
                except psutil.NoSuchProcess:children=[]
                for child in children:
                    try:
                        key=(child.pid,child.create_time())
                        if key not in tracked:child.cpu_percent();tracked[key]=child;continue
                        child=tracked[key];resources.append(dict(pid=child.pid,create_time=key[1],CPU_percent=child.cpu_percent(),RSS=child.memory_info().rss))
                    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                study.append(out/'resources.jsonl',dict(phase=name,seconds=time.monotonic()-tick,resources=resources))
                study.write(out/'status.json',dict(state='RUNNING',phase=name,completed=len(records)-1,total=6,pid=process.pid,seconds=time.monotonic()-tick));time.sleep(5)
            record.update(exit_code=process.wait(),seconds=time.monotonic()-tick);study.write(out/'launches.json',records)
            assert record['exit_code']==0,record;print(study.json.dumps(record),flush=True)
        study.verify(root);study.write(out/'status.json',dict(state='COMPLETE',phases=6,new_training_trajectories=2))
    except BaseException as e:
        if process and process.poll() is None:
            try:
                for child in psutil.Process(process.pid).children(recursive=True):
                    try:child.terminate()
                    except psutil.NoSuchProcess:pass
            except psutil.NoSuchProcess:pass
            process.terminate();process.wait(timeout=30)
        study.write(out/'status.json',dict(state='FAILED',error=repr(e),records=records));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
