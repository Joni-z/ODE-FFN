"""One monitored sequence; no automatic retries after failure or timeout."""
import argparse,subprocess,time
from pathlib import Path
import psutil
import study

def main(root):
    study.verify(root,True)
    assert study.load(root/'preflight.json')['status']==study.load(root/'documented-pass.json')['status']=='PASS'
    out=root/'orchestrator';out.mkdir(exist_ok=False);me=psutil.Process()
    study.write(out/'identity.json',dict(pid=me.pid,create_time=me.create_time()))
    phases=[('sampling',[study.TRAIN,'-B',str(root/'source/study.py'),'run','--root',str(root)],1300),
            ('audit-sampling',[study.TRAIN,'-B',str(root/'source/audit.py'),'--root',str(root),'--phase','sampling'],1200),
            ('metrics',[study.METRICS,'-B',str(root/'source/study.py'),'evaluate','--root',str(root)],600),
            ('audit-features',[study.METRICS,'-B',str(root/'source/audit.py'),'--root',str(root),'--phase','features'],600)]
    records=[];process=None
    try:
        for name,command,ceiling in phases:
            with (out/f'{name}.log').open('w') as log:process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            monitor=psutil.Process(process.pid);monitor.cpu_percent();tick=time.monotonic()
            record=dict(phase=name,command=command,pid=process.pid,create_time=monitor.create_time());records.append(record)
            study.write(out/'launches.json',records)
            while process.poll() is None:
                assert time.monotonic()-tick<ceiling,(name,'registered ceiling')
                try:study.append(out/'resources.jsonl',dict(phase=name,pid=process.pid,seconds=time.monotonic()-tick,RSS=monitor.memory_info().rss,CPU_percent=monitor.cpu_percent()))
                except psutil.NoSuchProcess:pass
                study.write(out/'status.json',dict(state='RUNNING',phase=name,pid=process.pid,completed=len(records)-1,total=4,seconds=time.monotonic()-tick))
                time.sleep(5)
            record.update(exit_code=process.returncode,seconds=time.monotonic()-tick);study.write(out/'launches.json',records)
            assert process.returncode==0,record;print(record,flush=True)
        study.verify(root,True);study.write(out/'status.json',dict(state='COMPLETE',phases=4,new_training=False))
    except BaseException as error:
        if process and process.poll() is None:
            try:
                for child in psutil.Process(process.pid).children(recursive=True):
                    try:child.terminate()
                    except psutil.NoSuchProcess:pass
            except psutil.NoSuchProcess:pass
            process.terminate();process.wait(timeout=30)
        study.write(out/'status.json',dict(state='FAILED',error=repr(error),phases=records));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
