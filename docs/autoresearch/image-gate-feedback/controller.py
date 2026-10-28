"""Monitor the fixed analysis and audit; retain exact child exits."""
import argparse,subprocess,time
from pathlib import Path
import psutil
import study

def main(root):
    out=root/'controller';out.mkdir();tick=time.monotonic();launches=[]
    phases=[('analysis',[study.TRAIN,'-B',str(root/'source/study.py'),'run','--root',str(root)],4100),
        ('audit',[study.TRAIN,'-B',str(root/'source/audit.py'),'--root',str(root)],2100)]
    for name,command,ceiling in phases:
        start=time.monotonic()
        with (out/f'{name}.log').open('w') as log:p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        tracked={};record=dict(phase=name,pid=p.pid,create_time=psutil.Process(p.pid).create_time(),command=command)
        launches.append(record);study.write(out/'launches.json',launches)
        try:
            while p.poll() is None:
                assert time.monotonic()-start<ceiling
                try:processes=[psutil.Process(p.pid),*psutil.Process(p.pid).children(recursive=True)]
                except psutil.NoSuchProcess:processes=[]
                resources=[]
                for child in processes:
                    try:
                        key=(child.pid,child.create_time())
                        if key not in tracked:child.cpu_percent();tracked[key]=child;continue
                        child=tracked[key];resources.append(dict(pid=child.pid,create_time=key[1],CPU_percent=child.cpu_percent(),RSS=child.memory_info().rss))
                    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                with (out/'resources.jsonl').open('a') as f:f.write(study.json.dumps(dict(phase=name,seconds=time.monotonic()-start,resources=resources))+'\n')
                study.write(out/'status.json',dict(state='RUNNING',phase=name,seconds=time.monotonic()-tick))
                time.sleep(5)
            record.update(exit_code=p.wait(),seconds=time.monotonic()-start);study.write(out/'launches.json',launches)
            assert record['exit_code']==0,record
        except BaseException:
            if p.poll() is None:p.terminate();p.wait(timeout=30)
            study.write(out/'status.json',dict(state='FAILED',phase=name,exit_code=p.returncode));raise
    study.write(out/'status.json',dict(state='COMPLETE',seconds=time.monotonic()-tick,phases=2))
    print(study.json.dumps(study.load(out/'status.json')),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
