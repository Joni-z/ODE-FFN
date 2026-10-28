"""Wait for the known evaluator, then verify the complete fixed cohort once."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
import psutil

TRAIN='/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python'
METRICS='/Users/mr.z/ode-workspace-prep/.venv-image-metrics-v2/bin/python'
load=lambda p:json.loads(Path(p).read_text())
def write(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def alive(identity):
    try:
        p=psutil.Process(identity['pid']);assert abs(p.create_time()-identity['create_time'])<1e-3
        return p.is_running() and p.status()!=psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:return False
def main(root):
    out=root/'finalization';out.mkdir(exist_ok=False);start=time.monotonic()
    write(out/'identity.json',dict(pid=psutil.Process().pid,create_time=psutil.Process().create_time()))
    target=load(root/'metric-controller/identity.json');active=[]
    try:
        while alive(target):
            assert time.monotonic()-start<28800
            write(out/'status.json',dict(phase='WAITING_FOR_EVALUATION',pid=target['pid']))
            time.sleep(5)
        result=load(root/'metric-output/result.json');assert result['status']=='PASS' and len(result['rows'])==14
        registration=load(root/'audit-source/registration.json')
        for name,digest in registration['files'].items():assert hashlib.sha256((root/'audit-source'/name).read_bytes()).hexdigest()==digest
        for phase,python in (('training',TRAIN),('features',METRICS)):
            command=[python,'-B',str(root/'audit-source/audit.py'),'--root',str(root),'--phase',phase]
            with (out/f'{phase}.log').open('w') as log:process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            active.append((phase,process));write(out/'launches.json',[dict(phase=k,pid=p.pid) for k,p in active])
        while any(p.poll() is None for _,p in active):
            assert time.monotonic()-start<36000
            for phase,p in active:
                if p.poll() not in (None,0):raise RuntimeError(f'{phase} audit failed; inspect retained log')
            write(out/'status.json',dict(phase='AUDITING',active=[k for k,p in active if p.poll() is None]))
            time.sleep(5)
        for phase,p in active:
            assert p.returncode==0 and load(root/f'audit-{phase}/result.json')['status']=='PASS'
        write(out/'status.json',dict(phase='AUDITS_COMPLETE',exits=[dict(phase=k,pid=p.pid,exit_code=p.returncode) for k,p in active]))
    except BaseException as error:
        for _,p in active:
            if p.poll() is None:p.terminate()
        for _,p in active:p.wait(timeout=30)
        write(out/'status.json',dict(phase='FAILED',error=repr(error)))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
