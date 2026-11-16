"""Monitor four fixed inference tasks; allocate only empty physical145GPUs4-7."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)


def main():
    p=argparse.ArgumentParser()
    for name in ('repo','root','output','readiness','python'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();ready=read(a.readiness)
    assert ready['passed'] and ready['portable_sha256']==sha(a.root/'portable.json')
    freeze=read(a.repo.parent/'source-freeze.json')
    assert all(sha(a.repo/f)==h for f,h in freeze['files'].items())
    script=a.repo/'studies/muon_pd_20261102/target_transport_20261117'
    sys.path.insert(0,str(script.parent/'coefficient_20261116'))
    from sharing_dispatch import eligible
    a.output.mkdir(parents=True,exist_ok=False)
    pending=[r['id'] for r in read(a.root/'portable.json')['cases']]
    active={};done=[];failed=False
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    query=lambda fields:subprocess.check_output(['nvidia-smi','--query-'+fields,'--format=csv,noheader,nounits'],text=True)
    while pending or active:
        gpu_csv=query('gpu=index,uuid,memory.used,utilization.gpu,power.draw')
        apps=list(csv.reader(io.StringIO(query('compute-apps=gpu_uuid,pid'))))
        occupied={r[0].strip() for r in apps if len(r)>1 and r[1].strip().isdigit()}
        for pid,(proc,log,row) in list(active.items()):
            if proc.poll() is None and row['stage']=='evaluate':
                foreign=[int(r[1]) for r in apps if len(r)>1 and r[0].strip()==row['gpu_uuid'] and
                         r[1].strip().isdigit() and int(r[1])!=pid]
                if foreign:
                    row['foreign_context_detected']=foreign;proc.terminate()
                    row['termination_scope']='our child only; never touch foreign process'
            code=proc.poll()
            if code is None:continue
            log.close();row['actual_terminal_exit']=code;row['finished_unix']=time.time()
            done.append(row);del active[pid]
            if code!=0:failed=True;continue
            if row['stage']=='evaluate':
                assert read(Path(row['output'])/'COMPLETE.json')['passed']
                command=[str(a.python),'-u',str(script/'verify_population_resolution.py'),
                    '--root',str(a.root),'--input',row['output'],
                    '--output',str(Path(row['output'])/'independent.json')]
                log=(a.output/(row['case']+'-independent.log')).open('x')
                child=subprocess.Popen(command,stdout=log,stderr=log,stdin=subprocess.DEVNULL,env=env)
                active[child.pid]=(child,log,dict(case=row['case'],stage='verify',pid=child.pid,
                    output=row['output'],command=command,started_unix=time.time()))
            else:assert read(Path(row['output'])/'independent.json')['passed']
        used={r['gpu_uuid'] for _,_,r in active.values() if r['stage']=='evaluate'}
        idle=eligible(list(csv.reader(io.StringIO(gpu_csv))),occupied|used)
        while pending and idle and not failed:
            physical,uuid=idle.pop(0);case=pending.pop(0);out=a.output/case
            command=[str(a.python),'-u',str(script/'population_resolution.py'),'evaluate',
                '--root',str(a.root),'--output',str(out),'--case',case,
                '--readiness',str(a.readiness),'--gpu-uuid',uuid]
            log=(a.output/(case+'-native.log')).open('x')
            child=subprocess.Popen(command,stdout=log,stderr=log,stdin=subprocess.DEVNULL,env=env)
            active[child.pid]=(child,log,dict(case=case,stage='evaluate',pid=child.pid,
                output=str(out),command=command,physical_gpu=physical,gpu_uuid=uuid,started_unix=time.time()))
        status=dict(pid=os.getpid(),unix_time=time.time(),running=[r for _,_,r in active.values()],
            completed=done,pending=pending,failed=failed,gpu_inventory=gpu_csv)
        dump(a.output/'progress.json',status)
        with (a.output/'history.jsonl').open('a') as f:f.write(json.dumps(status)+'\n')
        if failed and not active:break
        if pending or active:time.sleep(5)
    result=dict(passed=not failed and not pending and len(done)==8,cases=done,unstarted=pending,
        source_commit=freeze['commit'],source_freeze_sha256=sha(a.repo.parent/'source-freeze.json'),
        dispatcher_sha256=sha(__file__),readiness_sha256=sha(a.readiness),portable_sha256=sha(a.root/'portable.json'))
    dump(a.output/'TERMINAL.json',result)
    print('POPULATION_QUEUE_TERMINAL',result['passed'],flush=True)
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':main()
