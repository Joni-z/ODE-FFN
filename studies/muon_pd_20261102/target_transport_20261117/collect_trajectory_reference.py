"""Collect independently completed GPU banks and run bounded CPU reference checks."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from population_resolution import read, dump, sha

REMOTE_DISCOVERY = r'''
from pathlib import Path
import json,hashlib,subprocess,sys
r=Path(sys.argv[1]);read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
launch=read(r/'launch.json');folder=r/'formal';progress=read(folder/'progress.json')
records=read(folder/'completed.json') if (folder/'completed.json').exists() else []
evaluated={x['key']:x for x in records if x['stage']=='evaluate' and x['actual_terminal_exit']==0}
verified={x['key']:x for x in records if x['stage']=='verify' and x['actual_terminal_exit']==0}
rows=[]
for key in sorted(evaluated.keys() & verified.keys()):
    original=evaluated[key];p=Path(original['output']);assert p.parent==folder and p.name==key
    done=read(p/'COMPLETE.json');ind=read(p/'independent.json')
    assert done['passed'] and ind['passed'] and ind['complete_sha256']==sha(p/'COMPLETE.json')
    rows.append(dict(key=key,case=done['case'],split=done['split'],bank=done['bank'],steps=done['steps'],
        output=str(p),complete_sha256=sha(p/'COMPLETE.json'),independent_sha256=sha(p/'independent.json'),
        array_file=done['file'],array_sha256=done['sha256']))
ps=subprocess.run(['ps','-p',str(launch['controller_pid']),'-o','args='],capture_output=True,text=True)
alive=ps.returncode==0 and 'dispatch_analytic_target.py' in ps.stdout and str(r/'repo') in ps.stdout
terminal=read(folder/'TERMINAL.json') if (folder/'TERMINAL.json').exists() else None
print(json.dumps(dict(rows=rows,controller_pid=launch['controller_pid'],controller_alive=alive,
    progress=progress,terminal=terminal,source_commit=launch['source_commit'])))
'''


def discover(remote_root):
    command=shlex.join(['/data2/zz5070/miniconda3/envs/py312/bin/python','-c',REMOTE_DISCOVERY,str(remote_root)])
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','145',command],
                          capture_output=True,text=True,timeout=45,check=True)
    return json.loads(result.stdout)


def main():
    p=argparse.ArgumentParser()
    for name in ('repo','root','local-main','output','execution-proof','python'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--remote-root',type=Path,default=Path('/data2/zz5070/ODE_FFN/analytic-target-20261117'))
    p.add_argument('--readiness',type=Path)
    p.add_argument('--workers',type=int,choices=range(1,5),default=4)
    p.add_argument('--inventory-only',action='store_true')
    a=p.parse_args()
    script=a.repo/'studies/muon_pd_20261102/target_transport_20261117'
    freeze=read(a.repo.parent/'source-freeze.json')
    assert all(sha(a.repo/f)==h for f,h in freeze['files'].items())
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    remote=discover(a.remote_root)
    if a.inventory_only:
        dump(a.output/'inventory.json',dict(passed=True,inventory_only=True,
            remote=remote,source_commit=freeze['commit'],source_sha256=sha(__file__)))
        print('REFERENCE_COLLECTOR_INVENTORY_PASS',len(remote['rows']),remote['controller_alive'],flush=True)
        return
    ready=read(a.readiness)
    assert ready['passed'] and ready['source_sha256']==sha(script/'trajectory_reference.py')
    assert ready['portable_sha256']==sha(a.root/'portable.json')
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    a.local_main.mkdir(parents=True,exist_ok=True)
    expected={f"{case['id']}-{split}-bank{bank}-heun{steps}"
        for case in read(a.root/'portable.json')['cases'] for split in ('selection','confirmation')
        for bank in range(8) for steps in (128,2048)}
    assert len(expected)==128
    cache={};active={};completed=[];started=set();failed=False;remote_error=None
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='-1')
    began=time.monotonic();next_discovery=0.
    def event(row):
        with (a.output/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(unix_time=time.time(),**row))+'\n')
    while True:
        if time.monotonic()>=next_discovery:
            try:
                remote=discover(a.remote_root);remote_error=None
                for row in remote['rows']:
                    key=row['key'];assert key in expected
                    if key in cache:
                        assert cache[key]==row
                        continue
                    target=a.local_main/key
                    if target.exists() and (target/'COMPLETE.json').exists():
                        assert sha(target/'COMPLETE.json')==row['complete_sha256']
                    target.mkdir(exist_ok=True)
                    subprocess.run(['rsync','-az','--ignore-existing','--include=COMPLETE.json',
                        '--include=independent.json','--include=bank.npz','--exclude=*',
                        '145:'+row['output']+'/',str(target)+'/'],check=True,timeout=180)
                    assert sha(target/'COMPLETE.json')==row['complete_sha256']
                    assert sha(target/'independent.json')==row['independent_sha256']
                    assert sha(target/row['array_file'])==row['array_sha256']
                    cache[key]=row;event(dict(event='cached',key=key,**{k:v for k,v in row.items() if k!='key'}))
                    dump(a.output/'cached.json',cache)
            except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
                remote_error=str(error);event(dict(event='read_only_transfer_wait',error=remote_error))
            except Exception as error:
                failed=True;remote_error=repr(error);event(dict(event='provenance_failure',error=remote_error))
            next_discovery=time.monotonic()+20
        for pid,(proc,log,row) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            log.close();row.update(actual_terminal_exit=code,finished_unix=time.time())
            completed.append(row);del active[pid];event(dict(event='terminal',**row));dump(a.output/'completed.json',completed)
            if code!=0:
                failed=True
                continue
            if row['stage']=='evaluate':
                assert read(Path(row['output'])/'COMPLETE.json')['passed']
                command=[str(a.python),'-u',str(script/'verify_trajectory_reference.py'),
                    '--input',row['output'],'--main',str(a.local_main/row['key']),
                    '--output',str(Path(row['output'])/'independent.json')]
                log=(a.output/(row['key']+'-independent.log')).open('x')
                child=subprocess.Popen(command,stdout=log,stderr=log,stdin=subprocess.DEVNULL,env=env)
                item=dict(key=row['key'],stage='verify',pid=child.pid,output=row['output'],
                          command=command,started_unix=time.time())
                active[child.pid]=child,log,item;event(dict(event='started',**item))
            else:
                assert read(Path(row['output'])/'independent.json')['passed']
        pending=[key for key in cache if key not in started]
        while pending and len(active)<a.workers and not failed:
            key=pending.pop(0);out=a.output/key
            command=[str(a.python),'-u',str(script/'trajectory_reference.py'),
                '--root',str(a.root),'--input',str(a.local_main/key),'--output',str(out),
                '--execution-proof',str(a.execution_proof),'--batch-size','4096']
            log=(a.output/(key+'-native.log')).open('x')
            child=subprocess.Popen(command,stdout=log,stderr=log,stdin=subprocess.DEVNULL,env=env)
            row=dict(key=key,stage='evaluate',pid=child.pid,output=str(out),command=command,started_unix=time.time())
            started.add(key);active[child.pid]=child,log,row;event(dict(event='started',**row))
        status=dict(pid=os.getpid(),unix_time=time.time(),seconds=time.monotonic()-began,
            cached=len(cache),expected=128,started=len(started),completed=len(completed),
            running=[row for _,_,row in active.values()],failed=failed,remote_error=remote_error,
            remote_controller_alive=remote['controller_alive'],remote_progress=remote['progress'])
        dump(a.output/'progress.json',status)
        with (a.output/'history.jsonl').open('a') as f:
            f.write(json.dumps({k:v for k,v in status.items() if k not in ('running','remote_progress')})+'\n')
        if failed and not active:break
        if len(completed)==256 and len(cache)==128 and not active and remote['terminal'] is not None:break
        if remote['terminal'] is not None and not remote['terminal']['passed'] and not active and not pending:
            failed=True;break
        time.sleep(5)
    assert all(sha(a.repo/f)==h for f,h in freeze['files'].items())
    passed=not failed and len(completed)==256 and set(cache)==expected and remote['terminal']['passed']
    dump(a.output/'TERMINAL.json',dict(passed=passed,completed=completed,cached=len(cache),
        source_commit=freeze['commit'],source_freeze_sha256=sha(a.repo.parent/'source-freeze.json'),
        source_sha256=sha(__file__),seconds=time.monotonic()-began))
    print('REFERENCE_COLLECTOR_TERMINAL',passed,flush=True)
    raise SystemExit(0 if passed else 1)


if __name__=='__main__':main()
