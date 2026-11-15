"""Monitored symmetry inference; idle145physicalGPUs4-7 only, no retries."""
import argparse,csv,io,json,os,subprocess,sys,time,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'coefficient_20261116'))
from sharing_dispatch import eligible

def dump(path,obj):
    path=Path(path);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(path)

def inventory():
    query=lambda text:subprocess.check_output(['nvidia-smi','--query-'+text,'--format=csv,noheader,nounits'],text=True,timeout=15)
    rows=list(csv.reader(io.StringIO(query('gpu=index,uuid,memory.used'))))
    occupied={r[0].strip() for r in csv.reader(io.StringIO(query('compute-apps=gpu_uuid,pid'))) if len(r)>1 and r[1].strip().isdigit()}
    return eligible(rows,occupied),[r for r in rows if int(r[0]) in [4,5,6,7]]

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--readiness');p.add_argument('--witness',action='store_true')
    p.add_argument('--workers',type=int,choices=[1,2,3,4],default=4);p.add_argument('--detach',action='store_true');a=p.parse_args()
    root=Path(a.root);out=Path(a.output);manifest=json.loads((root/'portable.json').read_text());assert manifest['passed'] and len(manifest['cases'])==8
    commit=subprocess.check_output(['git','-C',str(HERE.parents[2]),'rev-parse','HEAD'],text=True).strip()
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    if a.witness:
        assert not a.detach and not out.exists();free,_=inventory()
        if not free:print(json.dumps(dict(started=False,reason='No idle physicalGPU4-7')));raise SystemExit(75)
        index,uuid=free[0];env['CUDA_VISIBLE_DEVICES']=uuid
        command=[sys.executable,'-u',str(HERE/'check.py'),'--root',str(root),'--output',str(out),'--device','cuda','--gpu-uuid',uuid]
        console=out.with_name(out.name+'-console.log');receipt=out.with_name(out.name+'-launch.json');assert not receipt.exists()
        with console.open('x') as log:
            proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,env=env)
            record=dict(pid=proc.pid,physical_gpu=index,gpu_uuid=uuid,frozen_commit=commit,command=command,unix_time=time.time())
            dump(receipt,record);print(json.dumps(record),flush=True);code=proc.wait();record['exit_code']=code;dump(receipt,record)
        raise SystemExit(code)
    assert a.readiness;ready=json.loads(Path(a.readiness).read_text());assert ready['passed'] and ready['cases']==8 and ready['device']=='cuda' and len(ready['entry_witnesses'])==2
    digest=hashlib.sha256((root/'portable.json').read_bytes()).hexdigest();assert ready['portable_manifest_sha256']==digest
    out.mkdir(parents=True,exist_ok=True)
    if a.detach:
        assert not (out/'queue.lock').exists() and not (out/'queue-start.json').exists()
        command=[sys.executable,'-u',str(Path(__file__).resolve())]+[v for v in sys.argv[1:] if v!='--detach']
        with (out/'queue-console.log').open('x') as log:
            proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,env=env,start_new_session=True)
        row=dict(queue_pid=proc.pid,command=command,frozen_commit=commit,unix_time=time.time());dump(out/'queue-start.json',row);print(json.dumps(row));return
    with (out/'queue.lock').open('x') as lock:lock.write(str(os.getpid()))
    pending=[c['id'] for c in manifest['cases']];active={};done=[];failed=False
    while pending or active:
        for pid,(proc,log,row) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            log.close();file=Path(row['output'])/'COMPLETE.json';complete=False
            if file.exists():
                check=json.loads(file.read_text());complete=check['passed'] and not check['witness'] and len(check['sample_rows'])==8
            row.update(exit_code=code,complete=complete,finished=time.time());done.append(row);del active[pid];failed|=code!=0 or not complete
        try:free,resource=inventory();error=None
        except (subprocess.SubprocessError,OSError) as e:free=[];resource=[];error=repr(e)
        reserved={r['gpu_uuid'] for _,_,r in active.values()}
        for index,uuid in free:
            if not pending or len(active)>=a.workers or failed:break
            if uuid in reserved:continue
            case=pending.pop(0);target=out/'cases'/case
            command=[sys.executable,'-u',str(HERE/'evaluate.py'),'--root',str(root),'--case',case,'--output',str(target),'--readiness',a.readiness,'--device','cuda','--gpu-uuid',uuid]
            log=(out/(case+'.log')).open('x');proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=log,env=dict(env,CUDA_VISIBLE_DEVICES=uuid))
            row=dict(id=case,pid=proc.pid,physical_gpu=index,gpu_uuid=uuid,output=str(target),command=command,started=time.time())
            active[proc.pid]=(proc,log,row);reserved.add(uuid)
        running=[]
        for proc,log,row in active.values():
            file=Path(row['output'])/'progress.json';latest=None
            if file.exists():
                try:latest=json.loads(file.read_text())
                except json.JSONDecodeError:pass
            running.append(dict(row,latest=latest))
        status=dict(queue_pid=os.getpid(),frozen_commit=commit,unix_time=time.time(),running=running,completed=done,pending=pending,failed=failed,gpus4to7=resource,inventory_error=error)
        dump(out/'status.json',status)
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(status)+'\n')
        if failed and not active:break
        if pending or active:time.sleep(10)
    passed=not failed and not pending and len(done)==8
    dump(out/'queue-finished.json',dict(passed=passed,completed=done,unstarted=pending,frozen_commit=commit,portable_manifest_sha256=digest))
    if not passed:raise SystemExit(1)

if __name__=='__main__':main()
