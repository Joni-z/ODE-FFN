"""Two monitored CPU inference workers; no training or retries."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from coefficient_trial import base,HERE

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--only',choices=['all','symmetry-full'],default='all');p.add_argument('--symmetry-dtype',choices=['float32','float64'],default='float32');a=p.parse_args()
    root=Path(a.root);out=Path(a.output);out.mkdir(exist_ok=False);tasks=[]
    for c in json.loads((root/'portable.json').read_text())['cases']:
        if a.only=='all':
            tasks.append((c['id'],'outer-subset'))
            if c['family'] in ['state_midpoint4','free_stage_midpoint4']:tasks.append((c['id'],'full-internal'))
    for c in json.loads((root/'portable.json').read_text())['cases']:
        if c['dataset']=='gmm_ring8_2d':tasks.append((c['id'],'symmetry-full'))
    active={};done=[];failed=False
    while tasks or active:
        for pid,(proc,log,row) in list(active.items()):
            if proc.poll() is not None:
                log.close();row['exit_code']=proc.returncode
                row['complete']=(Path(row['output'])/'COMPLETE.json').exists();failed|=proc.returncode!=0 or not row['complete'];done.append(row);del active[pid]
        while tasks and len(active)<2 and not failed:
            name,mode=tasks.pop(0);target=out/name/mode;log=(out/(name+'-'+mode+'.log')).open('x')
            script='full_symmetry_case.py' if mode=='symmetry-full' else 'cpu_resolution.py'
            cmd=[sys.executable,str(HERE/script),'--root',str(root),'--case',name,'--output',str(target)]
            if mode!='symmetry-full':cmd+=['--mode',mode]
            else:cmd+=['--dtype',a.symmetry_dtype]
            proc=subprocess.Popen(cmd,stdout=log,stderr=log,stdin=subprocess.DEVNULL,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
            active[proc.pid]=(proc,log,dict(id=name,mode=mode,pid=proc.pid,output=str(target),command=cmd,started=time.time()))
        base.dump(out/'status.json',dict(pid=os.getpid(),unix_time=time.time(),running=[r for _,_,r in active.values()],completed=done,pending=tasks,failed=failed,cpu_load=list(os.getloadavg())))
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),running=list(active),completed=len(done),pending=len(tasks),failed=failed))+'\n')
        if failed and not active:break
        if tasks or active:time.sleep(10)
    base.dump(out/'queue-finished.json',dict(passed=not failed,completed=done,unstarted=tasks))
    if failed:raise SystemExit(1)

if __name__=='__main__':main()
