"""Six cold seed-0 students, immutable teacher/input caches and real monitoring."""
import argparse,hashlib,os,subprocess,sys,threading,time,traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import psutil
import torch
from teacher_support import TEACHERS,STUDENTS,TIMES,SNAPSHOTS,BANKS,load,write,append,sha,setup,student,verify_cold,optimizer,update,predict,open_cache,batch,snapshot,signature,external_bindings,replay_hash
from prepare_cache import prepare

def observe():
    parent=psutil.Process();rows=[]
    for p in [parent]+parent.children(recursive=True):
        try:rows.append(dict(pid=p.pid,rss_bytes=p.memory_info().rss,cpu_seconds=sum(p.cpu_times()[:2]),command=p.cmdline()))
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),available_memory_bytes=psutil.virtual_memory().available,processes=rows)

def field_eval(model,cache_root,teacher_name,save=None):
    with np.load(BANKS/'field.npz') as d:x,t,oracle=d['x'],d['t'],d['oracle']
    with np.load(cache_root/'evaluation'/teacher_name/'field.npz') as d:truth=d['prediction'].astype(np.float64)
    actual=predict(model,x,t);err=actual.astype(np.float64)-truth;energy=float(np.mean(truth**2));assert energy>0
    if save is not None:np.savez_compressed(save,prediction=actual)
    return dict(teacher_mse=float(np.mean(err**2)),relative_teacher_mse=float(np.mean(err**2)/energy),teacher_energy=energy,
        secondary_oracle_mse=float(np.mean((actual.astype(np.float64)-oracle.astype(np.float64))**2)))

def final_panels(model,cache_root,teacher_name,folder):
    from scripts.check_gate_geometry import jacobian
    rows=[];model.eval()
    for t in TIMES:
        with np.load(BANKS/f'panel_{t:.1f}.npz') as d:x,clock=d['x'],d['t']
        with np.load(cache_root/'evaluation'/teacher_name/f'panel_{t:.1f}.npz') as d:y,j=d['prediction'].astype(np.float64),d['jacobian'].astype(np.float64)
        value,derivative=jacobian(model,torch.from_numpy(x),torch.from_numpy(clock));value=value.numpy();derivative=derivative.numpy()
        mse=float(np.mean((value.astype(np.float64)-y)**2));je=float(np.mean(np.sum((derivative.astype(np.float64)-j)**2,axis=(1,2))))
        rows.append(dict(time=t,teacher_mse=mse,relative_teacher_mse=mse/float(np.mean(y*y)),jacobian_error=je,
            relative_jacobian_error=je/float(np.mean(np.sum(j*j,axis=(1,2))))))
        np.savez_compressed(folder/f'panel_{t:.1f}.npz',prediction=value,jacobian=derivative)
    return rows

def cell(root,teacher_name,family):
    torch.set_num_threads(1);setup();source=Path(__file__).parent;out=root/'output';folder=out/'runs'/(teacher_name+'--'+family);folder.mkdir(exist_ok=False)
    started=time.monotonic()
    def status(state='RUNNING',step=0,**extra):write(folder/'status.json',dict(state=state,pid=os.getpid(),step=step,elapsed_seconds=time.monotonic()-started,**extra))
    status()
    try:
        for name,digest in load(source/'source-sha256.json').items():assert sha(source/name)==digest,name
        cache_root=out/'cache';metadata=load(cache_root/'metadata.json')
        assert metadata['steps']==5000
        # Cache hashes were established before launching any student; bind them
        # and recheck all bytes independently at final audit.
        model=student(family);cold_cp=verify_cold(model,family);initial=signature(model);opt=optimizer(model);cache=open_cache(cache_root)
        from synthetic.conservative_readout import compute_budget
        costs=compute_budget(model)
        write(folder/'provenance.json',dict(teacher=teacher_name,student=family,seed=0,steps=5000,batch_size=512,
            actual_device='cpu',threads=torch.get_num_threads(),initial_signature=initial,cold_reference=str(cold_cp),cold_reference_sha256=sha(cold_cp),
            cache_metadata_sha256=sha(cache_root/'metadata.json'),source_manifest_sha256=sha(source/'source-sha256.json'),costs=costs))
        actual_replay=dict(input=hashlib.sha256(b'').hexdigest(),label=hashlib.sha256(b'').hexdigest())
        snapshot(model,opt,0,folder,metadata,teacher_name,actual_replay.copy());append(folder/'metrics.jsonl',dict(step=0,**field_eval(model,cache_root,teacher_name,folder/'field_0000000.npz')))
        loss_sum=grad_sum=0.;clipped=count=0;model.train()
        for step in range(1,5001):
            x,t,y=batch(cache,teacher_name,step-1);loss,grad=update(model,opt,x,t,y)
            actual_replay['input']=replay_hash(actual_replay['input'],x,t);actual_replay['label']=replay_hash(actual_replay['label'],y)
            loss_sum+=loss;grad_sum+=grad;clipped+=int(grad>1);count+=1
            if step%100==0:status(step=step)
            if step%500==0:
                record=dict(step=step,training_loss=loss_sum/count,mean_gradient_norm=grad_sum/count,clip_fraction=clipped/count,elapsed_seconds=time.monotonic()-started)
                if step in SNAPSHOTS:
                    record.update(field_eval(model,cache_root,teacher_name,folder/f'field_{step:07d}.npz'))
                    snapshot(model,opt,step,folder,metadata,teacher_name,actual_replay.copy())
                    from scripts import run_recurrent_depth as base
                    write(folder/f'visits_{step:07d}.json',base.visit_diagnostics(model,BANKS))
                append(folder/'metrics.jsonl',record);print(record,flush=True);loss_sum=grad_sum=0.;clipped=count=0;model.train()
        panels=final_panels(model,cache_root,teacher_name,folder)
        result=dict(teacher=teacher_name,student=family,seed=0,step=5000,**field_eval(model,cache_root,teacher_name),panels=panels,
            mean_panel_teacher_mse=float(np.mean([r['teacher_mse'] for r in panels])),mean_panel_jacobian_error=float(np.mean([r['jacobian_error'] for r in panels])),
            **costs,input_stream_digest=metadata['snapshots']['5000']['input_digest'],teacher_label_digest=metadata['snapshots']['5000']['label_digests'][teacher_name],
            actual_replay_digests=actual_replay,
            final_signature=signature(model),elapsed_seconds=time.monotonic()-started)
        assert result['final_signature']!=initial
        write(folder/'result.json',result);status(state='COMPLETE',step=5000)
    except BaseException:
        status(state='FAILED',error=traceback.format_exc());raise

def run(root):
    torch.set_num_threads(1);source=Path(__file__).parent
    for name,digest in load(source/'source-sha256.json').items():assert sha(source/name)==digest,name
    assert load(root/'preflight.json')['status']=='PASS' and load(root/'preflight-streams.json')['status']=='PASS'
    out=root/'output';out.mkdir(exist_ok=False);started=time.monotonic();stop=threading.Event();abort=threading.Event();lock=threading.Lock();processes=set()
    progress=dict(state='RUNNING',phase='prepare',completed=0,total=6,pid=os.getpid())
    def status(**extra):progress.update(extra,elapsed_seconds=time.monotonic()-started);write(out/'status.json',progress);print(progress,flush=True)
    def monitor():
        while not stop.is_set():
            row=observe();append(out/'monitor.jsonl',row);write(out/'monitor.json',row);stop.wait(5)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start();status()
    try:
        bindings=external_bindings();bindings[str(source/'source-sha256.json')]=sha(source/'source-sha256.json');bindings[str(root/'preflight.json')]=sha(root/'preflight.json');bindings[str(root/'preflight-streams.json')]=sha(root/'preflight-streams.json')
        write(out/'input-bindings.json',bindings)
        prepare(out/'cache',steps=5000,progress=lambda **r:status(**r))
        (out/'runs').mkdir();status(phase='training',step=None)
        env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
        def launch(job):
            teacher_name,family=job;command=[sys.executable,str(source/'run_study.py'),'cell','--root',str(root),'--teacher',teacher_name,'--student',family]
            with (out/(teacher_name+'--'+family+'.log')).open('w') as log:
                with lock:
                    if abort.is_set():raise RuntimeError('study aborted')
                    p=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT);processes.add(p)
                    append(out/'launches.jsonl',dict(teacher=teacher_name,student=family,pid=p.pid,command=command))
                code=p.wait()
                with lock:processes.remove(p)
            if code:raise RuntimeError(f'cell {job} exited {code}')
            return job
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures=[executor.submit(launch,(t,s)) for t in TEACHERS for s in STUDENTS]
            try:
                for f in as_completed(futures):job=f.result();status(completed=progress['completed']+1,latest=list(job))
            except BaseException:
                abort.set()
                for f in futures:f.cancel()
                with lock:
                    for p in processes:p.terminate()
                raise
        for name,digest in bindings.items():assert sha(name)==digest,name
        for name,digest in load(source/'source-sha256.json').items():assert sha(source/name)==digest,name
        status(state='COMPLETE',phase='complete')
    except BaseException:
        status(state='FAILED',error=traceback.format_exc());raise
    finally:stop.set();watcher.join(timeout=6)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','cell']);p.add_argument('--root',type=Path,required=True);p.add_argument('--teacher',choices=TEACHERS);p.add_argument('--student',choices=STUDENTS);a=p.parse_args()
    if a.action=='run':run(a.root)
    else:cell(a.root,a.teacher,a.student)
