"""Watch the original training process, then run fixed evaluation/audits once."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
import psutil

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--training-pid',type=int,required=True)
a=ap.parse_args();root=a.root.resolve();sys.path.insert(0,str(root/'source'));import study
out=root/'postprocess';out.mkdir(exist_ok=False);started=time.monotonic()
metric_python=study.load(root/'settings.json')['metrics_python'];train_python='/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python'
phase='waiting_training';live=[]

def verify():
    study.verify(root,True)
    for p,h in study.load(root/'postprocess-bindings.json').items():assert study.sha(p)==h,p

def run_phase(name,commands):
    global phase,live
    phase=name;live=[];launches=[];tick=time.monotonic()
    for label,command in commands:
        f=(out/f'{label}.log').open('w');env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1')
        p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env)
        row=dict(label=label,pid=p.pid,command=command);launches.append(row);live.append((p,f,row))
    study.write(out/f'{name}-launches.json',launches)
    failures=[]
    while live:
        active=[]
        for p,f,row in list(live):
            code=p.poll()
            if code is not None:
                row['exit_code']=code;f.close();live.remove((p,f,row))
                if code:failures.append(row)
            else:
                try:
                    q=psutil.Process(p.pid);active.append(dict(pid=p.pid,rss_bytes=q.memory_info().rss,cpu_seconds=sum(q.cpu_times()[:2])))
                except psutil.NoSuchProcess:pass
        study.write(out/f'{name}-launches.json',launches)
        study.write(out/'status.json',dict(state='RUNNING',phase=name,active=active,elapsed=time.monotonic()-started))
        study.append(out/'monitor.jsonl',dict(phase=name,elapsed=time.monotonic()-started,processes=active))
        if time.monotonic()-tick>1800:
            for p,f,row in live:p.terminate()
            raise RuntimeError('Postprocessing phase ceiling exceeded')
        if live:time.sleep(5)
    assert not failures,failures
    verify()

try:
    verify();q=psutil.Process(a.training_pid)
    assert str(root/'source/study.py') in q.cmdline() and 'run' in q.cmdline()
    study.write(out/'training-watch.json',dict(pid=q.pid,create_time=q.create_time(),command=q.cmdline()))
    while True:
        state=study.load(root/'output/status.json')
        if state['state']=='TRAINING_COMPLETE':
            launches=study.load(root/'output/launches.json');assert len(launches)==4 and all(r['exit_code']==0 for r in launches)
            assert all(study.load(root/'output/runs'/arm/'status.json')['state']=='COMPLETE' for arm in study.ARMS)
            break
        assert state['state']=='RUNNING' and not state['failures'],state
        try:
            live_parent=psutil.Process(a.training_pid)
            assert live_parent.create_time()==q.create_time()
        except psutil.NoSuchProcess:
            # Re-read the terminal marker once; absence of a live process alone
            # never authorizes starting/restarting a training trajectory.
            if study.load(root/'output/status.json')['state']=='TRAINING_COMPLETE':continue
            raise RuntimeError('Original parent exited without a terminal completion marker')
        study.write(out/'status.json',dict(state='RUNNING',phase=phase,training_parent=a.training_pid,active=state['active'],pending=state['pending'],elapsed=time.monotonic()-started))
        time.sleep(5)
    run_phase('metrics_and_training_audit',[
        ('metrics',[metric_python,str(root/'source/image_metrics.py'),'evaluate','--root',str(root)]),
        ('training-audit',[train_python,str(root/'postprocess-source/audit_training.py'),'--root',str(root),'--output',str(root/'audit-training')])])
    run_phase('feature_audit',[
        ('feature-audit',[metric_python,str(root/'postprocess-source/audit_features.py'),'--root',str(root),'--output',str(root/'audit-features')])])
    assert study.load(root/'audit-training/audit.json')['status']=='PASS'
    assert study.load(root/'audit-features/audit.json')['status']=='PASS'
    study.write(out/'status.json',dict(state='COMPLETE',phase='AUDITED',pid=os.getpid(),seconds=time.monotonic()-started))
    print(json.dumps(study.load(out/'status.json')),flush=True)
except BaseException as e:
    study.write(out/'status.json',dict(state='FAILED',phase=phase,error=repr(e),pid=os.getpid()))
    traceback.print_exc();raise
