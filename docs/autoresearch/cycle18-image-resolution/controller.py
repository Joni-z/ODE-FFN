"""One-shot cycle18 orchestration: terminal exits, time ceilings, no retries."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import hashlib,json,subprocess,time,traceback
from pathlib import Path
import psutil
root=Path('/Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle18')
train='/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python';metrics='/Users/mr.z/ode-workspace-prep/.venv-image-metrics-v2/bin/python'
folder=root/'orchestrator';folder.mkdir();source=Path(__file__);(folder/'controller.py').write_bytes(source.read_bytes())
def write(name,value):
 p=folder/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,p)
amendment=root/'resource-amendment.md'
assert amendment.exists()
write('binding.json',{'controller_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'resource_amendment_sha256':hashlib.sha256(amendment.read_bytes()).hexdigest()})
assert json.loads((root/'preflight.json').read_text())['status']=='PASS'
phases=[('sampling',[train,str(root/'source/study.py'),'run','--root',str(root)],2500),
('audit-sampling',[train,str(root/'source/audit.py'),'--root',str(root),'--phase','sampling'],1200),
('metrics',[metrics,str(root/'source/study.py'),'evaluate','--root',str(root)],600),
('audit-features',[metrics,str(root/'source/audit.py'),'--root',str(root),'--phase','features'],600)]
records=[];p=None
try:
 for name,cmd,ceiling in phases:
  with (folder/(name+'.log')).open('w') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
  started=time.monotonic();r=dict(phase=name,command=cmd,pid=p.pid,create_time=psutil.Process(p.pid).create_time(),started_at=time.time());records.append(r);write('launches.json',records)
  while p.poll() is None:
   assert time.monotonic()-started<ceiling,(name,'registered ceiling')
   if psutil.pid_exists(p.pid):
    process=psutil.Process(p.pid)
    with (folder/'resources.jsonl').open('a') as f:f.write(json.dumps(dict(phase=name,pid=p.pid,seconds=time.monotonic()-started,cpu_seconds=sum(process.cpu_times()[:2]),rss=process.memory_info().rss))+'\n')
   write('status.json',dict(state='RUNNING',phase=name,pid=p.pid,seconds=time.monotonic()-started,completed=len(records)-1,total=4));time.sleep(5)
  r.update(exit_code=p.returncode,seconds=time.monotonic()-started);write('launches.json',records)
  assert p.returncode==0,r
  print(json.dumps(r),flush=True)
 write('status.json',dict(state='COMPLETE',phases=4,no_retries=True))
except BaseException:
 if p and p.poll() is None:
  process=psutil.Process(p.pid);children=process.children(recursive=True)
  for child in children:
   try:child.terminate()
   except psutil.NoSuchProcess:pass
  p.terminate()
 write('status.json',dict(state='FAILED',phases=records));traceback.print_exc();raise
