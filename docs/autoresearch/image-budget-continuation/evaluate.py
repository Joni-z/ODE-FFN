"""Reuse the frozen image evaluator after the fixed seven-arm cohort."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse,hashlib,json,shutil,sys,time
from pathlib import Path
import numpy as np
import psutil
import torch

PARENT=Path('/Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle17-v2')
RESOLUTION=PARENT.parent/'oct28-cycle18'
ARMS=('ode_k1','ode_k2','swiglu86','swiglu129','silu64','silu131','silu193')
sys.path.insert(0,str(PARENT/'source'))
import image_metrics as metrics
from metric_core import fid,uint8_images
load=lambda p:json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def reference():
    path=PARENT/'metric-inputs/reference-features.npy'
    for file,digest in load(PARENT/'metric-bindings.json').items():assert sha(file)==digest
    return np.load(path)
def preflight(root):
    torch.set_num_threads(1);assert not (root/'metric-preflight.json').exists()
    extractor=metrics.extractor(PARENT);raw=np.load(RESOLUTION/'output/runs/ode_k2/generated-128.npy')[:32]
    expected=np.load(RESOLUTION/'metric-output/ode_k2-128-features.npy')
    observed=metrics.features(extractor,uint8_images(raw));assert observed.tobytes()==expected[:32].tobytes()
    score=fid(expected,reference());assert abs(score-183.69806728894557)<1e-8
    receipt=dict(status='PASS',existing32features_bitwise=True,existing_K2_FID=score,reference=2048,
                 new_quality_estimate=False,source_sha256=sha(__file__),executable=sys.executable,torch=torch.__version__)
    write(root/'metric-preflight.json',receipt);print(json.dumps(receipt),flush=True)
def wait_training(root):
    identity=load(root/'controller-identity.json');tick=time.monotonic()
    while True:
        assert time.monotonic()-tick<28800
        try:
            process=psutil.Process(identity['pid'])
            assert abs(process.create_time()-identity['create_time'])<1e-3,'Controller PID reused'
            alive=process.is_running() and process.status()!=psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:alive=False
        summary=root/'output/summary.json'
        if summary.exists():
            result=load(summary);assert result['state']=='COMPLETE' and len(result['exits'])==7 and all(r['exit_code']==0 for r in result['exits']);return
        if not alive:raise RuntimeError('Training ended without a complete cohort; no automatic retry or evaluation')
        write(root/'metric-controller/status.json',dict(phase='WAITING_FOR_TRAINING',controller_pid=identity['pid'],elapsed=time.monotonic()-tick))
        time.sleep(5)
def evaluate(root,wait):
    torch.set_num_threads(1);receipt=load(root/'metric-preflight.json');assert receipt['status']=='PASS' and receipt['source_sha256']==sha(__file__)
    state=root/'metric-controller';state.mkdir(exist_ok=False)
    write(state/'identity.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),command=sys.argv))
    if wait:wait_training(root)
    summary=load(root/'output/summary.json');assert summary['state']=='COMPLETE'
    output=root/'metric-output';output.mkdir(exist_ok=False);shutil.copyfile(__file__,output/'evaluate-source.py')
    real=reference();extractor=metrics.extractor(PARENT);rows=[]
    for step in (10000,20000):
        for arm in ARMS:
            folder=root/'output/runs'/arm;assert load(folder/'status.json')['state']=='COMPLETE'
            raw_path=folder/f'generated-{step:07d}.npy';raw=np.load(raw_path);assert raw.shape==(1024,3,32,32)
            features=metrics.features(extractor,uint8_images(raw),progress=f'{arm}-{step}')
            assert features.shape==(1024,2048);np.save(output/f'{arm}-{step:07d}-features.npy',features)
            recorded=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
            fit=next(v for v in recorded if v['step']==step);score=fid(features,real)
            row=dict(arm=arm,step=step,**{k:fit[k] for k in ('weighted_mse','clean_mse','prediction_rms')},
                fid=score,generated=1024,reference=2048,Heun_steps=128,
                out_of_range_fraction=float(np.mean(np.abs(raw)>1)),sample_sha256=sha(raw_path),
                feature_sha256=sha(output/f'{arm}-{step:07d}-features.npy'))
            rows.append(row);write(output/'progress.json',dict(completed=len(rows),total=14,rows=rows))
            write(state/'status.json',dict(phase='EVALUATING',arm=arm,step=step,completed=len(rows),total=14))
    reference();write(output/'result.json',dict(status='PASS',rows=rows,source_sha256=sha(__file__),
        independent_audit_pending=True,new_architecture=False));write(state/'status.json',dict(phase='COMPLETE',completed=14))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('preflight','evaluate'));p.add_argument('--root',type=Path,required=True);p.add_argument('--wait',action='store_true');a=p.parse_args()
    preflight(a.root) if a.action=='preflight' else evaluate(a.root,a.wait)
