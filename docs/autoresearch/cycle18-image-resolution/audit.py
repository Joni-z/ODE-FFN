"""Separate sampling expression and independent FID SVD, with full replay."""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[name]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import scipy.linalg
import torch
import study

def replay(model,noise,n):
    batches=[];h=1./n
    with torch.no_grad():
        for offset in range(0,len(noise),64):
            state=torch.from_numpy(np.array(noise[offset:offset+64]));classes=torch.zeros(len(state),dtype=torch.int64)
            for step in range(n):
                left=torch.full((len(state),),step*h);right=torch.full_like(left,(step+1)*h)
                first=(model(state,left,classes)-state)/torch.maximum(1-left[:,None,None,None],torch.tensor(.05))
                trial=state+h*first
                second=(model(trial,right,classes)-trial)/torch.maximum(1-right[:,None,None,None],torch.tensor(.05))
                state=state+(.5*h)*(first+second)
            batches.append(state.numpy())
    return np.concatenate(batches)

def independent_fid(x,y):
    x=x.astype(np.float64);y=y.astype(np.float64);mx=x.mean(0);my=y.mean(0);a=x-mx;b=y-my
    singular=scipy.linalg.svd(a@b.T,compute_uv=False,lapack_driver='gesvd')
    return float(np.sum((mx-my)**2)+np.sum(a*a)/(len(a)-1)+np.sum(b*b)/(len(b)-1)-2*singular.sum()/np.sqrt((len(a)-1)*(len(b)-1)))

def main(root,phase):
    torch.set_num_threads(1);start=time.monotonic();study.verify(root);p=study.parent(root);out=root/f'audit-{phase}';out.mkdir()
    original={str(f):study.sha(f) for folder in (root/'output',root/'metric-output') if folder.exists() for f in folder.rglob('*') if f.is_file()}
    if phase=='sampling':
        assert study.load(root/'output/status.json')['state']=='SAMPLING_COMPLETE';source=np.load(p/'banks/sampling-source.npy');rows=[]
        for arm in study.ARMS:
            model=study.model_for(root,arm)
            for n in (64,128):
                got=replay(model,source,n);np.testing.assert_array_equal(got,np.load(root/'output/runs'/arm/f'generated-{n}.npy'))
                rows.append(dict(family=arm,N=n,images=len(got),bitwise=True))
                print(json.dumps(dict(phase='sampling_audit',family=arm,N=n,seconds=time.monotonic()-start)),flush=True)
    else:
        sys.path.insert(0,str(p/'source'));import image_metrics as m
        from metric_core import uint8_images
        model=m.extractor(p);reference=np.load(p/'metric-inputs/reference-features.npy');rows=[]
        real=m.features(model,np.load(p/'banks/fid-reference.uint8.npy'))
        np.testing.assert_array_equal(real,reference)
        formal=study.load(root/'metric-output/result.json')['rows']
        for arm in study.ARMS:
            for n in study.GRIDS:
                raw=np.load(p/'output/runs'/arm/'generated.npy') if n==32 else np.load(root/'output/runs'/arm/f'generated-{n}.npy')
                replayed=m.features(model,uint8_images(raw))
                saved=np.load(p/'metric-output'/f'{arm}-features.npy') if n==32 else np.load(root/'metric-output'/f'{arm}-{n}-features.npy')
                np.testing.assert_array_equal(replayed,saved)
                score=independent_fid(replayed,real);old=next(r for r in formal if r['family']==arm and r['N']==n)
                assert abs(score-old['fid'])<1e-8
                rows.append(dict(family=arm,N=n,fid=score,difference=score-old['fid'],features_bitwise=True))
                print(json.dumps(dict(phase='feature_audit',family=arm,N=n,seconds=time.monotonic()-start)),flush=True)
    for file,digest in original.items():assert study.sha(file)==digest,file
    study.verify(root);study.write(out/'result.json',dict(status='PASS',phase=phase,rows=rows,formal_files_unchanged=len(original),seconds=time.monotonic()-start))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--phase',choices=('sampling','features'),required=True);a=ap.parse_args();main(a.root,a.phase)
