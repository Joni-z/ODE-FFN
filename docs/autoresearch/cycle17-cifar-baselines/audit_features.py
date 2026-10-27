"""Independent pixel/feature replay and stable covariance-FID trace recomputation."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import scipy.linalg
import torch
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
a=ap.parse_args();root=a.root;out=a.output;out.mkdir(exist_ok=False);torch.set_num_threads(1);started=time.monotonic()
load=lambda p:json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for p,h in load(root/'bindings.json').items():assert sha(p)==h
assert load(root/'metric-output/result.json')['status']=='PASS'
original={str(p):sha(p) for p in (root/'metric-output').rglob('*') if p.is_file()}
weights=list(Path(load(root/'settings.json')['weight_cache']).glob('checkpoints/*.pth'));assert len(weights)==1
model=FeatureExtractorInceptionV3('inception-v3-compat',['2048'],feature_extractor_weights_path=str(weights[0])).eval().requires_grad_(False).to('mps')

def feature(images):
    answer=[]
    with torch.no_grad():
        for start in range(0,len(images),32):answer.append(model(torch.from_numpy(np.array(images[start:start+32])).to('mps'))[0].cpu().numpy())
    return np.concatenate(answer)
def compare(x,y):
    e=x.astype(np.float64)-y.astype(np.float64);maximum=float(np.max(np.abs(e)));rms=float(np.sqrt(np.mean(e*e)))
    assert maximum<1e-3 and rms<1e-4,(maximum,rms)
    return dict(maximum=maximum,rms=rms)
def independent_fid(x,y):
    x=x.astype(np.float64);y=y.astype(np.float64);mx=np.mean(x,axis=0);my=np.mean(y,axis=0)
    x-=mx;y-=my;cross=x@y.T
    # Different LAPACK driver and unnormalized cross product from producer.
    trace=float(scipy.linalg.svd(cross,compute_uv=False,lapack_driver='gesvd').sum())/np.sqrt((len(x)-1)*(len(y)-1))
    score=float(np.dot(mx-my,mx-my)+(x*x).sum()/(len(x)-1)+(y*y).sum()/(len(y)-1)-2*trace)
    assert score>=-1e-6;return max(score,0.)

real=np.load(root/'metric-inputs/reference-features.npy');real_check=compare(feature(np.load(root/'banks/fid-reference.uint8.npy')),real)
rows=[]
for saved in load(root/'metric-output/result.json')['results']:
    arm=saved['family'];raw=np.load(root/'output/runs'/arm/'generated.npy')
    pixels=torch.from_numpy(raw).clamp(-1,1).add(1).mul(127.5).round().to(torch.uint8).numpy()
    features=feature(pixels);stored=np.load(root/'metric-output'/f'{arm}-features.npy');check=compare(features,stored)
    score=independent_fid(stored,real);np.testing.assert_allclose(score,saved['fid'],atol=1e-6,rtol=1e-8)
    rows.append(dict(family=arm,fid=score,producer_fid=saved['fid'],difference=score-saved['fid'],feature_replay=check))
    print(json.dumps(rows[-1]),flush=True)
assert original=={str(p):sha(p) for p in (root/'metric-output').rglob('*') if p.is_file()}
result=dict(status='PASS',reference_images=2048,generated_images=4096,reference_replay=real_check,results=rows,formal_files_unchanged=len(original),seconds=time.monotonic()-started)
(out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
