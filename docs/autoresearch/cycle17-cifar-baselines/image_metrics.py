"""Pinned standard Inception features; no trained-network or feature changes."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,importlib.metadata,json,sys,time
from pathlib import Path
import numpy as np
import scipy.linalg
import torch
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from metric_core import fid,uint8_images


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
    p=Path(p);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load(p):return json.loads(Path(p).read_text())
def extractor(root=None,device='mps'):
    cache=Path('/Users/mr.z/ode-workspace-prep/image-bridge-inception');weights=list(cache.glob('checkpoints/*.pth'));assert len(weights)==1
    if root is not None:
        for p,h in load(root/'bindings.json').items():assert sha(p)==h,p
    return FeatureExtractorInceptionV3('inception-v3-compat',['2048'],feature_extractor_weights_path=str(weights[0])).eval().requires_grad_(False).to(device)
def features(model,images,device='mps',progress=None):
    answer=[]
    with torch.no_grad():
        for i in range(0,len(images),32):
            x=torch.from_numpy(np.array(images[i:i+32])).to(device);f=model(x)[0].cpu().numpy();answer.append(f)
            if progress and (i%256==0 or i+32>=len(images)):
                print(json.dumps(dict(phase=progress,processed=min(i+32,len(images)),total=len(images))),flush=True)
    return np.concatenate(answer)

def fixture(out):
    torch.set_num_threads(1);rng=np.random.default_rng(0);rows=[]
    for n,m,d in ((32,40,8),(12,15,20),(9,11,3)):
        x=rng.standard_normal((n,d));y=rng.standard_normal((m,d))+.2
        a=np.cov(x,rowvar=False);b=np.cov(y,rowvar=False);root=scipy.linalg.sqrtm(a@b)
        if np.iscomplexobj(root):assert np.max(np.abs(root.imag))<1e-6;root=root.real
        expected=float(np.sum((x.mean(0)-y.mean(0))**2)+np.trace(a)+np.trace(b)-2*np.trace(root))
        actual=fid(x,y);np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-8)
        np.testing.assert_allclose(fid(x,x),0,atol=1e-6)
        rows.append(dict(n=n,m=m,d=d,low_rank=actual,covariance=expected,abs_difference=abs(actual-expected)))
    # Deterministic pixel conversion, including exact black/white and roundtrip.
    pixels=rng.integers(0,256,(8,3,32,32),dtype=np.uint8)
    np.testing.assert_array_equal(uint8_images(pixels.astype(np.float32)/127.5-1),pixels)
    model=extractor(device='cpu');cpu=features(model,pixels,device='cpu');model.to('mps');mps=features(model,pixels)
    error=mps.astype(np.float64)-cpu.astype(np.float64);maximum=float(np.max(np.abs(error)));rms=float(np.sqrt(np.mean(error**2)))
    assert maximum<1e-3 and rms<1e-4,(maximum,rms)
    weights=list(Path('/Users/mr.z/ode-workspace-prep/image-bridge-inception').glob('checkpoints/*.pth'))
    write(out,dict(status='PASS',FID_identity=rows,feature_CPU_MPS_max=maximum,feature_CPU_MPS_RMS=rms,
        weights=str(weights[0]),weights_sha256=sha(weights[0]),pixel_roundtrip_bitwise=True,
        versions={n:importlib.metadata.version(n) for n in ('torch','torchvision','torch-fidelity','scipy','numpy')}))
    print(json.dumps(load(out)),flush=True)

def reference(root):
    start=time.monotonic();torch.set_num_threads(1);model=extractor(root)
    folder=root/'metric-inputs';folder.mkdir(exist_ok=False);images=np.load(root/'banks/fid-reference.uint8.npy',mmap_mode='r')
    data=features(model,images,progress='reference_features');assert data.shape==(2048,2048);np.save(folder/'reference-features.npy',data)
    write(root/'metric-bindings.json',{str(folder/'reference-features.npy'):sha(folder/'reference-features.npy')})
    write(root/'metric-ready.json',dict(status='PASS',reference_images=2048,features=2048,device='mps',seconds=time.monotonic()-start))

def evaluate(root):
    start=time.monotonic();torch.set_num_threads(1);model=extractor(root)
    assert load(root/'output/status.json')['state']=='TRAINING_COMPLETE'
    for p,h in load(root/'metric-bindings.json').items():assert sha(p)==h,p
    out=root/'metric-output';out.mkdir(exist_ok=False);real=np.load(root/'metric-inputs/reference-features.npy');rows=[]
    for arm in ('ode_k1','ode_k2','swiglu86','swiglu129'):
        raw=np.load(root/'output/runs'/arm/'generated.npy');pixels=uint8_images(raw)
        generated=features(model,pixels,progress=arm);assert generated.shape==(1024,2048)
        np.save(out/f'{arm}-features.npy',generated);score=fid(generated,real)
        rows.append(dict(family=arm,fid=score,generated=1024,reference=2048,feature_dimension=2048,device='mps'))
        write(out/'progress.json',dict(completed=len(rows),total=4,elapsed=time.monotonic()-start))
        print(json.dumps(rows[-1]),flush=True)
    write(out/'result.json',dict(status='PASS',results=rows,seconds=time.monotonic()-start))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('fixture','reference','evaluate'));ap.add_argument('--root',type=Path);ap.add_argument('--output',type=Path)
    a=ap.parse_args()
    if a.action=='fixture':fixture(a.output)
    elif a.action=='reference':reference(a.root)
    else:evaluate(a.root)
