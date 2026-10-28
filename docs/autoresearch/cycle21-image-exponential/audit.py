"""Independent matrix algebra, cached-map/Heun replay and FID calculation."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
import argparse,json,sys,time,types,subprocess
from pathlib import Path
import numpy as np
import torch
import study

def bitwise(x,y):assert x.shape==y.shape and x.dtype==y.dtype and x.tobytes()==y.tobytes()
def manual(model,source):
    result=[];h=1/128
    with torch.no_grad():
        for offset in range(0,len(source),64):
            state=torch.from_numpy(np.array(source[offset:offset+64]));classes=torch.zeros(len(state),dtype=torch.int64)
            for k in range(128):
                left=torch.full((len(state),),k*h);right=torch.full_like(left,(k+1)*h)
                first=(model(state,left,classes)-state)/torch.maximum(1-left[:,None,None,None],torch.tensor(.05))
                trial=state+h*first
                second=(model(trial,right,classes)-trial)/torch.maximum(1-right[:,None,None,None],torch.tensor(.05))
                state=state+(.5*h)*(first+second)
            result.append(state.numpy())
    return np.concatenate(result)
def independent_model(root,arm):
    base,kind=arm.rsplit('_',1);model=study.native(root,base)
    with np.load(root/'operators'/f'{base}-0005000.npz') as arrays:
        for name,layer in study.layers(root,model):
            w=torch.tensor(arrays[study.token(name)+('_E' if kind=='exp' else '_T')],dtype=torch.float32)
            def forward(self,x,t=None,w=w):
                assert t is None
                return torch.nn.functional.linear(x,w)
            layer.forward=types.MethodType(forward,layer)
    return model
def fid(x,y):
    import scipy.linalg
    x=x.astype(np.float64);y=y.astype(np.float64);a=x-x.mean(0);b=y-y.mean(0)
    s=scipy.linalg.svd(a@b.T,compute_uv=False,lapack_driver='gesvd')
    return float(np.sum((x.mean(0)-y.mean(0))**2)+np.sum(a*a)/(len(a)-1)+np.sum(b*b)/(len(b)-1)-2*s.sum()/np.sqrt((len(a)-1)*(len(b)-1)))

def main(root,phase):
    torch.set_num_threads(1);started=time.monotonic();study.verify(root,True);out=root/f'audit-{phase}';out.mkdir();p=study.paths(root)['image_parent'];old=study.imported(root,'image_parent')
    folders=(root/'output',) if phase=='sampling' else (root/'output',root/'metric-output')
    before={str(f):study.sha(f) for folder in folders for f in folder.rglob('*') if f.is_file()};rows=[]
    if phase=='sampling':
        subprocess.run([study.CHECK,str(root/'source/operator_check.py'),'--root',str(root),'--output',str(out/'operator-check.json')],check=True)
        count=study.load(out/'operator-check.json')['operator_records'];bank=old.bank(p/'banks/validation.npz')
        for arm in study.ARMS:
            model=independent_model(root,arm);saved_tensors=study.tensors(root,model);folder=root/'output/runs'/arm;pred=old.predict(model,bank)
            with np.load(folder/'validation.npz') as f:bitwise(pred,f['prediction'])
            delta=bank['delta'].astype(np.float64)[:,None,None,None];error=pred.astype(np.float64)-bank['target'].astype(np.float64);weighted=float(np.mean((error/delta)**2));clean=float(np.mean(error**2));formal=study.load(folder/'result.json')
            np.testing.assert_allclose([weighted,clean],[formal['weighted_mse'],formal['clean_mse']],rtol=1e-12,atol=1e-12)
            for t in (.1,.3,.5,.7,.9):
                b=old.bank(p/'banks'/f'panel_{t:.1f}.npz');v=old.predict(model,b)
                with np.load(folder/f'panel-{t:.1f}.npz') as f:bitwise(v,f['prediction'])
            samples=manual(model,np.load(p/'banks/sampling-source.npy'));bitwise(samples,np.load(folder/'generated.npy'));study.unchanged(root,model,saved_tensors)
            rows.append(dict(family=arm,weighted_mse=weighted,clean_mse=clean,predictions_bitwise=True,time_panels=5,samples_bitwise=True,generated=1024,parameters_buffers_unchanged=True))
            print(json.dumps(dict(phase=phase,family=arm,seconds=time.monotonic()-started)),flush=True)
        metadata=dict(operator_records=count,new_generated_images=4096,time_panels=20)
    else:
        sys.path.insert(0,str(p/'source'));import image_metrics as metrics
        from metric_core import uint8_images
        extractor=metrics.extractor(p);real=metrics.features(extractor,np.load(p/'banks/fid-reference.uint8.npy'));bitwise(real,np.load(p/'metric-inputs/reference-features.npy'))
        formal=study.load(root/'metric-output/result.json')['rows'];assert len(formal)==13
        for row in formal:
            arm=row['family'];features=np.load(study.feature_bank(root,arm))
            if arm in study.ARMS:
                raw=np.load(root/'output/runs'/arm/'generated.npy');fresh=metrics.features(extractor,uint8_images(raw));bitwise(fresh,features)
            score=fid(features,real);assert abs(score-row['fid'])<1e-8
            rows.append(dict(family=arm,fid=score,difference=score-row['fid'],feature_provenance='new bitwise replay' if arm in study.ARMS else 'sealed prior audit/bank'))
            print(json.dumps(rows[-1]),flush=True)
        metadata=dict(new_generated_feature_rows=4096,reference_feature_rows=2048)
    for path,digest in before.items():assert study.sha(path)==digest,path
    study.verify(root,True);study.write(out/'result.json',dict(status='PASS',phase=phase,rows=rows,formal_files_unchanged=len(before),seconds=time.monotonic()-started,**metadata))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--phase',choices=('sampling','features'),required=True);a=parser.parse_args();main(a.root,a.phase)
