"""Independent literal clocks, manual Heun and optional feature/FID replay."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
import study

def manual(model,source):
    batches=[];h=1./128
    with torch.no_grad():
        for start in range(0,len(source),64):
            x=torch.from_numpy(np.array(source[start:start+64]));labels=torch.zeros(len(x),dtype=torch.long)
            for j in range(128):
                t=torch.full((len(x),),j*h);nt=torch.full_like(t,(j+1)*h)
                a=(model(x,t,labels)-x)/torch.maximum(1-t[:,None,None,None],torch.tensor(.05))
                trial=x+h*a;b=(model(trial,nt,labels)-trial)/torch.maximum(1-nt[:,None,None,None],torch.tensor(.05))
                x=x+(.5*h)*(a+b)
            batches.append(x.numpy())
    return np.concatenate(batches)
def fid(x,y):
    import scipy.linalg
    x=x.astype(np.float64);y=y.astype(np.float64);a=x-x.mean(0);b=y-y.mean(0)
    s=scipy.linalg.svd(a@b.T,compute_uv=False,lapack_driver='gesvd')
    return float(np.sum((x.mean(0)-y.mean(0))**2)+np.sum(a*a)/(len(a)-1)+np.sum(b*b)/(len(b)-1)-2*s.sum()/np.sqrt((len(a)-1)*(len(b)-1)))
def bitwise(a,b):assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()

def main(root,phase):
    torch.set_num_threads(1);started=time.monotonic();study.verify(root,True);out=root/f'audit-{phase}';out.mkdir()
    p=study.paths(root)['image_parent'];old=study.imported(root,'image_parent');constants=study.load(root/'calibration.json')
    original={str(f):study.sha(f) for folder in (root/'output',root/'metric-output') if folder.exists() for f in folder.rglob('*') if f.is_file()};rows=[]
    if phase=='sampling':
        b=old.bank(p/'banks/validation.npz')
        for arm in study.ARMS:
            native=study.native_model(root,arm);recorded={};hooks=[]
            for name,layer in study.layers(native):
                def hook(layer,inputs,name=name):
                    x=inputs[0];l=layer.t_func;h=torch.nn.functional.linear(x,l[0].weight,l[0].bias)
                    z=torch.nn.functional.linear(torch.nn.functional.silu(h),l[2].weight,l[2].bias)
                    recorded[name]=(torch.sigmoid(z/10)*.8+.2).numpy()
                hooks.append(layer.register_forward_pre_hook(hook))
            with torch.no_grad():pred=native(torch.from_numpy(b['x'][:64]),torch.from_numpy(b['t'][:64]),torch.zeros(64,dtype=torch.long)).numpy()
            for hook in hooks:hook.remove()
            with np.load(p/'output/runs'/arm/'validation_0005000.npz') as f:bitwise(pred,f['prediction'][:64])
            with np.load(root/f'{arm}-native-calibration.npz') as saved:
                assert len(recorded)==6
                for name,value in recorded.items():
                    bitwise(value,saved[name]);assert float(np.float32(value.astype(np.float64).mean()))==constants[arm][name]['value']
            model=study.native_model(root,arm);before=study.tensors(model);study.patch(model,constants[arm],'independent');folder=root/'output/runs'/arm
            predicted=old.predict(model,b)
            with np.load(folder/'validation.npz') as saved:bitwise(predicted,saved['prediction'])
            error=predicted.astype(np.float64)-b['target'].astype(np.float64);delta=b['delta'].astype(np.float64)[:,None,None,None]
            # Explicit velocity difference instead of producer's clean residual.
            vp=(predicted.astype(np.float64)-b['x'].astype(np.float64))/delta;vt=(b['target'].astype(np.float64)-b['x'].astype(np.float64))/delta
            weighted=float(np.mean((vp-vt)**2));clean=float(np.mean(error**2));formal=study.load(folder/'result.json')
            np.testing.assert_allclose([weighted,clean],[formal['weighted_mse'],formal['clean_mse']],rtol=1e-10,atol=1e-12)
            for time_value in (.1,.3,.5,.7,.9):
                panel=old.bank(p/'banks'/f'panel_{time_value:.1f}.npz');prediction=old.predict(model,panel)
                with np.load(folder/f'panel-{time_value:.1f}.npz') as saved:bitwise(prediction,saved['prediction'])
            generated=manual(model,np.load(p/'banks/sampling-source.npy'));bitwise(generated,np.load(folder/'generated.npy'))
            study.unchanged(model,before);rows.append(dict(family=arm+'_constant',predictions_bitwise=True,samples=1024,samples_bitwise=True,panels=5,weighted_mse=weighted,clean_mse=clean,parameters_buffers_unchanged=True))
            print(json.dumps(dict(phase='sampling_audit',family=arm,seconds=time.monotonic()-started)),flush=True)
    else:
        sys.path.insert(0,str(p/'source'));import image_metrics as m
        from metric_core import uint8_images
        model=m.extractor(p);real=m.features(model,np.load(p/'banks/fid-reference.uint8.npy'));bitwise(real,np.load(p/'metric-inputs/reference-features.npy'))
        formal=study.load(root/'metric-output/result.json')['rows']
        for arm in (*study.NATIVE,*(a+'_constant' for a in study.ARMS)):
            stored=np.load(study.feature_bank(root,arm))
            if arm.endswith('_constant'):
                base=arm.removesuffix('_constant');generated=m.features(model,uint8_images(np.load(root/'output/runs'/base/'generated.npy')));bitwise(generated,stored)
            actual=fid(stored,real);expected=next(r['fid'] for r in formal if r['family']==arm);assert abs(actual-expected)<1e-8
            rows.append(dict(family=arm,fid=actual,difference=actual-expected,feature_provenance='fresh bitwise replay' if arm.endswith('_constant') else 'sealed cycle18 native bank/audit'))
            print(json.dumps(dict(phase='feature_audit',family=arm,seconds=time.monotonic()-started)),flush=True)
    for file,digest in original.items():assert study.sha(file)==digest,file
    study.verify(root,True);study.write(out/'result.json',dict(status='PASS',phase=phase,rows=rows,formal_files_unchanged=len(original),seconds=time.monotonic()-started))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--phase',choices=('sampling','features'),required=True);a=ap.parse_args();main(a.root,a.phase)
