"""Explicit clocks and a separate Heun expression for the two new banks."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
import study

def same(a,b):assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def manual(model,source):
    outputs=[];h=1./128
    with torch.no_grad():
        for offset in range(0,len(source),64):
            x=torch.from_numpy(np.array(source[offset:offset+64]));y=torch.zeros(len(x),dtype=torch.long)
            for j in range(128):
                t=torch.full((len(x),),j*h);nt=torch.full_like(t,(j+1)*h)
                first=(model(x,t,y)-x)/(1-t[:,None,None,None]).clamp_min(.05)
                trial=x+h*first
                second=(model(trial,nt,y)-trial)/(1-nt[:,None,None,None]).clamp_min(.05)
                x=x+(.5*h)*(first+second)
            outputs.append(x.numpy())
    return np.concatenate(outputs)
def fid(a,b):
    import scipy.linalg
    a=a.astype(np.float64);b=b.astype(np.float64);x=a-a.mean(0);y=b-b.mean(0)
    s=scipy.linalg.svd(x@y.T,compute_uv=False,lapack_driver='gesvd')
    return float(np.sum((a.mean(0)-b.mean(0))**2)+np.sum(x*x)/(len(a)-1)+np.sum(y*y)/(len(b)-1)-2*s.sum()/np.sqrt((len(a)-1)*(len(b)-1)))
def sampling(root):
    c=study.clocks();b=study.bank();constants=study.load(root/'calibration.json');rows=[]
    for arm in study.ARMS:
        native=study.native_model(arm);recorded={};hooks=[]
        for name,layer in c.layers(native):
            def hook(layer,args,name=name):
                assert (layer.tau,layer.scale,layer.shift)==(10.,.8,.2)
                x=args[0];network=layer.t_func
                hidden=torch.nn.functional.linear(x,network[0].weight,network[0].bias)
                logits=torch.nn.functional.linear(torch.nn.functional.silu(hidden),network[2].weight,network[2].bias)
                recorded[name]=(torch.sigmoid(logits/10)*.8+.2).numpy()
            hooks.append(layer.register_forward_pre_hook(hook))
        with torch.no_grad():prediction=native(torch.from_numpy(b['x'][:64]),torch.from_numpy(b['t'][:64]),torch.zeros(64,dtype=torch.long)).numpy()
        for h in hooks:h.remove()
        with np.load(study.PARENTS['budget']/'output/runs'/arm/'validation_0020000.npz') as saved:same(prediction,saved['prediction'][:64])
        with np.load(root/f'{arm}-native-calibration.npz') as saved:
            assert len(recorded)==6
            for name,value in recorded.items():
                same(value,saved[name]);assert float(np.float32(value.astype(np.float64).mean()))==constants[arm][name]['value']
        model=study.native_model(arm);before=c.tensors(model);c.patch(model,constants[arm],'independent')
        folder=root/'output/runs'/arm;prediction=study.old().predict(model,b)
        with np.load(folder/'validation.npz') as saved:same(prediction,saved['prediction'])
        delta=b['delta'].astype(np.float64)[:,None,None,None]
        predicted_v=(prediction.astype(np.float64)-b['x'].astype(np.float64))/delta
        target_v=(b['target'].astype(np.float64)-b['x'].astype(np.float64))/delta
        weighted=float(np.mean((predicted_v-target_v)**2));clean=float(np.mean((prediction.astype(np.float64)-b['target'].astype(np.float64))**2))
        formal=study.load(folder/'result.json')
        np.testing.assert_allclose([weighted,clean],[formal['weighted_mse'],formal['clean_mse']],rtol=1e-10,atol=1e-12)
        for t in (.1,.3,.5,.7,.9):
            observed=study.old().predict(model,study.bank(t))
            with np.load(folder/f'panel-{t:.1f}.npz') as saved:same(observed,saved['prediction'])
        generated=manual(model,np.load(study.PARENTS['native']/'banks/sampling-source.npy'))
        same(generated,np.load(folder/'generated.npy'));c.unchanged(model,before)
        rows.append(dict(family=arm+'_constant',weighted_mse=weighted,clean_mse=clean,predictions_bitwise=True,
            generated_images_bitwise=1024,time_panels_bitwise=5,calibrated_clock_arrays_bitwise=6,parameters_buffers_unchanged=True))
        print(json.dumps(rows[-1]),flush=True)
    return dict(rows=rows,new_generated_images_bitwise=2048,new_time_panels_bitwise=10)
def features(root):
    sys.path.insert(0,str(study.PARENTS['native']/'source'));import image_metrics as m
    from metric_core import uint8_images
    extractor=m.extractor(study.PARENTS['native']);reference=m.features(extractor,np.load(study.PARENTS['native']/'banks/fid-reference.uint8.npy'))
    same(reference,np.load(study.PARENTS['native']/'metric-inputs/reference-features.npy'))
    formal=study.load(root/'metric-output/result.json')['rows'];rows=[]
    for row in formal:
        family=row['family'];saved=np.load(study.feature_bank(root,family))
        if family.endswith('_constant'):
            observed=m.features(extractor,uint8_images(np.load(study.sample_bank(root,family))));same(observed,saved)
        value=fid(saved,reference);assert abs(value-row['fid'])<1e-8
        rows.append(dict(family=family,fid=value,difference=value-row['fid'],feature_provenance='fresh bitwise replay' if family.endswith('_constant') else 'sealed budget full audit'))
        print(json.dumps(rows[-1]),flush=True)
    assert len(rows)==9
    return dict(rows=rows,new_generated_features_bitwise=2048,reference_features_bitwise=2048)
def main(root,phase):
    torch.set_num_threads(1);study.verify(root,True);out=root/f'audit-{phase}';out.mkdir(exist_ok=False)
    original={str(p):study.sha(p) for folder in (root/'output',root/'metric-output') if folder.exists() for p in folder.rglob('*') if p.is_file()}
    tick=time.monotonic();result=sampling(root) if phase=='sampling' else features(root)
    for p,digest in original.items():assert study.sha(p)==digest,p
    study.verify(root,True);study.write(out/'result.json',dict(status='PASS',phase=phase,**result,formal_files_unchanged=len(original),seconds=time.monotonic()-tick))
    print(json.dumps(dict(status='PASS',phase=phase)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=('sampling','features'),required=True);a=p.parse_args();main(a.root,a.phase)
