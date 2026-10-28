"""Independent matrix algebra, cached-map/Heun replay and FID calculation."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
import argparse,json,sys,time,types
from pathlib import Path
import numpy as np
import scipy.linalg
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
def check_operators(root):
    history=study.load(root/'operator-history.json');assert len(history)==48
    for r in history:
        with np.load(root/'operators'/f"{r['family']}-{r['step']:07d}.npz") as bank:
            prefix=study.token(r['layer']);p=bank[prefix+'_P'];a=r['c']*p;e=scipy.linalg.expm(a);t=np.eye(64)+a
            if r['order']==2:t=t+(a@a)/2
            np.testing.assert_allclose(bank[prefix+'_E'],e,rtol=1e-11,atol=1e-11);np.testing.assert_allclose(bank[prefix+'_T'],t,rtol=1e-12,atol=1e-12)
            bitwise(bank[prefix+'_E32'],bank[prefix+'_E'].astype(np.float32));bitwise(bank[prefix+'_T32'],bank[prefix+'_T'].astype(np.float32))
        native=study.native(root,r['family'],r['step']);actual=next(layer for name,layer in study.layers(root,native) if name==r['layer']).proj.weight.detach().double().numpy();bitwise(p,actual)
        sv=np.linalg.svd(p,compute_uv=False);np.testing.assert_allclose(sv,r['P_singular_values'],rtol=1e-10,atol=1e-12)
        eig=list(np.linalg.eigvals(p));stored=np.asarray(r['P_eigen_real'])+1j*np.asarray(r['P_eigen_imag'])
        for value in stored:
            index=int(np.argmin(np.abs(np.asarray(eig)-value)));assert abs(eig.pop(index)-value)<1e-9
        h=np.eye(64)+a/2
        if r['order']==2:h=h+(a@a)/8
        half=scipy.linalg.expm(a/2);norm=np.linalg.norm(e)
        values=dict(P_sigma_max=sv[0],numerical_abscissa=np.linalg.eigvalsh((p+p.T)/2).max(),spectral_abscissa=max(r['P_eigen_real']),nonnormality=np.linalg.norm(p.T@p-p@p.T)/np.linalg.norm(p)**2,A_norm=np.linalg.norm(a,2),polynomial_norm=np.linalg.norm(t,2),exponential_norm=np.linalg.norm(e,2),relative_Frobenius_error=np.linalg.norm(e-t)/norm,polynomial_semigroup_defect=np.linalg.norm(h@h-t)/norm)
        for key,value in values.items():np.testing.assert_allclose(value,r[key],rtol=1e-9,atol=1e-11)
        assert np.linalg.norm(half@half-e)/norm<1e-11 and r['exponential_semigroup_defect']<1e-11
    return len(history)
def fid(x,y):
    x=x.astype(np.float64);y=y.astype(np.float64);a=x-x.mean(0);b=y-y.mean(0)
    s=scipy.linalg.svd(a@b.T,compute_uv=False,lapack_driver='gesvd')
    return float(np.sum((x.mean(0)-y.mean(0))**2)+np.sum(a*a)/(len(a)-1)+np.sum(b*b)/(len(b)-1)-2*s.sum()/np.sqrt((len(a)-1)*(len(b)-1)))

def main(root,phase):
    torch.set_num_threads(1);started=time.monotonic();study.verify(root,True);out=root/f'audit-{phase}';out.mkdir();p=study.paths(root)['image_parent'];old=study.imported(root,'image_parent')
    folders=(root/'output',) if phase=='sampling' else (root/'output',root/'metric-output')
    before={str(f):study.sha(f) for folder in folders for f in folder.rglob('*') if f.is_file()};rows=[]
    if phase=='sampling':
        count=check_operators(root);bank=old.bank(p/'banks/validation.npz')
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
