"""Separate affine calibration, clock lookup and Heun/feature replay."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse,sys,time,types
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import study

def same(a,b):assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def verify_calibration(root):
    model=study.native_model();before=study.clocks().tensors(model);b=study.bank();raw={name:[] for name,_ in study.clocks().layers(model)};hooks=[]
    for name,layer in study.clocks().layers(model):
        def record(m,args,name=name):
            a=F.linear(args[0],m.t_func[0].weight,m.t_func[0].bias)
            z=F.linear(F.silu(a),m.t_func[2].weight,m.t_func[2].bias)
            value=(torch.sigmoid(z/10.)*.8+.2).squeeze(-1)
            raw[name].append(value.detach().numpy())
        hooks.append(layer.register_forward_pre_hook(record))
    try:
        with torch.no_grad():
            for j in range(129):
                value=np.float32(j)*np.float32(1/128)
                x=b['target'][:64]*value+b['source'][:64]*(np.float32(1)-value)
                same(x,study.calibration_input(j,b))
                model(torch.from_numpy(x),torch.full((64,),float(value)),torch.zeros(64,dtype=torch.long))
    finally:
        for h in hooks:h.remove()
    study.clocks().unchanged(model,before);table,pooled=study.calibration(root)
    with np.load(root/'native-calibration.npz') as f:
        assert set(f.files)==set(raw)==set(table)==set(pooled)
        for name,parts in raw.items():
            value=np.stack(parts);same(value,f[name]);d=value.astype(np.float64)
            # Flatten the image/token axes explicitly, independent of producer.
            means=np.sum(d.reshape(129,-1),axis=1)/4096
            same(means.astype(np.float32),table[name])
            assert float(np.float32(np.sum(d)/d.size))==pooled[name]
    return dict(native_arrays_bitwise=6,table_scalars=774,pooled_scalars=6,parameters_buffers_unchanged=True)
def patched(root,mode):
    model=study.native_model();table,pooled=study.calibration(root);context={}
    def external(m,args):context['time']=args[1]
    model.register_forward_pre_hook(external)
    for name,layer in study.clocks().layers(model):
        values=torch.from_numpy(table[name].copy());constant=float(pooled[name])
        def forward(self,x,t=None,values=values,constant=constant):
            assert t is None and self.orders==1
            if mode=='time':
                q=context['time'].to(x.dtype)*128;lower=torch.floor(q).long();upper=torch.minimum(lower+1,lower.new_full(lower.shape,128))
                amount=q-lower.to(q.dtype);v=values.to(x.dtype)
                s=v[lower]+(v[upper]-v[lower])*amount
            else:s=x.new_full((len(x),),constant)
            return x+(F.linear(x,self.proj.weight)*s[:,None,None])/1
        layer.forward=types.MethodType(forward,layer)
    return model
def manual(model,source):
    output=[]
    with torch.no_grad():
        for i in range(0,len(source),64):
            x=torch.from_numpy(np.array(source[i:i+64]));labels=torch.zeros(len(x),dtype=torch.long)
            for j in range(128):
                t=torch.full((len(x),),j/128);next_t=torch.full_like(t,(j+1)/128)
                a=(model(x,t,labels)-x)/(1-t[:,None,None,None]).clamp_min(.05)
                trial=x+a/128
                b=(model(trial,next_t,labels)-trial)/(1-next_t[:,None,None,None]).clamp_min(.05)
                x=x+(a+b)/256
            output.append(x.numpy())
    return np.concatenate(output)
def sampling(root):
    check=verify_calibration(root);rows=[]
    for arm in study.ARMS:
        model=patched(root,arm.removeprefix('ode_k1_'));before=study.clocks().tensors(model);b=study.bank();folder=root/'output/runs'/arm
        prediction=study.old().predict(model,b)
        with np.load(folder/'validation.npz') as f:same(prediction,f['prediction'])
        delta=b['delta'].astype(np.float64)[:,None,None,None]
        predicted_v=(prediction.astype(np.float64)-b['x'].astype(np.float64))/delta
        target_v=(b['target'].astype(np.float64)-b['x'].astype(np.float64))/delta
        weighted=float(np.mean((predicted_v-target_v)**2));clean=float(np.mean((prediction.astype(np.float64)-b['target'].astype(np.float64))**2))
        actual=study.load(folder/'result.json');np.testing.assert_allclose([weighted,clean],[actual['weighted_mse'],actual['clean_mse']],rtol=1e-10,atol=1e-12)
        for t in (.1,.3,.5,.7,.9):
            value=study.old().predict(model,study.bank(t))
            with np.load(folder/f'panel-{t:.1f}.npz') as f:same(value,f['prediction'])
        generated=manual(model,np.load(study.PARENTS['native']/'banks/sampling-source.npy'))
        same(generated,np.load(folder/'generated.npy'));study.clocks().unchanged(model,before)
        rows.append(dict(family=arm,weighted_mse=weighted,clean_mse=clean,predictions_bitwise=1024,generated_images_bitwise=1024,
            time_panels_bitwise=5,parameters_buffers_unchanged=True))
        print(study.json.dumps(rows[-1]),flush=True)
    return dict(rows=rows,calibration=check,new_generated_images_bitwise=2048,new_time_panels_bitwise=10)
def fid(a,b):
    import scipy.linalg
    a=a.astype(np.float64);b=b.astype(np.float64);x=a-a.mean(0);y=b-b.mean(0)
    singular=scipy.linalg.svd(x@y.T,compute_uv=False,lapack_driver='gesvd')
    return float(np.sum((a.mean(0)-b.mean(0))**2)+np.sum(x*x)/(len(a)-1)+np.sum(y*y)/(len(b)-1)-2*singular.sum()/np.sqrt((len(a)-1)*(len(b)-1)))
def features(root):
    sys.path.insert(0,str(study.PARENTS['native']/'source'));import image_metrics as m
    from metric_core import uint8_images
    model=m.extractor(study.PARENTS['native']);reference=m.features(model,np.load(study.PARENTS['native']/'banks/fid-reference.uint8.npy'))
    same(reference,np.load(study.PARENTS['native']/'metric-inputs/reference-features.npy'));rows=[]
    for row in study.load(root/'metric-output/result.json')['rows']:
        name=row['family'];saved=np.load(study.feature_bank(root,name))
        if name in study.ARMS:same(m.features(model,uint8_images(np.load(study.sample_bank(root,name)))),saved)
        value=fid(saved,reference);assert abs(value-row['fid'])<1e-8
        rows.append(dict(family=name,fid=value,difference=value-row['fid'],new_features_bitwise=name in study.ARMS))
        print(study.json.dumps(rows[-1]),flush=True)
    assert len(rows)==11
    return dict(rows=rows,new_generated_features_bitwise=2048,reference_features_bitwise=2048)
def main(root,phase):
    torch.set_num_threads(1);study.verify(root,True);out=root/f'audit-{phase}';out.mkdir()
    before={str(p):study.sha(p) for name in ('output','metric-output') for p in (root/name).rglob('*') if p.is_file()};tick=time.monotonic()
    result=sampling(root) if phase=='sampling' else features(root)
    for p,digest in before.items():assert study.sha(p)==digest
    study.verify(root,True);study.write(out/'result.json',dict(status='PASS',phase=phase,**result,formal_files_unchanged=len(before),seconds=time.monotonic()-tick))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=('sampling','features'),required=True);a=p.parse_args();main(a.root,a.phase)
