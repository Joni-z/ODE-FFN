"""Native input replay, dense-reflection reference and independent scalar fits."""
from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import psutil
import torch
from torch.nn import functional as F

parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
root=args.root.resolve();out=args.output.resolve();out.mkdir(exist_ok=False)
sys.path.insert(0,str(root/'source'));import study as formal
started=time.monotonic();formal.verify_inputs(root)
settings,_,_,old,_,native,_=formal.setup(root)
original={str(p):native.sha(p) for p in (root/'output').rglob('*') if p.is_file()}
assert formal.load(root/'output/status.json')['state']=='COMPLETE'
def read(name):
    with (root/'output'/name).open() as stream: return list(csv.DictReader(stream))
rows=read('local.csv');complete=read('complete.csv');spectra=read('P-history.csv')
assert (len(rows),len(complete),len(spectra))==(720,24,144)
checks=[];peak=0;counts=dict(checkpoints=0,layers=0,scalar_source_rows=0,dense_frame_source_rows=0)


def close(actual,text):
    if actual is None: assert text==''
    elif isinstance(actual,(bool,np.bool_)): assert str(bool(actual))==text
    else: np.testing.assert_allclose(actual,float(text),atol=1e-10,rtol=1e-8)


@torch.no_grad()
def independent_samples(layer,x):
    # Controls are explicitly reconstructed, not called through formal.local/controls.
    layer=copy.deepcopy(layer).double();tx=torch.tensor(x,dtype=torch.float64)
    hidden=F.silu(F.linear(tx,layer.t_func[0].weight,layer.t_func[0].bias))
    logits=F.linear(hidden,layer.t_func[2].weight,layer.t_func[2].bias)
    s=(.2+.8*torch.sigmoid(logits/10)).numpy()
    if hasattr(layer,'frame_mode'):
        feature=hidden if layer.frame_mode=='state' else torch.ones_like(hidden)
        g=F.linear(feature,layer.frame_weight,layer.frame_bias).numpy()
    else: g=None
    p=layer.proj.weight.numpy();all_samples=[];outputs=[];identity=0.
    for start in range(0,len(x),256):
        a=x[start:start+256].astype(float);clock=s[start:start+256]
        if g is None: q=np.broadcast_to(np.eye(64),(len(a),64,64))
        else:
            u=np.column_stack((np.ones(len(a)),g[start:start+256]))
            q=np.eye(64)[None]-2*u[:,:,None]*u[:,None,:]/np.sum(u*u,axis=1)[:,None,None]
            q[:,:,0]*=-1;counts['dense_frame_source_rows']+=len(a)
        z=np.einsum('nji,nj->ni',q,a)
        first=z@p.T;second=first@p.T
        d=clock*first+clock*clock*second/2
        centre=z@(p+.3*p@p).T
        square=lambda v:np.sum(v*v,axis=1)
        def fit(v):
            numerator=np.sum(v*d,axis=1);denominator=square(v)
            free=np.zeros(len(a));np.divide(numerator,denominator,out=free,where=denominator!=0)
            box=np.minimum(1.,np.maximum(.2,free))
            return free,box,square(d-free[:,None]*v),square(d-box[:,None]*v)
        r,b,re,be=fit(first);cr,cb,ce,cbe=fit(centre)
        first_norm=square(first);coefficient=np.zeros(len(a))
        np.divide(np.sum(first*second,axis=1),first_norm,out=coefficient,where=first_norm!=0)
        orthogonal=second-coefficient[:,None]*first
        same=d-clock*centre;predicted=clock*(clock-.6)*second/2
        identity=max(identity,float(np.max(np.abs(same-predicted))))
        np.testing.assert_allclose(same,predicted,atol=1e-10,rtol=1e-10)
        all_samples.append(np.column_stack((square(d),square(clock*clock*second/2),square(second),
            square(orthogonal),square(same),re,be,ce,cbe,r,b,cr,cb,clock[:,0])))
        outputs.append(a+np.einsum('nij,nj->ni',q,d))
    return np.concatenate(all_samples),np.concatenate(outputs),identity


def summary(values):
    e=values[:,0].mean();r=dict(count=len(values),increment_rms=float(np.sqrt(e/64)),
        clock_mean=float(values[:,13].mean()),clock_std=float(values[:,13].std()))
    for name,index in [('quadratic',1),('sameclock_error',4),('same_free_error',5),
                        ('same_box_error',6),('center_free_error',7),('center_box_error',8)]:
        r[name+'_relative_rms']=float(np.sqrt(values[:,index].mean()/e)) if e>0 else None
    total=values[:,2].mean();r['second_action_orthogonal_fraction']=float(values[:,3].mean()/total) if total>0 else None
    for prefix,index in [('same',9),('center',11)]:
        v=values[:,index]
        r[prefix+'_clipped_fraction']=float(np.mean((v<.2)|(v>1.)))
        r[prefix+'_free_clock_min']=float(v.min());r[prefix+'_free_clock_max']=float(v.max())
    v=r['center_box_error_relative_rms'];r['center_box_below_one_percent']=v<=.01 if v is not None else None
    return r


@torch.no_grad()
def changed_hook(layer,args,output):
    # Literal native arithmetic, sharing only the established rotation primitive.
    x=args[0]
    if hasattr(layer,'frame_mode'):
        clock,g,_=layer.controls(x);rotate=sys.modules[type(layer).__module__].rotate
        z=rotate(x,g,transpose=True)
    else: clock=layer.duration_from_logit(layer.t_func(x));z=x;g=None
    a=F.linear(z,layer.proj.weight);b=F.linear(a,layer.proj.weight)
    response=clock*(a+.3*b)
    return x+(rotate(response,g) if g is not None else response)


with torch.no_grad():
    for entry in old['laws']:
        law=entry['law'];banks=Path(entry['banks'])
        with np.load(banks/'field.npz') as data: field={k:data[k] for k in ('x','t','oracle')}
        for family in formal.FAMILIES:
            for step in formal.CHECKPOINTS:
                model,folder=formal.restore(root,law,family,step);layers=list(model.ode_layers())
                inputs={name:[] for name,_ in layers};outputs={name:[] for name,_ in layers};handles=[]
                for name,layer in layers:
                    def capture(module,args,output,key=name):
                        inputs[key].append(args[0].numpy().copy());outputs[key].append(output.numpy().copy())
                    handles.append(layer.register_forward_hook(capture))
                prediction=np.concatenate([model(torch.from_numpy(field['x'][i:i+1024]),
                    torch.from_numpy(field['t'][i:i+1024])).numpy() for i in range(0,20000,1024)])
                for h in handles: h.remove()
                with np.load(folder/f'field_{step:07d}.npz') as data: np.testing.assert_array_equal(prediction,data['prediction'])
                maximum=0.;identity_max=0.
                with np.load(root/'output/raw'/f'{law}-{family}-{step:07d}.npz') as raw:
                    for index,(name,layer) in enumerate(layers):
                        x=np.concatenate(inputs[name]);np.testing.assert_array_equal(x,raw[f'input_{index}'])
                        values,reference,identity=independent_samples(layer,x);stored=raw[f'samples_{index}']
                        np.testing.assert_allclose(values,stored,atol=1e-10,rtol=1e-8)
                        maximum=max(maximum,float(np.max(np.abs(values-stored))));identity_max=max(identity_max,identity)
                        native_output=np.concatenate(outputs[name]);np.testing.assert_allclose(reference,native_output,atol=1e-5,rtol=1e-5)
                        difference=float(np.max(np.abs(reference-native_output)))
                        selected=[r for r in rows if r['law']==law and r['family']==family and int(r['step'])==step and r['layer']==name]
                        assert len(selected)==5
                        for row in selected:
                            if row['time_bin']=='pooled': mask=np.ones(20000,dtype=bool)
                            else:
                                left,right=map(float,row['time_bin'].split('-'))
                                mask=(field['t'][:,0]>=left)&(field['t'][:,0]<right)
                            for k,v in summary(values[mask]).items(): close(v,row[k])
                            close(difference,row['promoted_vs_native_output_max'])
                        singular=np.linalg.svd(layer.proj.weight.double().numpy(),compute_uv=False)
                        row=next(r for r in spectra if r['law']==law and r['family']==family and int(r['step'])==step and r['layer']==name)
                        for k,v in dict(sigma_max=singular[0],sigma_min=singular[-1],singular_cv=singular.std()/singular.mean()).items(): close(v,row[k])
                        counts['layers']+=1;counts['scalar_source_rows']+=len(values)
                handles=[layer.register_forward_hook(changed_hook) for _,layer in layers]
                changed=np.concatenate([model(torch.from_numpy(field['x'][i:i+1024]),
                    torch.from_numpy(field['t'][i:i+1024])).numpy() for i in range(0,20000,1024)])
                for h in handles: h.remove()
                with np.load(root/'output/raw'/f'{law}-{family}-{step:07d}-predictions.npz') as raw:
                    np.testing.assert_array_equal(prediction,raw['native']);np.testing.assert_array_equal(changed,raw['changed'])
                delta=changed.astype(float)-prediction;error=prediction.astype(float)-field['oracle']
                original_mse=float(np.mean(error**2));changed_mse=float(np.mean((error+delta)**2))
                energy=float(np.mean(delta**2));cross=float(2*np.mean(error*delta));base=float(np.mean(prediction.astype(float)**2))
                relative=float(np.sqrt(energy/base)) if base>0 else None
                row=next(r for r in complete if r['law']==law and r['family']==family and int(r['step'])==step)
                for k,v in dict(original_field_mse=original_mse,changed_field_mse=changed_mse,displacement_mse=energy,
                    cross_term=cross,relative_displacement_rms=relative,
                    displacement_below_one_percent=relative<=.01 if relative is not None else None,native_field_bitwise=True).items(): close(v,row[k])
                np.testing.assert_allclose(changed_mse-original_mse,energy+cross,atol=1e-10,rtol=1e-8)
                checks.append(dict(law=law,family=family,step=step,native_inputs_bitwise=True,native_fields_bitwise=True,
                    substituted_fields_bitwise=True,independent_sample_max=maximum,centered_identity_max=identity_max))
                counts['checkpoints']+=1;peak=max(peak,psutil.Process().memory_info().rss)
                assert time.monotonic()-started<600
                print(dict(checked=counts['checkpoints'],law=law,family=family,step=step,elapsed=time.monotonic()-started),flush=True)

assert counts==dict(checkpoints=24,layers=144,scalar_source_rows=2880000,dense_frame_source_rows=1920000)
formal.verify_inputs(root)
for name,digest in original.items(): assert native.sha(Path(name))==digest,name
formal.table(out/'independent-checks.csv',checks)
formal.write(out/'audit.json',dict(status='PASS',counts=counts,formal_files_unchanged=len(original),
    elapsed_seconds=time.monotonic()-started,peak_sampled_rss_gib=peak/2**30,new_training=False,
    scope='all native inputs/fields, explicit float64 controls/dense Q/scalar projections, literal native intervention'))
for name in ('local.csv','complete.csv','P-history.csv','telemetry.csv','receipt.json','sample-columns.json'):
    shutil.copyfile(root/'output'/name,out/name)
shutil.copyfile(root/'preflight.json',out/'preflight.json')
print(dict(status='AUDIT_PASS',counts=counts,elapsed=time.monotonic()-started),flush=True)
