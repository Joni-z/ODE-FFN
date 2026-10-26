"""Replay cold-trained readouts and audit their actual histories and costs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from synthetic.conservative_readout import build,compute_budget
from synthetic.population import fingerprint_batch
from synthetic.train import restore_model,load_targets,generated_samples
from scripts.run_conservative_readout import ARMS


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def table(path,rows):
    with Path(path).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def assert_same(a,b):
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:assert_same(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert type(a)==type(b) and len(a)==len(b)
        for left,right in zip(a,b):assert_same(left,right)
    else:assert a==b


class ExplicitReadout(nn.Module):
    """Use complete G Jacobian rows instead of differentiating x dot G."""
    def __init__(self,raw):
        super().__init__();self.raw=raw

    def forward(self,x,t):
        keep=torch.is_grad_enabled()
        with torch.enable_grad():
            xx=x if x.requires_grad else x.detach().requires_grad_(True)
            g=self.raw(xx,t)
            rows=[torch.autograd.grad(g[:,j].sum(),xx,create_graph=keep,
                retain_graph=keep or j+1<g.shape[1])[0] for j in range(g.shape[1])]
            derivative=torch.stack(rows,dim=1)
            value=.5*(g+torch.einsum('bij,bi->bj',derivative,xx))
        return value if keep else value.detach()


def field(model,x,t,dtype):
    result=[]
    with torch.no_grad():
        for start in range(0,len(x),1024):
            result.append(model(torch.from_numpy(x[start:start+1024]).to(dtype),
                torch.from_numpy(t[start:start+1024]).to(dtype)).numpy())
    return np.concatenate(result)


def derivative(model,x,t):
    xx=x.detach().clone().requires_grad_(True)
    v=model(xx,t)
    jac=torch.stack([torch.autograd.grad(v[:,j].sum(),xx,
        retain_graph=j+1<v.shape[1])[0] for j in range(v.shape[1])],dim=1)
    return v.detach().numpy(),jac.detach().numpy()


def chunked_derivative(model,x,t):
    values,jacobians=[],[]
    for start in range(0,len(x),64):
        value,jac=derivative(model,x[start:start+64],t[start:start+64])
        values.append(value);jacobians.append(jac)
    return np.concatenate(values),np.concatenate(jacobians)


def analyze(raw,parents,output):
    torch.set_num_threads(1)
    state=load(raw/'status.json')
    assert state['state']=='COMPLETE' and state['completed']==state['total']==21
    provenance=load(raw/'provenance.json')
    source=Path(provenance['source_root'])
    for name,value in provenance['source_sha256'].items():assert sha(source/name)==value
    for name in ('synthetic/models.py','synthetic/recurrent_depth.py','synthetic/conservative_readout.py','synthetic/train.py'):
        assert sha(ROOT/name)==provenance['source_sha256'][name]
    output.mkdir(parents=True,exist_ok=False)
    config=load(raw/'study.json')
    assert config['arms']==ARMS and config['steps']==5000 and config['seed']==0
    parent_by_task={d:p for p in parents for d in load(p/'study.json')['datasets']}
    metrics,curves,spectra,geometry=[],[],[],[]
    parent_states=checkpoints=panel_count=field_count=matrices=0
    max_independent_field=max_independent_jac=0.
    bindings={}
    for dataset in config['datasets']:
        banks=raw/'banks'/dataset
        parent=parent_by_task[dataset]
        for path in banks.glob('*.npz'):
            bindings[str(path)]=sha(path)
            with np.load(path) as a,np.load(parent/'banks'/dataset/path.name) as b:
                assert a.files==b.files
                for key in a.files:np.testing.assert_array_equal(a[key],b[key])
        with np.load(banks/'field.npz') as data:x,t,oracle,pair=[data[k] for k in ('x','t','oracle','pair')]
        root=Path(config['data_roots'][dataset])
        targets=torch.from_numpy(load_targets(root/dataset/'train.npz',x.shape[1]))
        rng=torch.Generator().manual_seed(100000)
        digest=hashlib.sha256(b'').hexdigest()
        expected={0:(rng.get_state().clone(),digest)}
        for step in range(1,5001):
            y=targets[torch.randint(len(targets),(512,),generator=rng)]
            z=torch.randn(y.shape,generator=rng)
            clock=torch.rand((512,1),generator=rng)
            digest=fingerprint_batch(digest,y,z,clock)
            if step in (1000,2500,5000):expected[step]=(rng.get_state().clone(),digest)
        for arm,setting in ARMS.items():
            folder=raw/'runs'/dataset/arm
            assert load(folder/'status.json')['state']=='COMPLETE'
            row=load(folder/'result.json')
            for step in (0,1000,2500,5000):
                path=folder/f'step_{step:07d}.pt';bindings[str(path)]=sha(path)
                snapshot=torch.load(path,map_location='cpu',weights_only=False)
                assert torch.equal(snapshot['data_rng'],expected[step][0])
                assert snapshot['input_stream_digest']==expected[step][1]
                if setting['readout']=='direct' and setting['family']!='swiglu384':
                    old_path=parent/'runs'/dataset/(setting['family']+'_native_final')/path.name
                    old=torch.load(old_path,map_location='cpu',weights_only=False)
                    for key in ('model','optimizer','runtime_buffers','data_rng','torch_rng','input_stream_digest'):
                        assert_same(snapshot[key],old[key])
                    bindings[str(old_path)]=sha(old_path)
                    parent_states+=1
                if setting['readout']=='gradient' and step==0:
                    direct=torch.load(raw/'runs'/dataset/(setting['family']+'_direct')/path.name,map_location='cpu',weights_only=False)
                    for key in ('model','runtime_buffers','data_rng','torch_rng'):assert_same(snapshot[key],direct[key])
                saved=load(folder/f'spectrum_{step:07d}.json')
                for name,values in saved.items():
                    matrix=snapshot['model']['raw.'+name+'.proj.weight'].double()
                    singular=torch.linalg.svdvals(matrix)
                    eigen=torch.linalg.eigvals(matrix)
                    np.testing.assert_allclose(singular.numpy(),values['singular_values'],rtol=1e-11,atol=1e-12)
                    np.testing.assert_allclose(eigen.real.numpy(),values['eigenvalue_real'],rtol=1e-10,atol=1e-11)
                    np.testing.assert_allclose(eigen.imag.numpy(),values['eigenvalue_imag'],rtol=1e-10,atol=1e-11)
                    spectra.append(dict(dataset=dataset,arm=arm,step=step,layer=name,
                        sigma_max=float(singular.max()),frobenius=float(matrix.norm())))
                    matrices+=1
                checkpoints+=1
            model=build(x.shape[1],setting['family'],setting['readout']).eval()
            restore_model(model,snapshot)
            counts=compute_budget(model)
            for key in ('parameters','dense_forward_macs','dense_training_forward_backward_macs'):assert row[key]==counts[key]
            model.eval()
            for p in model.parameters():p.requires_grad_(False)
            value=field(model,x,t,torch.float32)
            path=folder/'diagnostics/field.npz';bindings[str(path)]=sha(path)
            with np.load(path) as data:np.testing.assert_array_equal(value,data['prediction'])
            assert float(np.mean((value-oracle)**2))==row['oracle_mse']
            assert float(np.mean((value-pair)**2))==row['pair_mse']
            field_count+=1
            saved_panels=[]
            for clock in (.1,.3,.5,.7,.9):
                path=banks/f'panel_{clock:.1f}.npz'
                with np.load(path) as data:panel={k:data[k] for k in data.files}
                pv,pj=derivative(model,torch.from_numpy(panel['x']),torch.from_numpy(panel['t']))
                path=folder/'diagnostics'/f'panel_{clock:.1f}.npz';bindings[str(path)]=sha(path)
                with np.load(path) as data:
                    np.testing.assert_array_equal(pv,data['prediction'])
                    np.testing.assert_array_equal(pj,data['jacobian'])
                error=float(np.mean(np.sum((pj-panel['oracle_jacobian'])**2,axis=(1,2))))
                recorded=next(r for r in row['panels'] if r['time']==clock)
                np.testing.assert_allclose(error,recorded['jacobian_error'],rtol=1e-12,atol=1e-12)
                symmetric=.5*(pj+pj.transpose(0,2,1)).astype(np.float64)
                skew=.5*(pj-pj.transpose(0,2,1)).astype(np.float64)
                geometry.append(dict(dataset=dataset,arm=arm,time=clock,
                    symmetric_error=float(np.mean(np.sum((symmetric-panel['oracle_jacobian'])**2,axis=(1,2)))),
                    skew_error=float(np.mean(np.sum(skew**2,axis=(1,2))))))
                saved_panels.append(panel)
                panel_count+=1
            model=model.double()
            if setting['readout']=='gradient':
                independent=ExplicitReadout(model.raw)
                a=field(model,x,t,torch.float64)
                b=field(independent,x,t,torch.float64)
                np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-10)
                max_independent_field=max(max_independent_field,float(np.max(np.abs(a-b))))
                for panel in saved_panels:
                    px,pt=torch.from_numpy(panel['x']).double(),torch.from_numpy(panel['t']).double()
                    av,aj=derivative(model,px,pt)
                    bv,bj=chunked_derivative(independent,px,pt)
                    np.testing.assert_allclose(av,bv,rtol=1e-9,atol=1e-10)
                    np.testing.assert_allclose(aj,bj,rtol=1e-8,atol=1e-9)
                    asym=aj-aj.transpose(0,2,1)
                    relative=np.linalg.norm(asym,axis=(1,2))/np.maximum(1.,np.linalg.norm(aj,axis=(1,2)))
                    assert relative.max()<=1e-10
                    max_independent_jac=max(max_independent_jac,float(np.max(np.abs(aj-bj))))
            with np.load(banks/'sampling.npz') as data:
                starts,target,target2,directions=[data[k] for k in ('source','target','target2','directions')]
            model=model.float()
            samples=generated_samples(model,torch.from_numpy(starts),32).numpy()
            path=folder/'diagnostics/samples.npz';bindings[str(path)]=sha(path)
            with np.load(path) as data:np.testing.assert_array_equal(samples,data['generated'])
            proj=directions.astype(np.float64)
            sw2=float(np.sqrt(np.mean((np.sort(samples.astype(np.float64)@proj.T,axis=0)-np.sort(target.astype(np.float64)@proj.T,axis=0))**2)))
            np.testing.assert_allclose(sw2,row['sw2'],rtol=1e-6,atol=1e-7)
            metrics.append({k:row[k] for k in ('dataset','arm','family','readout','oracle_mse','jacobian_error','sw2','parameters','dense_forward_macs','dense_training_forward_backward_macs')})
            for line in (folder/'metrics.jsonl').read_text().splitlines():curves.append(dict(dataset=dataset,arm=arm,**json.loads(line)))
            print(json.dumps(dict(replayed=len(metrics),total=21,dataset=dataset,arm=arm)),flush=True)
    assert checkpoints==84 and parent_states==36 and panel_count==105 and field_count==21 and matrices==144
    comparisons={}
    for dataset in config['datasets']:
        rows=[r for r in metrics if r['dataset']==dataset]
        ode=next(r for r in rows if r['arm']=='ode_k2_gradient')
        parent=next(r for r in rows if r['arm']=='ode_k2_direct')
        generic=[r for r in rows if r['family'].startswith('swiglu')]
        comparisons[dataset]=dict(primary_ratio=ode['oracle_mse']/min(r['oracle_mse'] for r in generic),
            parent_ratio=ode['oracle_mse']/parent['oracle_mse'],
            jacobian_ratio=ode['jacobian_error']/min(r['jacobian_error'] for r in generic),
            sw2_ratio=ode['sw2']/min(r['sw2'] for r in generic))
    nonlinear=('gmm_ring8_2d','rotated_product_bimodal_16d')
    candidate=all(r['primary_ratio']<=1.05 for r in comparisons.values()) and any(
        comparisons[d]['primary_ratio']<=.9 and comparisons[d]['parent_ratio']<=.95 for d in nonlinear) and all(
        comparisons[d]['jacobian_ratio']<=1.1 and comparisons[d]['sw2_ratio']<=1.1 for d in nonlinear)
    table(output/'comparison.csv',metrics)
    table(output/'p-trajectories.csv',spectra)
    table(output/'symmetric-skew.csv',geometry)
    write(output/'training-curves.json',curves)
    write(output/'decision.json',dict(candidate=bool(candidate),comparisons=comparisons,
        action='separately register continuation and fresh validation' if candidate else 'stop this complete-field gradient readout; no tuning'))
    for path,value in bindings.items():assert sha(path)==value
    monitor=[json.loads(s) for s in (raw/'monitor.jsonl').read_text().splitlines()]
    workers=[r for sample in monitor for r in sample['processes'] if 'cell' in r['command']]
    write(output/'audit.json',dict(status='PASS',checkpoints=checkpoints,parent_exact_histories=parent_states,
        full_fields=field_count,full_panels_and_jacobians=panel_count,matrices=matrices,
        true_regenerated_streams=3,independent_gradient_fields=9,independent_gradient_panels=45,
        max_independent_prediction_error=max_independent_field,max_independent_jacobian_error=max_independent_jac,
        monitor_samples=len(monitor),minimum_available_GiB=min(r['available_memory_bytes'] for r in monitor)/2**30,
        maximum_worker_RSS_MiB=max(r['rss_bytes'] for r in workers)/2**20,
        source_manifest_sha256=sha(source/'source-sha256.json'),analyzer_sha256=sha(__file__)))
    write(output/'input-bindings.json',bindings)
    titles=('Gaussian2D','Ring8','Product16')
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,dataset,title in zip(axes,config['datasets'],titles):
        rows=[r for r in metrics if r['dataset']==dataset]
        ax.scatter([r['oracle_mse'] for r in rows],range(7),c=['#D55E00' if r['family']=='ode_k2' else '#0072B2' for r in rows])
        ax.set_yticks(range(7),[r['arm'].replace('_',' ') for r in rows] if ax is axes[0] else [])
        ax.set(xscale='log',ylim=(6.6,-.6),title=title,xlabel='Full validation oracle MSE')
        ax.margins(x=.4)
    fig.tight_layout()
    fig.savefig(output/'field-comparison.png',dpi=160,bbox_inches='tight')
    fig.savefig(output/'field-comparison.pdf',bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for ax,dataset,title in zip(axes,config['datasets'],titles):
        for arm in ARMS:
            rows=[r for r in curves if r['dataset']==dataset and r['arm']==arm and 'oracle_mse' in r]
            ax.plot([r['step'] for r in rows],[r['oracle_mse'] for r in rows],label=arm)
        ax.set(yscale='log',title=title,xlabel='Update',ylabel='Validation oracle MSE')
    axes[-1].legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=7)
    fig.tight_layout()
    fig.savefig(output/'learning-curves.png',dpi=160,bbox_inches='tight')
    fig.savefig(output/'learning-curves.pdf',bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(dict(audit='PASS',candidate=bool(candidate),models=len(metrics))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--parents',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    analyze(a.raw.resolve(),[v.resolve() for v in a.parents],a.output.resolve())
