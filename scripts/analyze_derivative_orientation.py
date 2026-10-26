"""Audit directional cold training and both fixed-weight output orientations."""
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
from synthetic.conservative_readout import build as parent_build,compute_budget
from synthetic.directional_readout import build
from synthetic.population import fingerprint_batch
from synthetic.train import restore_model,load_targets,generated_samples
from scripts.analyze_conservative_readout import sha,load,write,table,assert_same,field,derivative,chunked_derivative

TIMES=(.1,.3,.5,.7,.9)
FAMILIES=('ode_k2','swiglu128','swiglu192')


class ExplicitOrientation(nn.Module):
    def __init__(self,raw,orientation):
        super().__init__();self.raw=raw;self.orientation=orientation

    def forward(self,x,t):
        keep=torch.is_grad_enabled()
        with torch.enable_grad():
            xx=x if x.requires_grad else x.detach().requires_grad_(True)
            g=self.raw(xx,t)
            rows=[torch.autograd.grad(g[:,j].sum(),xx,create_graph=keep,
                retain_graph=keep or j+1<g.shape[1])[0] for j in range(g.shape[1])]
            jac=torch.stack(rows,dim=1)
            product=torch.einsum('bij,bi->bj',jac,xx) if self.orientation=='gradient' else torch.einsum('bij,bj->bi',jac,xx)
            result=.5*(g+product)
        return result if keep else result.detach()


def models(state,dim,family):
    a=parent_build(dim,family,'gradient').double().eval()
    b=build(dim,family,'directional').double().eval()
    direct=parent_build(dim,family,'direct').double().eval()
    for model in (a,b,direct):
        restore_model(model,state)
        for p in model.parameters():p.requires_grad_(False)
    return a,b,direct


def raw_values(raw,x,t):
    xx=x.detach().clone().requires_grad_(True)
    value=raw(xx,t)
    jac=torch.stack([torch.autograd.grad(value[:,j].sum(),xx,
        retain_graph=j+1<value.shape[1])[0] for j in range(value.shape[1])],dim=1)
    return value.detach(),jac.detach()


def analyze(raw,parent,parent_results,output):
    torch.set_num_threads(1)
    status=load(raw/'status.json');assert status['state']=='COMPLETE' and status['completed']==status['total']==9
    source=Path(load(raw/'provenance.json')['source_root'])
    manifest=load(source/'source-sha256.json')
    for name,value in manifest.items():assert sha(source/name)==value
    for name in ('synthetic/models.py','synthetic/recurrent_depth.py','synthetic/conservative_readout.py',
                 'synthetic/directional_readout.py','synthetic/differentiable_layernorm.py','synthetic/train.py'):
        assert sha(ROOT/name)==manifest[name]
    old_manifest=load(parent/'source-v2/source-sha256.json')
    for name,value in old_manifest.items():assert sha(parent/'source-v2'/name)==value
    report_manifest=load(parent_results/'report-file-sha256.json')
    for name in ('comparison.csv','audit.json','input-bindings.json','p-trajectories.csv'):
        assert sha(parent_results/name)==report_manifest['results/'+name]
    old_audit=load(parent_results/'audit.json');assert old_audit['status']=='PASS' and old_audit['checkpoints']==84
    for name,value in load(parent_results/'input-bindings.json').items():assert sha(Path(name))==value
    with (parent_results/'comparison.csv').open() as f:metrics=list(csv.DictReader(f))
    for r in metrics:
        for key in ('oracle_mse','jacobian_error','sw2'):r[key]=float(r[key])
        for key in ('parameters','dense_forward_macs','dense_training_forward_backward_macs'):r[key]=int(r[key])
    output.mkdir(parents=True,exist_ok=False);(output/'crossovers').mkdir()
    config=load(raw/'study.json')
    expected_arms={family+'_directional':dict(family=family,readout='directional') for family in FAMILIES}
    assert config['seed']==0 and config['steps']==5000 and config['arms']==expected_arms
    bindings={};p_rows=[];cross_rows=[];switch_rows=[];fd_rows=[];geometry=[]
    checkpoints=matrix_count=native_panels=crossover_panels=0
    max_value_error=max_jac_error=max_orientation_error=0.
    for dataset in config['datasets']:
        bank=raw/'banks'/dataset
        for path in bank.glob('*.npz'):
            bindings[str(path)]=sha(path)
            with np.load(path) as a,np.load(parent/'output/banks'/dataset/path.name) as b:
                assert a.files==b.files
                for key in a.files:np.testing.assert_array_equal(a[key],b[key])
        with np.load(bank/'field.npz') as data:x,t,truth=[data[k] for k in ('x','t','oracle')]
        panels=[]
        for clock in TIMES:
            with np.load(bank/f'panel_{clock:.1f}.npz') as data:panels.append({k:data[k] for k in data.files})
        target_root=Path(config['data_roots'][dataset]);targets=torch.from_numpy(load_targets(target_root/dataset/'train.npz',x.shape[1]))
        rng=torch.Generator().manual_seed(100000);digest=hashlib.sha256(b'').hexdigest()
        expected={0:(rng.get_state().clone(),digest)}
        for step in range(1,5001):
            target=targets[torch.randint(len(targets),(512,),generator=rng)]
            noise=torch.randn(target.shape,generator=rng);clock=torch.rand((512,1),generator=rng)
            digest=fingerprint_batch(digest,target,noise,clock)
            if step in (1000,2500,5000):expected[step]=(rng.get_state().clone(),digest)
        for family in FAMILIES:
            folder=raw/'runs'/dataset/(family+'_directional')
            assert load(folder/'status.json')['state']=='COMPLETE'
            row=load(folder/'result.json')
            for step in (0,1000,2500,5000):
                path=folder/f'step_{step:07d}.pt';bindings[str(path)]=sha(path)
                state=torch.load(path,map_location='cpu',weights_only=False)
                assert torch.equal(state['data_rng'],expected[step][0]) and state['input_stream_digest']==expected[step][1]
                if step==0:
                    old=torch.load(parent/'output/runs'/dataset/(family+'_gradient')/path.name,map_location='cpu',weights_only=False)
                    for key in ('model','runtime_buffers','data_rng','torch_rng'):assert_same(state[key],old[key])
                for name,record in load(folder/f'spectrum_{step:07d}.json').items():
                    p=state['model']['raw.'+name+'.proj.weight'].double()
                    sv=torch.linalg.svdvals(p);ev=torch.linalg.eigvals(p)
                    np.testing.assert_allclose(sv.numpy(),record['singular_values'],rtol=1e-11,atol=1e-12)
                    np.testing.assert_allclose(ev.real.numpy(),record['eigenvalue_real'],rtol=1e-10,atol=1e-11)
                    np.testing.assert_allclose(ev.imag.numpy(),record['eigenvalue_imag'],rtol=1e-10,atol=1e-11)
                    p_rows.append(dict(dataset=dataset,arm=family+'_directional',step=step,layer=name,
                        sigma_max=float(sv.max()),frobenius=float(p.norm())))
                    matrix_count+=1
                checkpoints+=1
            native=build(x.shape[1],family,'directional').eval();restore_model(native,state)
            for key,value in compute_budget(native).items():
                if key!='matrix_count_scope':assert row[key]==value
            prediction=field(native,x,t,torch.float32)
            path=folder/'diagnostics/field.npz';bindings[str(path)]=sha(path)
            with np.load(path) as data:np.testing.assert_array_equal(prediction,data['prediction'])
            assert float(np.mean((prediction-truth)**2))==row['oracle_mse']
            for clock,panel in zip(TIMES,panels):
                value,jac=derivative(native,torch.from_numpy(panel['x']),torch.from_numpy(panel['t']))
                path=folder/'diagnostics'/f'panel_{clock:.1f}.npz';bindings[str(path)]=sha(path)
                with np.load(path) as data:
                    np.testing.assert_array_equal(value,data['prediction']);np.testing.assert_array_equal(jac,data['jacobian'])
                native_panels+=1
            with np.load(bank/'sampling.npz') as data:starts,target,dirs=[data[k] for k in ('source','target','directions')]
            samples=generated_samples(native,torch.from_numpy(starts),32).numpy()
            path=folder/'diagnostics/samples.npz';bindings[str(path)]=sha(path)
            with np.load(path) as data:np.testing.assert_array_equal(samples,data['generated'])
            sw=float(np.sqrt(np.mean((np.sort(samples.astype(np.float64)@dirs.astype(np.float64).T,axis=0)
                -np.sort(target.astype(np.float64)@dirs.astype(np.float64).T,axis=0))**2)))
            np.testing.assert_allclose(sw,row['sw2'],rtol=1e-6,atol=1e-7)
            metrics.append({k:row[k] for k in metrics[0]})
            for trained,checkpoint_path in (
                ('gradient',parent/'output/runs'/dataset/(family+'_gradient')/'step_0005000.pt'),
                ('directional',folder/'step_0005000.pt')):
                bindings[str(checkpoint_path)]=sha(checkpoint_path)
                saved=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
                a,b,g=models(saved,x.shape[1],family)
                if trained=='gradient':
                    native_a=parent_build(x.shape[1],family,'gradient').eval();restore_model(native_a,saved)
                    value=field(native_a,x,t,torch.float32)
                    old_folder=checkpoint_path.parent/'diagnostics'
                    with np.load(old_folder/'field.npz') as data:np.testing.assert_array_equal(value,data['prediction'])
                    for clock,panel in zip(TIMES,panels):
                        pv,pj=derivative(native_a,torch.from_numpy(panel['x']),torch.from_numpy(panel['t']))
                        with np.load(old_folder/f'panel_{clock:.1f}.npz') as data:
                            np.testing.assert_array_equal(pv,data['prediction']);np.testing.assert_array_equal(pj,data['jacobian'])
                        native_panels+=1
                out=output/'crossovers'/dataset/family/trained;out.mkdir(parents=True)
                av,bv,identity=[],[],[]
                for start in range(0,len(x),1024):
                    xx=torch.from_numpy(x[start:start+1024]).double();tt=torch.from_numpy(t[start:start+1024]).double()
                    gv,gj=raw_values(g.raw,xx,tt)
                    expected_a=.5*(gv+torch.einsum('bij,bi->bj',gj,xx))
                    expected_b=.5*(gv+torch.einsum('bij,bj->bi',gj,xx))
                    action=.5*torch.einsum('bij,bj->bi',gj.transpose(1,2)-gj,xx)
                    with torch.no_grad():va,vb=a(xx,tt),b(xx,tt)
                    torch.testing.assert_close(va,expected_a,rtol=1e-9,atol=1e-10)
                    torch.testing.assert_close(vb,expected_b,rtol=1e-9,atol=1e-10)
                    torch.testing.assert_close(va-vb,action,rtol=1e-9,atol=1e-10)
                    max_value_error=max(max_value_error,float((va-expected_a).abs().max()),float((vb-expected_b).abs().max()))
                    max_orientation_error=max(max_orientation_error,float((va-vb-action).abs().max()))
                    av.append(va.numpy());bv.append(vb.numpy());identity.append(action.numpy())
                av,bv,identity=np.concatenate(av),np.concatenate(bv),np.concatenate(identity)
                errors_a=np.mean((av-truth)**2,axis=1);errors_b=np.mean((bv-truth)**2,axis=1)
                energy=np.mean((bv-av)**2,axis=1)
                alignment_a=2*np.mean((av-truth)*(bv-av),axis=1)
                alignment_b=2*np.mean((bv-truth)*(av-bv),axis=1)
                np.testing.assert_allclose(errors_b-errors_a,alignment_a+energy,rtol=1e-9,atol=1e-11)
                np.testing.assert_allclose(errors_a-errors_b,alignment_b+energy,rtol=1e-9,atol=1e-11)
                np.savez_compressed(out/'fields.npz',prediction_A=av,prediction_B=bv,raw_skew_action=identity,
                    error_A=errors_a,error_B=errors_b,change_energy=energy,alignment_A_to_B=alignment_a,alignment_B_to_A=alignment_b)
                switch_rows.append(dict(dataset=dataset,family=family,trained=trained,
                    gradient_mse=float(errors_a.mean()),directional_mse=float(errors_b.mean()),
                    alignment_A_to_B=float(alignment_a.mean()),alignment_B_to_A=float(alignment_b.mean()),change_energy=float(energy.mean())))
                for evaluated,model,values in (('gradient',a,av),('directional',b,bv)):
                    explicit=ExplicitOrientation(g.raw,evaluated)
                    jac_errors=[]
                    for clock,panel in zip(TIMES,panels):
                        px,pt=torch.from_numpy(panel['x']).double(),torch.from_numpy(panel['t']).double()
                        pv,pj=derivative(model,px,pt)
                        qv,qj=chunked_derivative(explicit,px,pt)
                        np.testing.assert_allclose(pv,qv,rtol=1e-9,atol=1e-10)
                        np.testing.assert_allclose(pj,qj,rtol=1e-8,atol=1e-9)
                        max_jac_error=max(max_jac_error,float(np.max(np.abs(pj-qj))))
                        np.savez_compressed(out/f'{evaluated}-panel-{clock:.1f}.npz',prediction=pv,jacobian=pj)
                        skew=.5*(pj-pj.transpose(0,2,1));sym=.5*(pj+pj.transpose(0,2,1))
                        je=float(np.mean(np.sum((pj-panel['oracle_jacobian'])**2,axis=(1,2))));jac_errors.append(je)
                        geometry.append(dict(dataset=dataset,family=family,trained=trained,evaluated=evaluated,time=clock,
                            symmetric_error=float(np.mean(np.sum((sym-panel['oracle_jacobian'])**2,axis=(1,2)))),
                            skew_error=float(np.mean(np.sum(skew**2,axis=(1,2)))),jacobian_energy=float(np.mean(np.sum(pj**2,axis=(1,2))))))
                        crossover_panels+=1
                        if trained=='directional' and evaluated=='directional':
                            direction=torch.linspace(-1.,1.,x.shape[1],dtype=torch.float64).reshape(1,-1);direction/=direction.norm()
                            with torch.no_grad():finite=(model(px[:1]+1e-5*direction,pt[:1])-model(px[:1]-1e-5*direction,pt[:1]))/2e-5
                            analytic=torch.einsum('bij,bj->bi',torch.from_numpy(pj[:1]),direction)
                            error=float((finite-analytic).norm()/max(1.,float(analytic.norm())))
                            assert error<1e-5,(dataset,family,clock,error)
                            fd_rows.append(dict(dataset=dataset,family=family,time=clock,relative_error=error))
                    cross_rows.append(dict(dataset=dataset,family=family,trained=trained,evaluated=evaluated,
                        field_mse=float(np.mean((values-truth)**2)),jacobian_error=float(np.mean(jac_errors))))
            print(json.dumps(dict(completed_backbones=len(switch_rows)//2,total=9,dataset=dataset,family=family)),flush=True)
    assert checkpoints==36 and matrix_count==72 and native_panels==90 and crossover_panels==180 and len(metrics)==30 and len(fd_rows)==45
    comparisons={}
    for dataset in config['datasets']:
        rows=[r for r in metrics if r['dataset']==dataset]
        new=next(r for r in rows if r['arm']=='ode_k2_directional')
        direct=next(r for r in rows if r['arm']=='ode_k2_direct')
        generic=[r for r in rows if r['family'].startswith('swiglu')]
        comparisons[dataset]=dict(primary_ratio=new['oracle_mse']/min(r['oracle_mse'] for r in generic),
            parent_ratio=new['oracle_mse']/direct['oracle_mse'],jacobian_ratio=new['jacobian_error']/min(r['jacobian_error'] for r in generic),
            sw2_ratio=new['sw2']/min(r['sw2'] for r in generic))
    nonlinear=('gmm_ring8_2d','rotated_product_bimodal_16d')
    candidate=all(r['primary_ratio']<=1.05 for r in comparisons.values()) and any(
        comparisons[d]['primary_ratio']<=.9 and comparisons[d]['parent_ratio']<=.95 for d in nonlinear) and all(
        comparisons[d]['jacobian_ratio']<=1.1 and comparisons[d]['sw2_ratio']<=1.1 for d in nonlinear)
    table(output/'comparison.csv',metrics);table(output/'crossovers.csv',cross_rows)
    table(output/'switch-risk.csv',switch_rows);table(output/'symmetric-skew.csv',geometry)
    table(output/'new-p-trajectories.csv',p_rows);table(output/'directional-finite-differences.csv',fd_rows)
    write(output/'decision.json',dict(candidate=bool(candidate),comparisons=comparisons,
        action='separately register continuation/fresh validation' if candidate else 'stop radial directional readout; no rescue or tuning'))
    for name,value in bindings.items():assert sha(Path(name))==value
    for name,value in manifest.items():assert sha(source/name)==value
    monitor=[json.loads(s) for s in (raw/'monitor.jsonl').read_text().splitlines()]
    workers=[r for s in monitor for r in s['processes'] if 'cell' in r['command']]
    write(output/'audit.json',dict(status='PASS',new_checkpoints=checkpoints,new_P_matrices=matrix_count,
        native_saved_field_replays=18,native_saved_panel_replays=native_panels,crossover_fields=36,crossover_full_panels=crossover_panels,
        true_regenerated_streams=3,new_trained_finite_differences=len(fd_rows),maximum_fd_error=max(r['relative_error'] for r in fd_rows),
        reused_controls=21,reused_parent_input_bindings_verified=len(load(parent_results/'input-bindings.json')),
        maximum_value_formula_error=max_value_error,maximum_input_J_formula_error=max_jac_error,
        maximum_orientation_identity_error=max_orientation_error,monitor_samples=len(monitor),
        minimum_available_GiB=min(r['available_memory_bytes'] for r in monitor)/2**30,
        peak_worker_RSS_MiB=max(r['rss_bytes'] for r in workers)/2**20,
        source_manifest_sha256=sha(source/'source-sha256.json'),analyzer_sha256=sha(__file__),
        helper_analyzer_sha256=sha(ROOT/'scripts/analyze_conservative_readout.py')))
    write(output/'input-bindings.json',bindings)
    tasks=config['datasets'];titles=('Gaussian2D','Ring8','Product16')
    arms=list(dict.fromkeys(r['arm'] for r in metrics))
    fig,axes=plt.subplots(1,3,figsize=(12,5.6))
    for ax,task,title in zip(axes,tasks,titles):
        rows=[next(r for r in metrics if r['dataset']==task and r['arm']==arm) for arm in arms]
        ax.scatter([r['oracle_mse'] for r in rows],range(10),c=['#D55E00' if r['family']=='ode_k2' else '#0072B2' for r in rows])
        ax.set_yticks(range(10),[r['arm'].replace('_',' ') for r in rows] if ax is axes[0] else [])
        ax.set(xscale='log',ylim=(9.7,-.7),title=title,xlabel='Full validation oracle MSE');ax.margins(x=.4)
    fig.tight_layout()
    fig.savefig(output/'field-comparison.png',dpi=160,bbox_inches='tight');fig.savefig(output/'field-comparison.pdf',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,4.3))
    for ax,task,title in zip(axes,tasks,titles):
        values=[];labels=[]
        for family in FAMILIES:
            for trained in ('gradient','directional'):
                r=next(r for r in switch_rows if r['dataset']==task and r['family']==family and r['trained']==trained)
                ratio=r['directional_mse']/r['gradient_mse'] if trained=='gradient' else r['gradient_mse']/r['directional_mse']
                values.append(ratio);labels.append(family+' trained '+trained)
        ax.barh(range(6),np.log10(values),color=['#D55E00' if v>1 else '#0072B2' for v in values])
        ax.set_yticks(range(6),labels if ax is axes[0] else []);ax.invert_yaxis()
        ax.set(title=title,xlabel='log10(switched/native field MSE)');ax.axvline(0,color='gray',lw=.7)
    fig.tight_layout();fig.savefig(output/'crossover-errors.png',dpi=160,bbox_inches='tight')
    fig.savefig(output/'crossover-errors.pdf',bbox_inches='tight');plt.close(fig)
    print(json.dumps(dict(audit='PASS',candidate=bool(candidate),models=30)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True);p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--parent-results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.raw.resolve(),a.parent.resolve(),a.parent_results.resolve(),a.output.resolve())
