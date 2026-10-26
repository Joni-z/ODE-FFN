"""Independent native replay and partial-Jacobian audit of gate-path cold runs."""
from __future__ import annotations
import argparse
import copy
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
from synthetic.gate_adjoint import build
from synthetic.models import OriginalODELayer,SwiGLUFFN
from synthetic.conservative_readout import compute_budget
from synthetic.population import fingerprint_batch
from synthetic.train import restore_model,load_targets,generated_samples

DATASET='rotated_product_bimodal_16d'
FAMILIES=('ode_k2','swiglu128','swiglu192')
TIMES=(.1,.3,.5,.7,.9)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def table(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def same(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:same(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert type(a)==type(b) and len(a)==len(b)
        for aa,bb in zip(a,b):same(aa,bb)
    else:assert a==b


class PartialJacobianReference(nn.Module):
    """Independent values from native G and AD with locally fixed gate values.

    This is only a value oracle. Its returned tensors are detached. Recomputing
    its values at perturbed inputs/parameters gives independent finite differences
    of the explicitly defined omission field, without a surrogate training rule.
    """
    def __init__(self,raw,omitted):
        super().__init__();self.raw=raw;self.omitted=omitted

    def forward(self,x,t):
        hooks=[]
        try:
            if self.omitted:
                for module in self.raw.modules():
                    if type(module) is OriginalODELayer:
                        hooks.append(module.t_func.register_forward_hook(lambda m,a,y:y.detach()))
                    elif type(module) is SwiGLUFFN:
                        def split(m,a,y):
                            gate,value=y.chunk(2,-1)
                            return torch.cat((gate.detach(),value),-1)
                        hooks.append(module.w12.register_forward_hook(split))
            with torch.enable_grad():
                xx=x.detach().requires_grad_(True)
                raw=self.raw(xx,t)
                dual=torch.autograd.grad(raw,xx,grad_outputs=xx)[0]
                result=.5*(raw.detach()+dual.detach())
            return result
        finally:
            for hook in hooks:hook.remove()


def field(model,x,t,dtype):
    out=[]
    with torch.no_grad():
        for i in range(0,len(x),1024):
            out.append(model(torch.from_numpy(x[i:i+1024]).to(dtype),torch.from_numpy(t[i:i+1024]).to(dtype)).numpy())
    return np.concatenate(out)


def derivative(model,x,t):
    xx=x.detach().clone().requires_grad_(True)
    value=model(xx,t)
    jac=torch.stack([torch.autograd.grad(value[:,j].sum(),xx,retain_graph=j+1<value.shape[1])[0]
                     for j in range(value.shape[1])],1)
    return value.detach(),jac.detach()


def main(runtime,parent,context_results,output):
    torch.set_num_threads(1)
    raw=runtime/'output';source=runtime/'source'
    terminal=load(raw/'status.json');assert terminal['state']=='COMPLETE' and terminal['completed']==terminal['total']==6
    assert load(runtime/'preflight/PASS.json')['status']=='PASS'
    manifest=load(source/'source-sha256.json')
    for name,value in manifest.items():assert sha(source/name)==value
    for name in ('synthetic/gate_adjoint.py','synthetic/explicit_adjoint.py','synthetic/models.py','scripts/run_conservative_readout.py'):
        assert sha(ROOT/name)==manifest[name]
    old_manifest=load(context_results/'report-file-sha256.json')
    for name,value in old_manifest.items():assert sha(context_results.parent/name)==value
    output.mkdir(parents=True,exist_ok=False)
    write(output/'status.json',dict(state='RUNNING',phase='independent_audit'))
    config=load(raw/'study.json');assert config['seed']==0 and config['steps']==5000 and config['datasets']==[DATASET]
    bank=raw/'banks'/DATASET
    bindings={str(raw/'study.json'):sha(raw/'study.json')}
    for path in bank.glob('*.npz'):
        bindings[str(path)]=sha(path)
        with np.load(path) as a,np.load(parent/'output/banks'/DATASET/path.name) as b:
            assert a.files==b.files
            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    with np.load(bank/'field.npz') as data:x,t,truth,target=[data[k] for k in ('x','t','oracle','pair')]
    root=Path(config['data_roots'][DATASET])
    targets=torch.from_numpy(load_targets(root/DATASET/'train.npz',16))
    rng=torch.Generator().manual_seed(100000);digest=hashlib.sha256(b'').hexdigest()
    expected={0:(rng.get_state().clone(),digest)}
    for step in range(1,5001):
        yy=targets[torch.randint(len(targets),(512,),generator=rng)]
        noise=torch.randn(yy.shape,generator=rng);clock=torch.rand((512,1),generator=rng)
        digest=fingerprint_batch(digest,yy,noise,clock)
        if step in (1000,2500,5000):expected[step]=(rng.get_state().clone(),digest)
    rows=[];geometry=[];p_rows=[];curves=[];finite=[];parameter_checks=[];predictions={}
    checkpoints=0;max_formula=max_fd=max_parameter_fd=0.
    for family in FAMILIES:
        initial_pair=None
        for readout in ('gradient','gate_omitted'):
            arm=family+'_'+readout;folder=raw/'runs'/DATASET/arm
            assert load(folder/'status.json')['state']=='COMPLETE'
            result=load(folder/'result.json')
            assert config['arms'][arm]==dict(family=family,readout=readout)
            for step in (0,1000,2500,5000):
                path=folder/f'step_{step:07d}.pt';bindings[str(path)]=sha(path)
                state=torch.load(path,map_location='cpu',weights_only=False)
                assert torch.equal(state['data_rng'],expected[step][0]) and state['input_stream_digest']==expected[step][1]
                if step==0:
                    if initial_pair is None:initial_pair=state
                    else:
                        for key in ('model','runtime_buffers','data_rng','torch_rng'):same(state[key],initial_pair[key])
                    prior_path=parent/'output/runs'/DATASET/(family+'_gradient')/path.name
                    bindings[str(prior_path)]=sha(prior_path)
                    prior=torch.load(prior_path,map_location='cpu',weights_only=False)
                    for key in ('model','runtime_buffers','data_rng','torch_rng'):same(state[key],prior[key])
                spectra=load(folder/f'spectrum_{step:07d}.json')
                assert len(spectra)==(6 if family=='ode_k2' else 0)
                for name,record in spectra.items():
                    p=state['model']['raw.'+name+'.proj.weight'].double()
                    sv=torch.linalg.svdvals(p);ev=torch.linalg.eigvals(p)
                    np.testing.assert_allclose(sv.numpy(),record['singular_values'],rtol=1e-11,atol=1e-12)
                    np.testing.assert_allclose(ev.real.numpy(),record['eigenvalue_real'],rtol=1e-10,atol=1e-11)
                    np.testing.assert_allclose(ev.imag.numpy(),record['eigenvalue_imag'],rtol=1e-10,atol=1e-11)
                    p_rows.append(dict(arm=arm,step=step,layer=name,sigma_max=float(sv.max()),frobenius=float(p.norm())))
                visits=load(folder/f'visits_{step:07d}.json')
                assert len(visits)==(6 if family=='ode_k2' else 0)
                checkpoints+=1
            model=build(16,family,readout).eval();restore_model(model,state)
            for key,value in compute_budget(model).items():
                if key!='matrix_count_scope':assert result[key]==value
            for parameter in model.parameters():parameter.requires_grad_(False)
            prediction=field(model,x,t,torch.float32)
            path=folder/'diagnostics/field.npz';bindings[str(path)]=sha(path)
            with np.load(path) as data:np.testing.assert_array_equal(prediction,data['prediction'])
            assert float(np.mean((prediction-truth)**2))==result['oracle_mse']
            predictions[arm]=prediction
            double=build(16,family,readout).double().eval();restore_model(double,state)
            for parameter in double.parameters():parameter.requires_grad_(False)
            reference=PartialJacobianReference(copy.deepcopy(double.raw),readout=='gate_omitted')
            dv=field(double,x,t,torch.float64);rv=field(reference,x,t,torch.float64)
            np.testing.assert_allclose(dv,rv,rtol=1e-9,atol=1e-10)
            max_formula=max(max_formula,float(np.max(np.abs(dv-rv))))
            np.savez_compressed(output/(arm+'-independent-field.npz'),explicit=dv,partial_jacobian_reference=rv)
            jac_errors=[]
            for clock in TIMES:
                with np.load(bank/f'panel_{clock:.1f}.npz') as data:px,pt,pj=[data[k] for k in ('x','t','oracle_jacobian')]
                value,jac=derivative(model,torch.from_numpy(px),torch.from_numpy(pt))
                path=folder/'diagnostics'/f'panel_{clock:.1f}.npz';bindings[str(path)]=sha(path)
                with np.load(path) as data:
                    np.testing.assert_array_equal(value.numpy(),data['prediction']);np.testing.assert_array_equal(jac.numpy(),data['jacobian'])
                jj=jac.numpy().astype(np.float64);sym=.5*(jj+jj.transpose(0,2,1));skew=.5*(jj-jj.transpose(0,2,1))
                err=float(np.mean(np.sum((jj-pj)**2,axis=(1,2))));jac_errors.append(err)
                geometry.append(dict(arm=arm,time=clock,jacobian_error=err,
                    symmetric_error=float(np.mean(np.sum((sym-pj)**2,axis=(1,2)))),skew_error=float(np.mean(np.sum(skew**2,axis=(1,2))))))
                xx=torch.from_numpy(px[:1]).double();tt=torch.from_numpy(pt[:1]).double()
                _,j=derivative(double,xx,tt)
                direction=torch.cos(torch.arange(16,dtype=torch.float64)).reshape(1,16);direction=direction/direction.norm()
                eps=1e-5
                fd=(reference(xx+eps*direction,tt)-reference(xx-eps*direction,tt))/(2*eps)
                analytic=torch.einsum('bij,bj->bi',j,direction)
                error=float((fd-analytic).norm()/max(1.,float(analytic.norm())))
                assert error<1e-5,(arm,clock,error)
                max_fd=max(max_fd,error);finite.append(dict(arm=arm,time=clock,input_fd_error=error))
            np.testing.assert_allclose(np.mean(jac_errors),result['jacobian_error'],rtol=1e-12,atol=1e-12)
            with np.load(bank/'sampling.npz') as data:starts,ends,dirs=[data[k] for k in ('source','target','directions')]
            samples=generated_samples(model,torch.from_numpy(starts),32).numpy()
            path=folder/'diagnostics/samples.npz';bindings[str(path)]=sha(path)
            with np.load(path) as data:np.testing.assert_array_equal(samples,data['generated'])
            sw=float(np.sqrt(np.mean((np.sort(samples.astype(np.float64)@dirs.astype(np.float64).T,axis=0)
                -np.sort(ends.astype(np.float64)@dirs.astype(np.float64).T,axis=0))**2)))
            np.testing.assert_allclose(sw,result['sw2'],rtol=1e-6,atol=1e-7)
            # A separate native partial-Jacobian parameter difference checks that
            # the omitted forward function's parameter derivatives remain live.
            name=('raw.blocks.1.ffn.ode1.t_func.0.weight' if family=='ode_k2' else 'raw.blocks.1.ffn.w12.weight')
            parameter=dict(double.named_parameters())[name];parameter.requires_grad_(True)
            xx=torch.from_numpy(x[:2]).double();tt=torch.from_numpy(t[:2]).double();yy=torch.from_numpy(target[:2]).double()
            gradient=torch.autograd.grad((double(xx,tt)-yy).square().mean(),parameter)[0]
            direction=torch.cos(torch.arange(parameter.numel(),dtype=torch.float64)).reshape_as(parameter);direction/=direction.norm()
            rp=dict(reference.named_parameters())[name];eps=1e-5
            with torch.no_grad():
                original=rp.clone();rp.copy_(original+eps*direction);plus=(reference(xx,tt)-yy).square().mean()
                rp.copy_(original-eps*direction);minus=(reference(xx,tt)-yy).square().mean();rp.copy_(original)
            actual=float((plus-minus)/(2*eps));expected_gradient=float((gradient*direction).sum())
            error=abs(actual-expected_gradient)/max(1.,abs(expected_gradient))
            assert error<1e-6,(arm,'parameter FD',error)
            max_parameter_fd=max(max_parameter_fd,error)
            parameter_checks.append(dict(arm=arm,parameter=name,analytic=expected_gradient,independent_finite_difference=actual,relative_error=error))
            rows.append({key:result[key] for key in ('dataset','arm','family','readout','oracle_mse','jacobian_error','sw2','parameters','dense_forward_macs','dense_training_forward_backward_macs')})
            for line in (folder/'metrics.jsonl').read_text().splitlines():curves.append(dict(arm=arm,**json.loads(line)))
            print(json.dumps(dict(audited=arm,oracle_mse=result['oracle_mse'])),flush=True)
    ratios={family:next(r['oracle_mse'] for r in rows if r['arm']==family+'_gate_omitted')/
            next(r['oracle_mse'] for r in rows if r['arm']==family+'_gradient') for family in FAMILIES}
    old_ode=next(float(r['oracle_mse']) for r in csv.DictReader((context_results/'comparison.csv').open())
        if r['dataset']==DATASET and r['arm']=='ode_k2_gradient')
    new_ode=next(r['oracle_mse'] for r in rows if r['arm']=='ode_k2_gradient')
    compatibility=abs(new_ode/old_ode-1)<=.1
    largest_generic=max(ratios['swiglu128'],ratios['swiglu192'])
    extend=compatibility and ratios['ode_k2']>2 and ratios['ode_k2']>2*largest_generic and largest_generic<=2
    decision=dict(practical_candidate=False,contextual_full_ODE_compatibility=compatibility,
        full_explicit_to_old_ODE_ratio=new_ode/old_ode,omitted_to_full=ratios,
        extend_exact_factor_to_Gaussian_Ring=extend,
        action='replicate exact mechanism factor on two tasks' if extend else 'stop this registered mechanism factor without rescue')
    table(output/'comparison.csv',rows);table(output/'symmetric-skew.csv',geometry)
    table(output/'p-trajectories.csv',p_rows);table(output/'finite-differences.csv',finite)
    table(output/'parameter-finite-differences.csv',parameter_checks);write(output/'training-curves.json',curves)
    write(output/'input-bindings.json',bindings);write(output/'decision.json',decision)
    monitor=[json.loads(line) for line in (raw/'monitor.jsonl').read_text().splitlines()]
    audit=dict(status='PASS',checkpoints=checkpoints,P_matrices=len(p_rows),native_field_replays=len(rows),
        native_panel_replays=len(geometry),sample_replays=len(rows),independent_complete_fields=len(rows),
        independent_input_finite_differences=len(finite),independent_parameter_finite_differences=len(parameter_checks),
        maximum_formula_error=max_formula,maximum_input_fd_error=max_fd,maximum_parameter_fd_error=max_parameter_fd,
        true_regenerated_streams=1,monitor_samples=len(monitor),source_manifest_sha256=sha(source/'source-sha256.json'),
        analyzer_sha256=sha(Path(__file__)))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    positions=np.arange(3)
    for j,readout in enumerate(('gradient','gate_omitted')):
        values=[next(r['oracle_mse'] for r in rows if r['arm']==f+'_'+readout) for f in FAMILIES]
        axes[0].bar(positions+(-.18 if j==0 else .18),values,width=.36,label=readout.replace('_',' '))
    axes[0].set(xticks=positions,xticklabels=['ODE','Swi128','Swi192'],ylabel='Full validation oracle MSE',yscale='log')
    axes[0].legend(fontsize=8)
    axes[1].bar(positions,[ratios[f] for f in FAMILIES]);axes[1].axhline(1,color='black',linewidth=.8)
    axes[1].set(xticks=positions,xticklabels=['ODE','Swi128','Swi192'],ylabel='Omitted/full field MSE')
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(output/f'gate-adjoint-comparison.{suffix}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    for ax,family in zip(axes,FAMILIES):
        for readout in ('gradient','gate_omitted'):
            rr=[r for r in curves if r['arm']==family+'_'+readout and 'oracle_mse' in r]
            ax.plot([r['step'] for r in rr],[r['oracle_mse'] for r in rr],label=readout.replace('_',' '))
        ax.set(title=family,xlabel='Optimizer update',yscale='log');ax.grid(alpha=.18)
    axes[0].set_ylabel('Full validation oracle MSE');axes[0].legend(fontsize=8);fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(output/f'learning-curves.{suffix}',dpi=180)
    plt.close(fig)
    for name,value in manifest.items():assert sha(source/name)==value
    for name,value in bindings.items():assert sha(name)==value
    write(output/'audit.json',audit)
    write(output/'status.json',dict(state='COMPLETE',audit_status='PASS'))
    print(json.dumps(dict(audit=audit,decision=decision)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',type=Path,required=True);p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--context-results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.runtime.resolve(),a.parent.resolve(),a.context_results.resolve(),a.output.resolve())
