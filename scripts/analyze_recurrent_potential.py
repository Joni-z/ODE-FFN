"""Audit the completed Product16 layout/readout factor and fixed eligibility."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from synthetic.recurrent_potential import build
from synthetic.conservative_readout import compute_budget
from synthetic.population import fingerprint_batch
from synthetic.train import restore_model,load_targets,generated_samples

DATASET='rotated_product_bimodal_16d'
FAMILIES=('ode_k2','swiglu128','swiglu192')
TIMES=(.1,.3,.5,.7,.9)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def table(p,rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


class RawJacobianReference(nn.Module):
    def __init__(self,raw,gradient):
        super().__init__();self.raw=raw;self.gradient=gradient

    def forward(self,x,t):
        if not self.gradient:
            with torch.no_grad():return self.raw(x,t)
        with torch.enable_grad():
            xx=x.detach().requires_grad_(True);g=self.raw(xx,t)
            jac=torch.stack([torch.autograd.grad(g[:,j].sum(),xx,retain_graph=j+1<g.shape[1])[0]
                             for j in range(g.shape[1])],1)
            return .5*(g.detach()+torch.einsum('bij,bi->bj',jac,xx).detach())


def field(model,x,t,dtype):
    pieces=[]
    with torch.no_grad():
        for i in range(0,len(x),1024):
            pieces.append(model(torch.from_numpy(x[i:i+1024]).to(dtype),torch.from_numpy(t[i:i+1024]).to(dtype)).numpy())
    return np.concatenate(pieces)


def derivative(model,x,t):
    xx=x.detach().clone().requires_grad_(True);value=model(xx,t)
    jac=torch.stack([torch.autograd.grad(value[:,j].sum(),xx,retain_graph=j+1<value.shape[1])[0]
                     for j in range(value.shape[1])],1)
    return value.detach().numpy(),jac.detach()


def main(runtime,recurrent,parent,context,old_results,output):
    torch.set_num_threads(1)
    raw=runtime/'output';source=runtime/'source'
    state=load(raw/'status.json');assert state['state']=='COMPLETE' and state['completed']==state['total']==8
    assert load(runtime/'preflight/PASS.json')['status']=='PASS'
    hashes=load(source/'source-sha256.json')
    for name,value in hashes.items():assert sha(source/name)==value
    for name in ('synthetic/models.py','synthetic/recurrent_depth.py','synthetic/recurrent_potential.py','synthetic/conservative_readout.py'):
        assert sha(ROOT/name)==hashes[name]
    for name,value in load(context/'report-file-sha256.json').items():assert sha(context.parent/name)==value
    for name,value in load(old_results/'summary-provenance.json')['files'].items():assert sha(old_results/name)==value
    old_group=recurrent/'output-product'
    assert load(old_group/'status.json')['state']=='COMPLETE'
    old_source=Path(load(old_group/'provenance.json')['source_root'])
    for name,value in load(old_source/'source-sha256.json').items():assert sha(old_source/name)==value
    output.mkdir(parents=True,exist_ok=False);write(output/'status.json',dict(state='RUNNING'))
    config=load(raw/'study.json');assert config['seed']==0 and config['steps']==5000
    bank=raw/'banks'/DATASET;bindings={str(raw/'study.json'):sha(raw/'study.json')}
    for p in bank.glob('*.npz'):
        bindings[str(p)]=sha(p)
        for older in (parent/'output/banks'/DATASET/p.name,old_group/'banks'/DATASET/p.name):
            with np.load(p) as a,np.load(older) as b:
                assert a.files==b.files
                for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    with np.load(bank/'field.npz') as data:x,t,truth,target=[data[k] for k in ('x','t','oracle','pair')]
    data_root=Path(config['data_roots'][DATASET])
    targets=torch.from_numpy(load_targets(data_root/DATASET/'train.npz',16))
    rng=torch.Generator().manual_seed(100000);digest=hashlib.sha256(b'').hexdigest()
    expected={0:(rng.get_state().clone(),digest)}
    for step in range(1,5001):
        yy=targets[torch.randint(len(targets),(512,),generator=rng)]
        z=torch.randn(yy.shape,generator=rng);clock=torch.rand((512,1),generator=rng)
        digest=fingerprint_batch(digest,yy,z,clock)
        if step in (1000,2500,5000):expected[step]=(rng.get_state().clone(),digest)
    rows=[];spectra=[];geometry=[];fds=[];paramfds=[];curves=[]
    count=0;maximum_formula=maximum_fd=maximum_parameter_fd=0.
    for family in FAMILIES:
        for layout in ('shared','cloned'):
            for readout in ('direct','gradient'):
                new=readout=='gradient' or family=='swiglu128'
                arm=family+'_'+layout+'_'+readout
                actual=arm if new else family+'_'+layout+'_final'
                folder=(raw if new else old_group)/'runs'/DATASET/actual
                assert load(folder/'status.json')['state']=='COMPLETE'
                result=load(folder/'result.json')
                for step in (0,1000,2500,5000):
                    p=folder/f'step_{step:07d}.pt';bindings[str(p)]=sha(p)
                    saved=torch.load(p,map_location='cpu',weights_only=False)
                    assert torch.equal(saved['data_rng'],expected[step][0]) and saved['input_stream_digest']==expected[step][1]
                    if step==0:
                        prior_path=parent/'output/runs'/DATASET/(family+'_gradient')/p.name
                        bindings[str(prior_path)]=sha(prior_path)
                        prior=torch.load(prior_path,map_location='cpu',weights_only=False)
                        for kind in ('model','runtime_buffers'):
                            for name,value in saved[kind].items():
                                mapped=name[4:] if new else name
                                for index in (1,2):mapped=mapped.replace(f'.blocks.{index}.','.blocks.0.')
                                assert torch.equal(value,prior[kind][mapped]),(arm,name)
                        assert torch.equal(saved['torch_rng'],prior['torch_rng'])
                    records=load(folder/f'spectrum_{step:07d}.json')
                    assert len(records)==((2 if layout=='shared' else 6) if family=='ode_k2' else 0)
                    for name,record in records.items():
                        pweight=saved['model']['raw.'+name+'.proj.weight'].double()
                        sv=torch.linalg.svdvals(pweight);ev=torch.linalg.eigvals(pweight)
                        np.testing.assert_allclose(sv.numpy(),record['singular_values'],rtol=1e-11,atol=1e-12)
                        np.testing.assert_allclose(ev.real.numpy(),record['eigenvalue_real'],rtol=1e-10,atol=1e-11)
                        np.testing.assert_allclose(ev.imag.numpy(),record['eigenvalue_imag'],rtol=1e-10,atol=1e-11)
                        spectra.append(dict(arm=arm,step=step,layer=name,sigma_max=float(sv.max()),frobenius=float(pweight.norm())))
                    visits=load(folder/f'visits_{step:07d}.json')
                    assert len(visits)==(6 if family=='ode_k2' else 0)
                    count+=1
                model=build(16,family,layout+'_'+readout).eval()
                restore_model(model if new else model.raw,saved)
                costs=compute_budget(model)
                for key in ('parameters','dense_forward_macs'):assert costs[key]==result[key]
                for parameter in model.parameters():parameter.requires_grad_(False)
                native=field(model,x,t,torch.float32)
                p=folder/'diagnostics/field.npz';bindings[str(p)]=sha(p)
                with np.load(p) as data:np.testing.assert_array_equal(native,data['prediction'])
                assert float(np.mean((native-truth)**2))==result['oracle_mse']
                double=build(16,family,layout+'_'+readout).double().eval()
                restore_model(double if new else double.raw,saved)
                for parameter in double.parameters():parameter.requires_grad_(False)
                reference=RawJacobianReference(copy.deepcopy(double.raw),readout=='gradient')
                dv=field(double,x,t,torch.float64);rv=field(reference,x,t,torch.float64)
                np.testing.assert_allclose(dv,rv,rtol=1e-9,atol=1e-10)
                maximum_formula=max(maximum_formula,float(np.max(np.abs(dv-rv))))
                np.savez_compressed(output/(arm+'-independent-field.npz'),explicit=dv,raw_jacobian_reference=rv)
                for clock in TIMES:
                    with np.load(bank/f'panel_{clock:.1f}.npz') as data:px,pt,truthj=[data[k] for k in ('x','t','oracle_jacobian')]
                    value,jac=derivative(model,torch.from_numpy(px),torch.from_numpy(pt))
                    p=folder/'diagnostics'/f'panel_{clock:.1f}.npz';bindings[str(p)]=sha(p)
                    with np.load(p) as data:
                        np.testing.assert_array_equal(value,data['prediction']);np.testing.assert_array_equal(jac.numpy(),data['jacobian'])
                    jj=jac.numpy().astype(np.float64);sym=.5*(jj+jj.transpose(0,2,1));skew=.5*(jj-jj.transpose(0,2,1))
                    geometry.append(dict(arm=arm,time=clock,jacobian_error=float(np.mean(np.sum((jj-truthj)**2,axis=(1,2)))),
                        symmetric_error=float(np.mean(np.sum((sym-truthj)**2,axis=(1,2)))),skew_error=float(np.mean(np.sum(skew**2,axis=(1,2))))))
                    xx=torch.from_numpy(px[:1]).double();tt=torch.from_numpy(pt[:1]).double()
                    _,j=derivative(double,xx,tt)
                    direction=torch.cos(torch.arange(16,dtype=torch.float64)).reshape(1,16);direction/=direction.norm()
                    eps=1e-5
                    finite=(reference(xx+eps*direction,tt)-reference(xx-eps*direction,tt))/(2*eps)
                    analytic=torch.einsum('bij,bj->bi',j,direction)
                    error=float((finite-analytic).norm()/max(1.,float(analytic.norm())))
                    assert error<1e-5,(arm,clock,error)
                    maximum_fd=max(maximum_fd,error);fds.append(dict(arm=arm,time=clock,relative_error=error))
                with np.load(bank/'sampling.npz') as data:starts,ends,dirs=[data[k] for k in ('source','target','directions')]
                samples=generated_samples(model,torch.from_numpy(starts),32).numpy()
                p=folder/'diagnostics/samples.npz';bindings[str(p)]=sha(p)
                with np.load(p) as data:np.testing.assert_array_equal(samples,data['generated'])
                sw=float(np.sqrt(np.mean((np.sort(samples.astype(np.float64)@dirs.astype(np.float64).T,axis=0)-
                    np.sort(ends.astype(np.float64)@dirs.astype(np.float64).T,axis=0))**2)))
                np.testing.assert_allclose(sw,result['sw2'],rtol=1e-6,atol=1e-7)
                pname='raw.raw.blocks.0.ffn.'+('ode1.proj.weight' if family=='ode_k2' else 'w12.weight')
                parameter=dict(double.named_parameters())[pname];parameter.requires_grad_(True)
                xx=torch.from_numpy(x[:2]).double();tt=torch.from_numpy(t[:2]).double();yy=torch.from_numpy(target[:2]).double()
                gradient=torch.autograd.grad((double(xx,tt)-yy).square().mean(),parameter)[0]
                direction=torch.cos(torch.arange(parameter.numel(),dtype=torch.float64)).reshape_as(parameter);direction/=direction.norm()
                rp=dict(reference.named_parameters())[pname];eps=1e-5
                with torch.no_grad():
                    original=rp.clone();rp.copy_(original+eps*direction);plus=(reference(xx,tt)-yy).square().mean()
                    rp.copy_(original-eps*direction);minus=(reference(xx,tt)-yy).square().mean();rp.copy_(original)
                numeric=float((plus-minus)/(2*eps));analytic=float((gradient*direction).sum())
                error=abs(numeric-analytic)/max(1.,abs(analytic))
                assert error<1e-6,(arm,'parameter',error)
                maximum_parameter_fd=max(maximum_parameter_fd,error)
                paramfds.append(dict(arm=arm,parameter=pname,analytic=analytic,numeric=numeric,relative_error=error))
                rows.append(dict(dataset=DATASET,arm=arm,family=family,layout=layout,readout=readout,reused=not new,
                    oracle_mse=result['oracle_mse'],jacobian_error=result['jacobian_error'],sw2=result['sw2'],
                    parameters=costs['parameters'],dense_forward_macs=costs['dense_forward_macs'],
                    dense_training_forward_backward_macs=costs['dense_training_forward_backward_macs']))
                for line in (folder/'metrics.jsonl').read_text().splitlines():curves.append(dict(arm=arm,reused=not new,**json.loads(line)))
                print(json.dumps(dict(audited=arm,oracle_mse=result['oracle_mse'])),flush=True)
    context_rows=[]
    for r in csv.DictReader((context/'comparison.csv').open()):
        if r['dataset']==DATASET:
            context_rows.append(dict(dataset=DATASET,arm=r['arm'],family=r['family'],layout='native',readout=r['readout'],
                reused=True,**{k:float(r[k]) for k in ('oracle_mse','jacobian_error','sw2')},
                **{k:int(r[k]) for k in ('parameters','dense_forward_macs','dense_training_forward_backward_macs')}))
    all_rows=rows+context_rows
    generic=[r for r in all_rows if r['family']!='ode_k2']
    candidate=next(r for r in rows if r['arm']=='ode_k2_shared_gradient')
    own=[next(r for r in all_rows if r['arm']==name) for name in ('ode_k2_shared_direct','ode_k2_cloned_gradient','ode_k2_gradient')]
    best=min(r['oracle_mse'] for r in generic)
    parity=[r for r in generic if r['oracle_mse']<=1.05*best]
    quality=candidate['oracle_mse']<=.9*best and all(candidate['oracle_mse']<=.95*r['oracle_mse'] for r in own)
    parameter=(candidate['oracle_mse']<=1.05*best and candidate['oracle_mse']<=1.05*own[1]['oracle_mse']
        and candidate['parameters']<=.75*min(r['parameters'] for r in parity)
        and candidate['dense_forward_macs']<=1.05*min(r['dense_forward_macs'] for r in parity))
    guard=all(candidate[key]<=1.1*min(r[key] for r in generic) for key in ('jacobian_error','sw2'))
    decision=dict(phaseA_eligible=(quality or parameter) and guard,quality_route=quality,parameter_route=parameter,
        derivative_sampling_guard=guard,best_generic_MSE=best,candidate_MSE=candidate['oracle_mse'],
        action='register two-task replication' if (quality or parameter) and guard else 'stop exact layout/readout factor without rescue',
        accepted_architecture=False)
    interactions=[]
    for family in FAMILIES:
        values={(r['layout'],r['readout']):r['oracle_mse'] for r in rows if r['family']==family}
        direct=values['shared','direct']/values['cloned','direct']
        grad=values['shared','gradient']/values['cloned','gradient']
        interactions.append(dict(family=family,shared_cloned_direct=direct,shared_cloned_gradient=grad,ratio_of_ratios=grad/direct))
    table(output/'comparison.csv',all_rows);table(output/'factor.csv',rows)
    table(output/'interaction.csv',interactions);table(output/'p-trajectories.csv',spectra)
    table(output/'symmetric-skew.csv',geometry);table(output/'finite-differences.csv',fds)
    table(output/'parameter-finite-differences.csv',paramfds);write(output/'training-curves.json',curves)
    write(output/'decision.json',decision);write(output/'input-bindings.json',bindings)
    for name,value in hashes.items():assert sha(source/name)==value
    for name,value in bindings.items():assert sha(name)==value
    audit=dict(status='PASS',checkpoints=count,unique_P_snapshots=len(spectra),native_fields=len(rows),native_panels=len(geometry),
        native_samples=len(rows),independent_complete_fields=len(rows),input_finite_differences=len(fds),
        parameter_finite_differences=len(paramfds),maximum_formula_error=maximum_formula,maximum_input_fd_error=maximum_fd,
        maximum_parameter_fd_error=maximum_parameter_fd,source_manifest_sha256=sha(source/'source-sha256.json'),
        analyzer_sha256=sha(Path(__file__)),monitor_samples=len((raw/'monitor.jsonl').read_text().splitlines()))
    write(output/'audit.json',audit);write(output/'status.json',dict(state='COMPLETE'))
    print(json.dumps(dict(audit=audit,decision=decision)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime','recurrent','parent','context','old-results','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();main(a.runtime.resolve(),a.recurrent.resolve(),a.parent.resolve(),a.context.resolve(),a.old_results.resolve(),a.output.resolve())
