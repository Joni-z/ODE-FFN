"""Frozen native-Heun paths, common-state errors and oracle-switch diagnostics."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cycle14'))
import audit as parent
from audit import base,build_model,torch

HERE=Path(__file__).resolve().parent
TAG='professor_transport15_20261104'


def cohort():
    grid=json.loads((HERE/'grid.json').read_text())
    assert base.digest(HERE.parent/'cycle14/grid.json')==grid['source_grid_sha256']
    old=parent.cohort()
    assert len(grid['cells'])==6 and grid['model_seed']==0
    return grid,old


class Oracle(torch.nn.Module):
    def __init__(self,law):
        super().__init__();self.law=law
    def forward(self,x,t):
        value=self.law.velocity(x.detach().cpu().numpy(),t.detach().cpu().numpy())
        return torch.from_numpy(value).to(device=x.device,dtype=x.dtype)


def oracle_jacobian(law,x,t):
    # Differentiate the same native diagonal-GMM posterior mixture formula.
    time_,variance,residual,w,_=law._terms(x,t)
    a=(time_*law.variances[None]-(1-time_))/variance
    component=law.means[None]+a*residual
    u=law.velocity(x,t)
    mean_a=np.einsum('nk,nkd->nd',w,a)
    jac=np.zeros((len(x),law.dim,law.dim),dtype=np.float64)
    k=np.arange(law.dim);jac[:,k,k]=mean_a
    jac+=np.einsum('nk,nki,nkj->nij',w,component-u[:,None],-residual/variance)
    return jac


def full_jacobian(model,x,t):
    with torch.enable_grad():
        x=x.detach().requires_grad_(True);v=model(x,t)
        return torch.stack([torch.autograd.grad(v[:,j].sum(),x,retain_graph=j+1<v.shape[1])[0]
                            for j in range(v.shape[1])],dim=1).detach()


@torch.no_grad()
def native_step(model,x,index,steps):
    # Verbatim arithmetic from reference/synthetic/train.py generated_samples.
    dt=1.0/steps
    t=torch.full((len(x),1),index*dt,device=x.device,dtype=x.dtype)
    velocity=model(x,t)
    predictor=x+dt*velocity
    t_next=torch.full_like(t,(index+1)*dt)
    return x+0.5*dt*(velocity+model(predictor,t_next))


class Policy(torch.nn.Module):
    def __init__(self,model,oracle,mode,cut,steps,record=False):
        super().__init__();self.model=model;self.oracle=oracle;self.mode=mode
        self.cut=cut;self.steps=steps;self.calls=0;self.states=[];self.record=record
    def forward(self,x,t):
        index=self.calls//2
        assert index<self.steps
        if self.record and self.calls%2==0:self.states.append(x.detach().cpu().numpy().copy())
        self.calls+=1
        learned=(self.mode=='learned' or self.mode=='learned_prefix' and index<self.cut
                 or self.mode=='oracle_prefix' and index>=self.cut)
        return (self.model if learned else self.oracle)(x,t)


def sample(model,oracle,source,steps,mode='learned',cut=0,record=False):
    policy=Policy(model,oracle,mode,cut,steps,record)
    end=base.train.generated_samples(policy,source,steps)
    assert policy.calls==2*steps
    if record:
        policy.states.append(end.detach().cpu().numpy().copy())
        return end,np.stack(policy.states)
    return end


def loaded(cell,grid,old,data_root,device):
    case=next(c for c in old['cells'] if c['id']==cell['source_case'])
    folder,cfg,manifest,law,fixed=parent.inputs(case,old,data_root,device)
    assert law.dim==2 and case['host']=='lenovo'
    model,identity=parent.restored(case,folder,manifest,2,cell['checkpoint_step'],device)
    original=Path(grid['source_runtime'])/parent.TAG/case['dataset']/case['id']/'seed000'
    assert (original/'ADAPTER_COMPLETE').exists()
    binding=json.loads((original/'binding.json').read_text())
    with np.load(original/'banks.npz',allow_pickle=False) as z:banks={k:z[k].copy() for k in z.files}
    assert {k:parent.array_hash(v) for k,v in banks.items()}==binding['bank_identity']
    records=[json.loads(line) for line in (original/'rows.jsonl').read_text().splitlines()]
    row=next(r for r in records if r['checkpoint_step']==cell['checkpoint_step'] and r['bank']==0 and r['heun_steps']==128)
    archive=original/row['generated_file'];assert base.digest(archive)==row['generated_sha256']
    assert row['checkpoint_sha256']==identity['checkpoint_sha256']
    return model,law,banks,archive,identity


def self_test(device):
    from synthetic.data import suite
    torch.set_num_threads(1)
    law=suite()['gmm_ring8_2d'];x=np.linspace(-2,2,32).reshape(16,2)
    errors=[]
    for t0 in [.05,.25,.5,.75,.95]:
        t=np.full((len(x),1),t0);jac=oracle_jacobian(law,x,t);columns=[]
        for j in range(2):
            d=np.zeros_like(x);d[:,j]=1e-6
            columns.append((law.velocity(x+d,t)-law.velocity(x-d,t))/2e-6)
        finite=np.stack(columns,axis=2)
        np.testing.assert_allclose(jac,finite,rtol=1e-5,atol=1e-7)
        errors.append(float(np.max(np.abs(jac-finite))))
    class Constant(torch.nn.Module):
        def __init__(self,value):super().__init__();self.value=value
        def forward(self,x,t):return torch.full_like(x,self.value)
    source=torch.zeros(8,2,device=device,dtype=torch.float64)
    model,oracle=Constant(2),Constant(1)
    for mode,cut,expected in [('learned',0,2.),('oracle',0,1.),('learned_prefix',32,1.25),('oracle_prefix',32,1.75)]:
        end,path=sample(model,oracle,source,128,mode,cut,True)
        assert torch.equal(end,torch.full_like(source,expected)) and path.shape==(129,8,2)
    x=torch.linspace(-.3,.8,16,device=device,dtype=torch.double).reshape(8,2)
    rows=[]
    for family in ['pd_isru_k1','free_isru','swiglu64']:
        torch.manual_seed(0);model=build_model(2,family).to(device).double().eval()
        with torch.no_grad():model.output.weight.copy_(torch.linspace(-.1,.1,model.output.weight.numel(),device=device,dtype=torch.double).reshape_as(model.output.weight))
        t=torch.full((len(x),1),.4,device=device,dtype=x.dtype)
        jac=full_jacobian(model,x,t)
        with torch.no_grad():
            columns=[]
            for j in range(2):
                d=torch.zeros_like(x);d[:,j]=1e-5
                columns.append((model(x+d,t)-model(x-d,t))/2e-5)
            finite=torch.stack(columns,dim=2)
            assert torch.allclose(jac,finite,rtol=1e-5,atol=1e-7)
            y=x.clone()
            for i in range(32):y=native_step(model,y,i,32)
            assert torch.equal(y,base.train.generated_samples(model,x,32))
            changed=x.clone();changed[1]+=1
            assert torch.equal(model(x,t)[0],model(changed,t)[0])
        rows.append(dict(family=family,nonzero_jacobian=bool(jac.abs().max()>0),native_step_exact=True,batch_independent=True))
    return dict(passed=True,device=str(device),rows=rows,oracle_jacobian_max_error=max(errors),switch_boundaries_exact=True,producer_sha256=base.digest(__file__))


def prepare(grid,old,data_root,device,out):
    out.mkdir(parents=True,exist_ok=False)
    model,law,banks,archive,identity=loaded(grid['cells'][0],grid,old,data_root,device)
    source=torch.from_numpy(banks['source_0']).to(device);oracle=Oracle(law);arrays=dict(source=banks['source_0'],target=banks['target_0'],target2=banks['target2_0'],directions=banks['directions']);rows=[]
    for steps in grid['oracle_resolutions']:
        end,path=sample(model,oracle,source,steps,'oracle',record=True)
        if steps==128:arrays['oracle_path']=path
        arrays[f'oracle_end_{steps}']=end.cpu().numpy()
        rows.append(dict(steps=steps,sw2=base.train.sliced_w2(end.double(),torch.from_numpy(arrays['target']).to(device).double(),torch.from_numpy(arrays['directions']).to(device).double())))
    assert np.isfinite(arrays['oracle_path']).all()
    np.savez_compressed(out/'common.npz',**arrays)
    differences={f'{a}_vs_{b}':dict(rmse=float(np.sqrt(np.mean((arrays[f'oracle_end_{a}'].astype(float)-arrays[f'oracle_end_{b}'].astype(float))**2))),maximum=float(np.max(np.abs(arrays[f'oracle_end_{a}']-arrays[f'oracle_end_{b}'])))) for a,b in [(128,256),(256,512),(128,512)]}
    base.dump(out/'binding.json',dict(passed=True,source_sha256=base.digest(__file__),grid_sha256=base.digest(HERE/'grid.json'),array_hashes={k:parent.array_hash(v) for k,v in arrays.items()},oracle_rows=rows,resolution_differences=differences,manifest_sha256=identity['manifest_sha256']))
    (out/'COMPLETE').write_text('Frozen common oracle reference complete.\n')
    return dict(passed=True,oracle_rows=rows,resolution_differences=differences)


def witness(grid,old,data_root,device,out):
    out.mkdir(parents=True,exist_ok=False);kernel=self_test(device);rows=[]
    for cell in grid['cells']:
        model,law,banks,archive,identity=loaded(cell,grid,old,data_root,device);before=parent.tensor_snapshot(model)
        source=torch.from_numpy(banks['source_0']).to(device);oracle=Oracle(law)
        end,path=sample(model,oracle,source,128,record=True)
        assert np.array_equal(end.cpu().numpy(),np.load(archive,allow_pickle=False))
        rebuilt=source.clone()
        for i in range(128):rebuilt=native_step(model,rebuilt,i,128)
        assert torch.equal(end,rebuilt)
        px=source[:16];pt=torch.full((len(px),1),.5,device=device)
        assert bool(torch.isfinite(full_jacobian(model,px,pt)).all())
        for mode,cut,reference in [('learned_prefix',128,end),('oracle_prefix',0,end)]:
            assert torch.equal(sample(model,oracle,source,128,mode,cut),reference)
        parent.same_tensors(model,before)
        rows.append(dict(cell=cell['id'],native4096_exact=True,helper128_exact=True,policy_degenerate_exact=True,jacobian_finite=True,tensors_unchanged=True,**identity))
        base.dump(out/'progress.json',dict(completed=len(rows),rows=rows))
    result=dict(passed=True,device=str(device),kernel=kernel,rows=rows,physical_checkpoints=6)
    base.dump(out/'witness.json',result);return result


def run_case(cell,grid,old,data_root,device,output):
    out=output/TAG/cell['dataset']/cell['id']/'seed000';out.mkdir(parents=True,exist_ok=False)
    common=output.parent/'common';assert (common/'COMPLETE').exists()
    bind=json.loads((common/'binding.json').read_text());assert bind['source_sha256']==base.digest(__file__) and bind['grid_sha256']==base.digest(HERE/'grid.json')
    with np.load(common/'common.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    assert {k:parent.array_hash(v) for k,v in arrays.items()}==bind['array_hashes']
    model,law,banks,archive,identity=loaded(cell,grid,old,data_root,device);before=parent.tensor_snapshot(model);oracle=Oracle(law)
    assert np.array_equal(arrays['source'],banks['source_0']) and np.array_equal(arrays['target'],banks['target_0'])
    source=torch.from_numpy(arrays['source']).to(device);start=time.monotonic()
    end,path=sample(model,oracle,source,128,record=True);assert np.array_equal(end.cpu().numpy(),np.load(archive,allow_pickle=False))
    np.save(out/'learned_path.npy',path,allow_pickle=False)
    z=arrays['oracle_path'];step_z=[];profiles=[];probe_x=[];preds=[];oracles=[];jacs=[];u_jacs=[]
    for index in range(129):
        state=torch.from_numpy(z[index]).to(device);t=torch.full((len(state),1),index/128,device=device)
        if index<128:step_z.append(native_step(model,state,index,128).cpu().numpy())
        if index%grid['profile_stride']==0:
            with torch.no_grad():pred=model(state,t).cpu().numpy()
            teacher=law.velocity(z[index],np.full((len(state),1),index/128))
            err=pred.astype(float)-teacher;radius=np.linalg.norm(z[index].astype(float),axis=1);valid=radius>1e-8
            radial=np.divide(z[index],radius[:,None],out=np.zeros_like(z[index],dtype=float),where=valid[:,None]);angular=np.stack([-radial[:,1],radial[:,0]],axis=1)
            er=np.sum(err*radial,axis=1);ea=np.sum(err*angular,axis=1)
            assert np.allclose(er[valid]**2+ea[valid]**2,np.sum(err[valid]**2,axis=1),rtol=1e-12,atol=1e-12)
            n=grid['jacobian_samples'];jac=full_jacobian(model,state[:n],t[:n]).cpu().numpy();uj=oracle_jacobian(law,z[index,:n],np.full((n,1),index/128))
            profiles.append(dict(time=index/128,coordinate_mse=float(np.mean(err**2)),radial_energy=float(np.mean(er[valid]**2)),angular_energy=float(np.mean(ea[valid]**2)),radial_bias=float(np.mean(er[valid])),angular_bias=float(np.mean(ea[valid])),valid_radius=int(valid.sum()),total=len(radius),jacobian_norm_mean=float(np.linalg.svd(jac,compute_uv=False)[:,0].mean()),symmetric_growth_mean=float(np.linalg.eigvalsh((jac+jac.transpose(0,2,1))/2)[:,-1].mean()),oracle_jacobian_mse=float(np.mean((jac-uj)**2))))
            preds.append(pred);oracles.append(teacher);jacs.append(jac);u_jacs.append(uj)
        base.train.json_line(out/'metrics.jsonl',dict(step=index,wall_seconds=time.monotonic()-start,validation_oracle_mse=profiles[-1]['coordinate_mse'] if profiles else None,sample=dict(phase='common_path',completed=index,total=128)))
    step_z=np.stack(step_z);np.savez_compressed(out/'profiles.npz',prediction=np.stack(preds),oracle=np.stack(oracles),jacobian=np.stack(jacs),oracle_jacobian=np.stack(u_jacs),learned_step_at_oracle=step_z)
    # Differences are evaluated in float64 from unchanged native float32 states.
    error=path.astype(float)-z.astype(float);defect=step_z.astype(float)-z[1:].astype(float);response=path[1:].astype(float)-step_z.astype(float)
    identity_error=float(np.max(np.abs(error[1:]-response-defect)));assert identity_error<1e-12
    step_rows=[]
    for i in range(128):
        norm=np.linalg.norm(error[i],axis=1);mask=norm>1e-8;gain=np.linalg.norm(response[i,mask],axis=1)/norm[mask]
        step_rows.append(dict(step=i,defect_mse=float(np.mean(defect[i]**2)),state_error_mse=float(np.mean(error[i]**2)),response_mse=float(np.mean(response[i]**2)),gain_count=int(mask.sum()),gain_mean=float(gain.mean()) if len(gain) else None,gain_median=float(np.median(gain)) if len(gain) else None,gain_p90=float(np.quantile(gain,.9)) if len(gain) else None,gain_max=float(gain.max()) if len(gain) else None))
    target=torch.from_numpy(arrays['target']).to(device).double();directions=torch.from_numpy(arrays['directions']).to(device).double()
    switches=[]
    for mode,cuts in [('learned',[0]),('oracle',[0]),('learned_prefix',grid['switch_steps']),('oracle_prefix',grid['switch_steps'])]:
        for cut in cuts:
            value=sample(model,oracle,source,128,mode,cut);name=f'{mode}_{cut}.npy';np.save(out/name,value.cpu().numpy(),allow_pickle=False)
            if mode=='learned':assert torch.equal(value,end)
            if mode=='oracle':assert np.array_equal(value.cpu().numpy(),z[-1])
            switches.append(dict(mode=mode,cut_step=cut,sw2=base.train.sliced_w2(value.double(),target,directions),endpoint_oracle_mse=float(np.mean((value.cpu().numpy().astype(float)-z[-1].astype(float))**2)),file=name,sha256=base.digest(out/name)))
    parent.same_tensors(model,before);assert all(p.grad is None for p in model.parameters())
    result=dict(passed=True,cell=cell,device=str(device),profiles=profiles,steps=step_rows,switches=switches,identity_max_error=identity_error,tensors_unchanged=True,native_bank0_exact=True,checkpoint=identity,common_binding_sha256=base.digest(common/'binding.json'),producer_sha256=base.digest(__file__),grid_sha256=base.digest(HERE/'grid.json'),wall_seconds=time.monotonic()-start)
    base.dump(out/'result.json',result);(out/'ADAPTER_COMPLETE').write_text('Frozen transport diagnostic complete.\n');return dict(passed=True,cell=cell['id'],profile_times=len(profiles),step_decompositions=len(step_rows),switches=len(switches),wall_seconds=result['wall_seconds'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',required=True);p.add_argument('--output',required=True);p.add_argument('--data-root');p.add_argument('--cell');p.add_argument('--self-test',action='store_true');p.add_argument('--witness',action='store_true');p.add_argument('--prepare',action='store_true')
    a=p.parse_args();torch.set_num_threads(1);device=torch.device(a.device);grid,old=cohort()
    if a.self_test:
        result=self_test(device);base.dump(a.output,result)
    elif a.witness:result=witness(grid,old,a.data_root,device,Path(a.output))
    elif a.prepare:result=prepare(grid,old,a.data_root,device,Path(a.output))
    else:
        cell=next(c for c in grid['cells'] if c['id']==a.cell)
        result=run_case(cell,grid,old,a.data_root,device,Path(a.output))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
