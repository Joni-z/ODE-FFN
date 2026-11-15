"""Oracle variational transport and time-window interventions, frozen weights."""
import argparse,json,time
from pathlib import Path
import numpy as np
from coefficient_trial import base,torch
import coefficient_evaluate as evaluation
from transport import Oracle,oracle_jacobian,native_step

def jacobian(law,x,t):
    if hasattr(law,'jacobian'):value=law.jacobian(x.numpy(),t.numpy())
    else:value=oracle_jacobian(law,x.numpy(),t.numpy())
    return torch.from_numpy(value).to(x)

@torch.no_grad()
def reference(law,source,steps):
    oracle=Oracle(law);dt=1/steps;y=source.clone();trace=[]
    for i in range(steps):
        t=y.new_full((len(y),1),i*dt);tn=torch.full_like(t,(i+1)*dt)
        v=oracle(y,t);pred=y+dt*v;vn=oracle(pred,tn)
        trace.append((y.clone(),pred.clone(),t,tn,v,vn,jacobian(law,y,t),jacobian(law,pred,tn)))
        y=y+.5*dt*(v+vn)
    assert torch.equal(y,base.train.generated_samples(oracle,source,steps))
    return y,trace

@torch.no_grad()
def sensitivity(model,trace):
    steps=len(trace);assert steps%5==0;dt=1/steps
    shape=trace[0][0].shape;z=trace[0][0].new_zeros((5,*shape));mse=[[] for _ in range(5)]
    for i,(x,pred,t,tn,v,vn,j,jn) in enumerate(trace):
        e=model(x,t)-v;en=model(pred,tn)-vn;window=i//(steps//5)
        forcing=torch.zeros_like(z);forcing[window]=e
        rhs=torch.einsum('bij,wbj->wbi',j,z)+forcing
        zp=z+dt*rhs;forcing_next=torch.zeros_like(z);forcing_next[window]=en
        rhsp=torch.einsum('bij,wbj->wbi',jn,zp)+forcing_next
        z=z+.5*dt*(rhs+rhsp);mse[window].append(float(e.square().mean()))
    return z,[float(np.mean(v)) for v in mse]

class Mixture(torch.nn.Module):
    def __init__(self,model,oracle,weight):super().__init__();self.model=model;self.oracle=oracle;self.weight=weight
    def forward(self,x,t):
        truth=self.oracle(x,t);return truth+self.weight*(self.model(x,t)-truth)

class WindowRepair(torch.nn.Module):
    def __init__(self,model,oracle,window,steps):
        super().__init__();self.model=model;self.oracle=oracle;self.window=window;self.steps=steps;self.calls=0
    def forward(self,x,t):
        index=self.calls//2;self.calls+=1
        return (self.oracle if index//(self.steps//5)==self.window else self.model)(x,t)

def self_test(law,source):
    x=source[:4];t=x.new_full((len(x),1),.37);j=jacobian(law,x,t);eps=1e-5
    v=torch.linspace(-.3,.7,x.numel(),dtype=x.dtype).reshape_as(x)
    finite=(law.velocity((x+eps*v).numpy(),t.numpy())-law.velocity((x-eps*v).numpy(),t.numpy()))/(2*eps)
    np.testing.assert_allclose(torch.einsum('bij,bj->bi',j,v).numpy(),finite,atol=1e-7,rtol=1e-5)
    class Constant(torch.nn.Module):
        def __init__(self,c):super().__init__();self.c=c
        def forward(self,x,t):return torch.full_like(x,self.c)
    for window in range(5):
        policy=WindowRepair(Constant(2),Constant(1),window,20)
        y=base.train.generated_samples(policy,torch.zeros_like(x),20)
        assert policy.calls==40 and torch.allclose(y,torch.full_like(x,1.8),atol=1e-12)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--data-root',required=True);p.add_argument('--output',required=True)
    group=p.add_mutually_exclusive_group();group.add_argument('--existing-only',action='store_true');group.add_argument('--new-only',action='store_true')
    p.add_argument('--steps',type=int,default=160);p.add_argument('--samples',type=int,default=64)
    a=p.parse_args();root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    assert a.steps%5==0;torch.set_num_threads(1);rows=[];reference_cache={}
    cases=json.loads((root/'anchors.json').read_text())['cases']
    if a.existing_only:cases=[c for c in cases if c['existing']]
    if a.new_only:cases=[c for c in cases if not c['existing']]
    for case in cases:
        started=time.monotonic();folder=Path(case['folder']);cfg=json.loads((folder/'config.json').read_text());cell=cfg['cell']
        cache=base.verify_data(a.data_root,cell);law=base.train.load_distribution(json.loads((cache/'manifest.json').read_text())['datasets'][cell['dataset']])
        model,identity=evaluation.evaluator.ev.restore(folder,cfg,cache/'manifest.json',20000,'cpu');model=model.double()
        arrays,_,binding=evaluation.evaluator.assets(root/'assets',cell['dataset'],'selection','cpu')
        source=torch.from_numpy(arrays['source_0'][:a.samples]).double();oracle=Oracle(law)
        if cell['dataset'] not in reference_cache:
            self_test(law,source);end,trace=reference(law,source,a.steps)
            fine=base.train.generated_samples(oracle,source,2*a.steps)
            reference_cache[cell['dataset']]=(end,trace,float((fine-end).square().mean()))
        truth,trace,oracle_refinement=reference_cache[cell['dataset']]
        z,mse=sensitivity(model,trace);total=z.sum(0)
        native=base.train.generated_samples(model,source,a.steps);delta=native-truth
        predicted=[float(total.square().mean()-(total-z[i]).square().mean()) for i in range(5)]
        predicted_order=np.argsort(-np.array(predicted)).tolist()
        # Freeze the ranking before observing any window-repair outcomes.
        base.dump(out/(case['id']+'-prediction.json'),dict(predicted_window_order=predicted_order,predicted_mse_reduction=predicted))
        repairs=[]
        for window in range(5):
            policy=WindowRepair(model,oracle,window,a.steps)
            y=base.train.generated_samples(policy,source,a.steps);assert policy.calls==2*a.steps
            repairs.append(float((y-truth).square().mean()))
        calibration=[]
        for weight in [.125,.25,.5,1.]:
            y=native if weight==1 else base.train.generated_samples(Mixture(model,oracle,weight),source,a.steps)
            residual=y-truth-weight*total
            calibration.append(dict(weight=weight,endpoint_mse=float((y-truth).square().mean()),
                sensitivity_residual_mse=float(residual.square().mean()),
                relative_rms_residual=float(residual.norm()/(y-truth).norm().clamp_min(1e-12))))
        fine=base.train.generated_samples(model,source,2*a.steps)
        observed=float(delta.square().mean())
        row=dict(id=case['id'],dataset=cell['dataset'],states=a.samples,heun_steps=a.steps,nfe=2*a.steps,
             field_mse_by_time_window=mse,transport_predicted_endpoint_mse=float(total.square().mean()),
             endpoint_mse=observed,predicted_window_order=predicted_order,
             predicted_reduction=predicted,repair_endpoint_mse=repairs,
             observed_reduction=[observed-v for v in repairs],calibration=calibration,
             oracle_solver_refinement_mse=oracle_refinement,model_solver_refinement_mse=float((fine-native).square().mean()),
             checkpoint=identity,asset_sha256=binding['sha256'],seconds=time.monotonic()-started)
        rows.append(row);base.dump(out/'progress.json',dict(completed=len(rows),total=len(cases),last=row))
    base.dump(out/'COMPLETE.json',dict(passed=True,scope='bounded frozen-checkpoint transport probes, not SW2 or training conclusions',rows=rows,source_sha256=base.digest(__file__)))
    print(json.dumps(dict(passed=True,cases=len(rows),output=str(out))))

if __name__=='__main__':main()
