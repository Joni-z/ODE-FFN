"""Frozen operator-sharing diagnostics; native trained checkpoints stay unchanged."""
import argparse,copy,json,types
from pathlib import Path
import numpy as np
from sharing_trial import base,torch
import sharing_evaluate as evaluation
ev=evaluation.evaluator.ev

def q(x):return x*torch.rsqrt(1+1.2*x.square())
def drift(x,p):return torch.nn.functional.linear(q(x),p)
def rms(x):return float(x.square().mean().sqrt())
def matrices(layer):
    p=layer.operator()
    return [p]+[p+v for v in layer.stage_delta] if hasattr(layer,'stage_delta') else [p]*4
def solve(x,weights,dt=.25):
    y=x
    for p in weights:y=y+dt*drift(y+.5*dt*drift(y,p),p)
    return y
def bracket(x,a,b):
    derivative=torch.rsqrt(1+1.2*x.square()).pow(3)
    return torch.nn.functional.linear(derivative*drift(x,a),b)-torch.nn.functional.linear(derivative*drift(x,b),a)

def self_test():
    torch.manual_seed(0);x=torch.randn(7,5,dtype=torch.double);a=torch.randn(5,5,dtype=torch.double)*.2;b=torch.randn_like(a)*.2
    exact=bracket(x,a,b)
    _,left=torch.func.jvp(lambda z:drift(z,b),(x,),(drift(x,a),))
    _,right=torch.func.jvp(lambda z:drift(z,a),(x,),(drift(x,b),))
    error=float((exact-(left-right)).abs().max());assert error<1e-12
    errors=[]
    for eps in [.02,.01,.005]:
        delta=solve(x,[a,b],eps)-solve(x,[b,a],eps)
        errors.append(rms(delta/eps**2-exact)/max(rms(exact),1e-12))
    assert errors[-1]<errors[0]*.4 and errors[-1]<.02
    assert torch.equal(solve(x,[a,a]),solve(x,[a,a][::-1])) and torch.count_nonzero(bracket(x,a,a))==0
    return dict(passed=True,autograd_bracket_max_error=error,small_step_relative_errors=errors,shared_null_exact=True)

def replacement(layer,x,context=None,*,return_duration=False):
    weights=matrices(layer)
    if layer.probe_mode=='reverse':weights=weights[::-1]
    elif layer.probe_mode=='mean':weights=[torch.stack(weights).mean(0)]*4
    elif layer.probe_mode=='base':weights=[weights[0]]*4
    else:raise ValueError(layer.probe_mode)
    y=solve(x,weights)
    return (y,layer.step_size(x,context)) if return_duration else y

def gradients(model,fixed):
    params=[(n,p) for n,p in model.named_parameters() if p.requires_grad];rows=[]
    for offset in range(0,4096,512):
        s=slice(offset,offset+512);loss=(model(fixed['x'][s],fixed['t'][s])-fixed['y'][s]).square().mean()
        grads=torch.autograd.grad(loss,[p for _,p in params],allow_unused=True)
        common=0.;new=0.
        for (name,_),g in zip(params,grads):
            if g is None:continue
            amount=float(g.double().square().sum())
            if name.endswith('stage_delta'):new+=amount
            else:common+=amount
        norm=(common+new)**.5
        rows.append(dict(batch=offset//512,pair_mse=float(loss.detach()),common_gradient_norm=common**.5,
                         new_gradient_norm=new**.5,total_gradient_norm=norm,clip_coefficient=min(1.,1/max(norm,1e-12))))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--data-root');p.add_argument('--output');p.add_argument('--self-test',action='store_true');a=p.parse_args()
    torch.set_num_threads(1);check=self_test()
    if a.self_test:print(json.dumps(check));return
    assert a.root and a.data_root and a.output
    root=Path(a.root);out=Path(a.output);assert not out.exists();rows=[]
    for case in json.loads((root/'sharing-anchors.json').read_text())['cases']:
        folder=Path(case['folder']);cfg=json.loads((folder/'config.json').read_text());cell=cfg['cell']
        if cell['family'] not in ['state_midpoint4','stage_operator_midpoint4']:continue
        assert (folder/'ADAPTER_COMPLETE').exists() and cfg['steps']==20000
        cache=base.verify_data(a.data_root,cell);model,identity=ev.restore(folder,cfg,cache/'manifest.json',20000,'cpu')
        arrays,fixed,binding=evaluation.evaluator.assets(root/'sharing-assets',cell['dataset'],'selection','cpu')
        snapshot=ev.tensor_snapshot(model);prediction,mse=ev.field(model,fixed);interventions=[]
        for mode in ['reverse','mean','base']:
            changed=copy.deepcopy(model)
            for _,layer in changed.ode_layers():layer.probe_mode=mode;layer.forward=types.MethodType(replacement,layer)
            pred,cmse=ev.field(changed,fixed);shift=float(np.sqrt(np.mean((pred.astype(float)-prediction.astype(float))**2)))
            if cell['family']=='state_midpoint4':assert shift<1e-6
            interventions.append(dict(mode=mode,field_mse=cmse,prediction_shift_rms=shift))
        source=torch.from_numpy(arrays['source_0'][:16]);target=torch.from_numpy(arrays['target_0'][:16]);layers=[]
        for time in [.1,.3,.5,.7,.9]:
            x=(1-time)*source+time*target;t=x.new_full((len(x),1),time);captured={};hooks=[]
            for name,layer in model.ode_layers():
                def hook(module,args,key=name):captured[key]=args[0].detach().clone()
                hooks.append(layer.register_forward_pre_hook(hook))
            with torch.no_grad():model(x,t)
            for h in hooks:h.remove()
            with torch.no_grad():
                for name,layer in model.ode_layers():
                    entrance=captured[name];weights=matrices(layer)
                    assert torch.equal(solve(entrance,weights),layer(entrance)),(case['id'],name,'native formula')
                    h=entrance.double();weights=[w.detach().double() for w in weights];mean=torch.stack(weights).mean(0);native=solve(h,weights)
                    denominator=max(rms(native-h),1e-12);fbar=drift(h,mean);drift_scale=max(rms(fbar),1e-12)
                    variation=(sum(rms(drift(h,w)-fbar)**2 for w in weights)/4)**.5/drift_scale
                    brackets=[rms(bracket(h,weights[i],weights[j])) for i in range(4) for j in range(i+1,4)]
                    layers.append(dict(layer=name,time=time,matrix_relative_deviation=[float((w-mean).norm()/mean.norm().clamp_min(1e-12)) for w in weights],
                        stage_drift_relative_variation=variation,bracket_rms=brackets,increment_rms=denominator,
                        reverse_relative_shift=rms(solve(h,weights[::-1])-native)/denominator,
                        mean_relative_shift=rms(solve(h,[mean]*4)-native)/denominator,
                        base_relative_shift=rms(solve(h,[weights[0]]*4)-native)/denominator))
        grad=gradients(model,fixed);ev.same_tensors(model,snapshot)
        history=[json.loads(x) for x in (folder/'metrics.jsonl').read_text().splitlines()]
        curve=[{k:r.get(k) for k in ['step','train_pair_mse','validation_pair_mse','validation_oracle_mse','grad_norm_before_clip','wall_seconds']} for r in history]
        rows.append(dict(id=case['id'],dataset=cell['dataset'],family=cell['family'],parameters=model.parameter_count(),checkpoint=identity,
                         assets_sha256=binding['sha256'],field_mse=mse,interventions=interventions,layers=layers,gradient_probes=grad,training_curve=curve))
    assert len(rows)==4
    base.dump(out,dict(passed=True,self_test=check,scope='frozen-weight action/ordering sensitivity, not a retrained benefit',rows=rows,source_sha256=base.digest(__file__)))
    print(json.dumps(dict(passed=True,cases=len(rows),output=str(out))))

if __name__=='__main__':main()
