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
    if hasattr(layer,'stage_weights'):return list(layer.stage_weights().unbind(0))
    p=layer.operator()
    return [p]+[p+v for v in layer.stage_delta] if hasattr(layer,'stage_delta') else [p]*4
def solve(x,weights,dt=.25):
    y=x
    for p in weights:y=y+dt*drift(y+.5*dt*drift(y,p),p)
    return y
def bracket(x,a,b):
    derivative=torch.rsqrt(1+1.2*x.square()).pow(3)
    return torch.nn.functional.linear(derivative*drift(x,a),b)-torch.nn.functional.linear(derivative*drift(x,b),a)

def mean_flow_expansion(x: torch.Tensor, weights: list, horizons=(1.,.5,.25,.125,.0625),capture: dict | None=None) -> dict:
    """Four-stage minus equal-mean local map; no field or sampler replacement."""
    assert len(weights)==4
    mean=torch.stack(weights).mean(0)
    leading=sum(bracket(x,weights[i],weights[j]) for i in range(4) for j in range(i+1,4))/32
    if capture is not None:
        capture.update(entrance=x.numpy(),weights=torch.stack(weights).numpy(),leading=leading.numpy())
    rows=[]
    for tau in horizons:
        native=solve(x,weights,tau/4)
        difference=native-solve(x,[mean]*4,tau/4)
        reverse=native-solve(x,weights[::-1],tau/4)
        prediction=tau*tau*leading
        if capture is not None:
            capture[f'tau{len(rows)}_mean_difference']=difference.numpy()
            capture[f'tau{len(rows)}_reverse_difference']=reverse.numpy()
        scale=rms(difference);pred_scale=rms(prediction)
        cosine=float((difference*prediction).sum()/(difference.norm()*prediction.norm())) if scale>1e-14 and pred_scale>1e-14 else None
        rows.append(dict(tau=tau,mean_difference_rms=scale,reverse_difference_rms=rms(reverse),
            predicted_mean_rms=pred_scale,mean_remainder_rms=rms(difference-prediction),
            reverse_remainder_rms=rms(reverse-2*prediction),
            relative_mean_prediction_error=rms(difference-prediction)/max(scale,1e-14),prediction_cosine=cosine))
    return dict(leading_coefficient_rms=rms(leading),rows=rows)

def expansion_checks() -> dict:
    """Independently check coefficient/sign with exact linear flow and JVPs."""
    torch.manual_seed(0)
    x=torch.randn(7,5,dtype=torch.double)
    weights=[torch.randn(5,5,dtype=torch.double)*.2 for _ in range(4)]
    mean=torch.stack(weights).mean(0)
    prediction=sum(torch.nn.functional.linear(x,b@a-a@b) for i,a in enumerate(weights) for b in weights[i+1:])/32
    linear=[]
    for tau in [.2,.1,.05,.025]:
        value=x
        for w in weights:value=torch.nn.functional.linear(value,torch.matrix_exp(tau*w/4))
        difference=value-torch.nn.functional.linear(x,torch.matrix_exp(tau*mean))
        linear.append(dict(tau=tau,remainder_rms=rms(difference-tau*tau*prediction),
            relative_prediction_error=rms(difference-tau*tau*prediction)/rms(difference)))
    observed=float(np.log2(linear[-2]['remainder_rms']/linear[-1]['remainder_rms']))
    assert 2.8<observed<3.2 and linear[-1]['relative_prediction_error']<.05
    nonlinear=mean_flow_expansion(x,weights,[.2,.1,.05,.025])
    obs=float(np.log2(nonlinear['rows'][-2]['mean_remainder_rms']/nonlinear['rows'][-1]['mean_remainder_rms']))
    assert 2.8<obs<3.2 and nonlinear['rows'][-1]['relative_mean_prediction_error']<.05
    shared=mean_flow_expansion(x,[mean]*4)
    assert shared['leading_coefficient_rms']==0 and all(r['mean_difference_rms']==0 and r['reverse_difference_rms']==0 for r in shared['rows'])
    scalar=mean_flow_expansion(x,[v*mean for v in [.5,1.5,.75,1.25]],[.2,.1,.05,.025])
    assert scalar['leading_coefficient_rms']<1e-14
    scalar_order=float(np.log2(scalar['rows'][-2]['mean_difference_rms']/scalar['rows'][-1]['mean_difference_rms']))
    assert 2.8<scalar_order<3.2
    return dict(passed=True,linear_matrix_exponential=linear,linear_remainder_order=observed,
        nonlinear_midpoint=nonlinear,nonlinear_remainder_order=obs,shared_null_exact=True,
        scalar_clock=scalar,scalar_remainder_order=scalar_order)

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
    return dict(passed=True,autograd_bracket_max_error=error,small_step_relative_errors=errors,shared_null_exact=True,
        four_stage_expansion=expansion_checks())

def replacement(layer,x,context=None,*,return_duration=False):
    weights=matrices(layer)
    if layer.probe_mode=='reverse':weights=weights[::-1]
    elif layer.probe_mode=='mean':weights=[torch.stack(weights).mean(0)]*4
    elif layer.probe_mode=='base':weights=[layer.operator()]*4
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
            if name.endswith(('stage_delta', 'stage_offset')):new+=amount
            else:common+=amount
        norm=(common+new)**.5
        rows.append(dict(batch=offset//512,pair_mse=float(loss.detach()),common_gradient_norm=common**.5,
                         new_gradient_norm=new**.5,total_gradient_norm=norm,clip_coefficient=min(1.,1/max(norm,1e-12))))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--data-root');p.add_argument('--output');p.add_argument('--contrast-study');p.add_argument('--schedule-study');p.add_argument('--arrays-output');p.add_argument('--self-test',action='store_true');a=p.parse_args()
    assert not (a.contrast_study and a.schedule_study)
    torch.set_num_threads(1);check=self_test()
    if a.self_test:print(json.dumps(check));return
    base.verify_sources()
    assert a.root and a.data_root and a.output
    root=Path(a.root);out=Path(a.output);assert not out.exists();rows=[]
    raw=Path(a.arrays_output) if a.arrays_output else None
    if a.contrast_study or a.schedule_study:assert raw is not None,'Retain the actual entrances and matrices for independent reconstruction'
    if raw:raw.mkdir(parents=True,exist_ok=False)
    cases=json.loads((root/'sharing-anchors.json').read_text())['cases']
    if a.contrast_study:
        from contrast_trial import build_model as contrast_builder
        ev.build_model=contrast_builder
        registered=json.loads((Path(a.contrast_study)/'anchors.json').read_text())
        assert registered['passed'] and len(registered['new_cases'])==2
        cases=cases+registered['new_cases']
    families = ['state_midpoint4','stage_operator_midpoint4','contrast_operator_midpoint4']
    if a.schedule_study:
        from gpu_resolution import builder
        ev.build_model=builder
        registered=json.loads((Path(a.schedule_study)/'anchors.json').read_text())
        assert registered['passed'] and registered['protocol']=='schedule_order' and len(registered['new_cases'])==4
        cases=cases+registered['new_cases']
        families=['state_midpoint4','palindrome_operator_midpoint4','alternating_operator_midpoint4']
    for case in cases:
        folder=Path(case['folder']);cfg=json.loads((folder/'config.json').read_text());cell=cfg['cell']
        if cell['family'] not in families:continue
        assert (folder/'ADAPTER_COMPLETE').exists() and cfg['steps']==20000
        cache=base.verify_data(a.data_root,cell);model,identity=ev.restore(folder,cfg,cache/'manifest.json',20000,'cpu')
        arrays,fixed,binding=evaluation.evaluator.assets(root/'sharing-assets',cell['dataset'],'selection','cpu')
        snapshot=ev.tensor_snapshot(model);prediction,mse=ev.field(model,fixed);interventions=[]
        captured_arrays=dict(field_original=prediction,field_oracle=fixed['oracle'].cpu().numpy())
        for mode in ['reverse','mean','base']:
            changed=copy.deepcopy(model)
            for _,layer in changed.ode_layers():layer.probe_mode=mode;layer.forward=types.MethodType(replacement,layer)
            pred,cmse=ev.field(changed,fixed);shift=float(np.sqrt(np.mean((pred.astype(float)-prediction.astype(float))**2)))
            captured_arrays['field_'+mode]=pred
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
                    local_capture={} if raw else None
                    expansion=mean_flow_expansion(h,weights,capture=local_capture)
                    prefix=f'entry{len(layers)}_'
                    if raw:captured_arrays.update({prefix+k:v for k,v in local_capture.items()})
                    layers.append(dict(layer=name,time=time,matrix_relative_deviation=[float((w-mean).norm()/mean.norm().clamp_min(1e-12)) for w in weights],
                        stage_drift_relative_variation=variation,bracket_rms=brackets,increment_rms=denominator,
                        reverse_relative_shift=rms(solve(h,weights[::-1])-native)/denominator,
                        mean_relative_shift=rms(solve(h,[mean]*4)-native)/denominator,
                        base_relative_shift=rms(solve(h,[layer.operator().detach().double()]*4)-native)/denominator,
                        mean_flow_expansion=expansion,capture_prefix=prefix if raw else None))
        grad=gradients(model,fixed);ev.same_tensors(model,snapshot)
        history=[json.loads(x) for x in (folder/'metrics.jsonl').read_text().splitlines()]
        curve=[{k:r.get(k) for k in ['step','train_pair_mse','validation_pair_mse','validation_oracle_mse','grad_norm_before_clip','wall_seconds']} for r in history]
        rows.append(dict(id=case['id'],dataset=cell['dataset'],family=cell['family'],parameters=model.parameter_count(),checkpoint=identity,
                         source_folder=str(folder),dataset_manifest=str(cache/'manifest.json'),
                         assets_sha256=binding['sha256'],field_mse=mse,interventions=interventions,layers=layers,gradient_probes=grad,training_curve=curve))
        if raw:
            file=raw/(case['id']+'.npz');np.savez_compressed(file,**captured_arrays)
            rows[-1]['arrays']=dict(file=str(file),sha256=base.digest(file))
            base.dump(raw/'progress.json',dict(completed=len(rows),last=case['id']))
    assert len(rows)==(6 if a.contrast_study or a.schedule_study else 4)
    base.dump(out,dict(passed=True,self_test=check,scope='frozen-weight action/ordering sensitivity, not a retrained benefit',rows=rows,source_sha256=base.digest(__file__)))
    print(json.dumps(dict(passed=True,cases=len(rows),output=str(out))))

if __name__=='__main__':main()
