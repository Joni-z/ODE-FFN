"""Probe trained K8 states; distinguish truncation from restarted generators."""
import argparse,copy,json,types
from pathlib import Path
from coefficient_trial import HERE,base,torch
import coefficient_evaluate as evaluation

def exponential(layer,h,s=1.):
    a=layer.operator()[None]*layer.step_size(h)[:,None,:]
    return (torch.matrix_exp(s*a)@h[:,:,None])[:,:,0]

def nonlinear(layer,h,steps=4):
    y=h;dt=1/steps;p=layer.operator()
    q=lambda x:x*torch.rsqrt(1+1.2*x.square())
    for _ in range(steps):
        mid=y+.5*dt*torch.nn.functional.linear(q(y),p)
        y=y+dt*torch.nn.functional.linear(q(mid),p)
    return y

def rms(x):return float(x.square().mean().sqrt())

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--data-root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);rows=[]
    cases=json.loads((root/'anchors.json').read_text())['cases']
    for case in [c for c in cases if c['existing'] and 'screen_isru_k8' in c['id']]:
        folder=Path(case['folder']);cfg=json.loads((folder/'config.json').read_text());cell=cfg['cell']
        cache=base.verify_data(a.data_root,cell);law=base.train.load_distribution(json.loads((cache/'manifest.json').read_text())['datasets'][cell['dataset']])
        model,identity=evaluation.evaluator.ev.restore(folder,cfg,cache/'manifest.json',20000,'cpu');model=model.double()
        arrays,_,binding=evaluation.evaluator.assets(root/'assets',cell['dataset'],'selection','cpu')
        source=torch.from_numpy(arrays['source_0'][:8]).double();target=torch.from_numpy(arrays['target_0'][:8]).double()
        refined=copy.deepcopy(model);feedback=copy.deepcopy(model)
        def exact_forward(layer,h,context=None,*,return_duration=False):
            value=exponential(layer,h)
            return (value,layer.step_size(h,context)) if return_duration else value
        def feedback_forward(layer,h,context=None,*,return_duration=False):
            value=nonlinear(layer,h)
            return (value,layer.step_size(h,context)) if return_duration else value
        for _,layer in refined.ode_layers():layer.forward=types.MethodType(exact_forward,layer)
        for _,layer in feedback.ode_layers():layer.forward=types.MethodType(feedback_forward,layer)
        for t0 in [.1,.3,.5,.7,.9]:
            x=(1-t0)*source+t0*target;t=x.new_full((len(x),1),t0);captures={};hooks=[]
            for name,layer in model.ode_layers():
                def hook(m,args,key=name):captures[key]=args[0].detach().clone()
                hooks.append(layer.register_forward_pre_hook(hook))
            with torch.no_grad():prediction=model(x,t)
            for hook in hooks:hook.remove()
            with torch.no_grad():
                layers=[]
                for name,layer in model.ode_layers():
                    h=captures[name];native=layer(h);exact=exponential(layer,h)
                    half=exponential(layer,h,.5);restart=exponential(layer,half,.5)
                    state=nonlinear(layer,h);state_ref=nonlinear(layer,h,16)
                    denom=max(rms(exact-h),1e-12)
                    layers.append(dict(layer=name,entrance_rms=rms(h),increment_rms=denom,
                        truncation_relative=rms(native-exact)/denom,
                        restarted_composition_relative=rms(restart-exact)/denom,
                        feedback_correction_relative=rms(state-exact)/denom,
                        midpoint4_vs16_relative=rms(state-state_ref)/denom))
                oracle=torch.from_numpy(law.velocity(x.numpy(),t.numpy()))
                exact_pred=refined(x,t);feedback_pred=feedback(x,t)
                row=dict(dataset=cell['dataset'],time=t0,states=len(x),layers=layers,
                    original_field_mse=float((prediction-oracle).square().mean()),
                    exact_exponential_field_mse=float((exact_pred-oracle).square().mean()),
                    feedback_field_mse=float((feedback_pred-oracle).square().mean()),
                    exact_prediction_shift=rms(exact_pred-prediction),feedback_prediction_shift=rms(feedback_pred-prediction),
                    checkpoint=identity,assets_sha256=binding['sha256'])
                rows.append(row);base.dump(out/'progress.json',dict(completed=len(rows),last=row))
    base.dump(out/'COMPLETE.json',dict(passed=True,scope='fixed-weight diagnostic, not retraining evidence',rows=rows,
               source_sha256=base.digest(__file__)))
    print(json.dumps(dict(passed=True,rows=len(rows),output=str(out))))

if __name__=='__main__':main()
