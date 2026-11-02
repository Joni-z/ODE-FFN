"""Frozen P-only native-Muon coordinate diagnostic; no actual run is updated."""
import argparse,copy,json,sys
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--folder',required=True);p.add_argument('--save',required=True);a=p.parse_args()
sys.path.insert(0,str(Path(a.source)/'autoresearch_20261103'))
from trial import base,build_model
base.verify_sources();torch.set_num_threads(1);folder=Path(a.folder);cfg=json.loads((folder/'config.json').read_text());cell=cfg['cell'];assert cell['family']=='pd_mlp_affine' and cell['optimizer']=='muon'
file=folder/'checkpoints/step_0020000.pt';checkpoint=torch.load(file,map_location='cpu',weights_only=False);assert checkpoint['config_sha256']==base.digest(folder/'config.json')
manifest=json.loads((Path(cfg['data_root'])/'manifest.json').read_text());spec=manifest['datasets'][cell['dataset']]
fixed=base.train.fixed_eval_data(Path(cfg['data_root'])/cell['dataset'],base.train.load_distribution(spec),spec['dim'],0,torch.device('cpu'),4096,256,'validation')
x,t,y=[fixed[k][:512].double() for k in ['x','t','y']]
records=[];reference=None;coefficient=3.
for mode in ['original','rescaled_same_lr','rescaled_compensated_lr']:
    torch.manual_seed(0);model=build_model(spec['dim'],cell['family']).double();base.train.restore_model(model,checkpoint);model.eval()
    opt=base.ProjectionOptimizer(model,'muon',cell['projection_lr']);opt.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
    if mode!='original':
        with torch.no_grad():
            for _,layer in model.ode_layers():
                layer.proj.weight.div_(coefficient);layer.t_func[-1].weight.mul_(coefficient);layer.t_func[-1].bias.mul_(coefficient).add_(30*(coefficient-1))
                opt.projections.state[layer.proj.weight]['momentum_buffer'].mul_(coefficient)
        if mode=='rescaled_compensated_lr':
            for group in opt.projections.param_groups:group['lr']/=coefficient
    captures={};hooks=[]
    for name,layer in model.ode_layers():hooks.append(layer.register_forward_pre_hook(lambda m,ins,key=name:captures.__setitem__(key,ins[0].detach().clone())))
    prediction=model(x,t);loss=(prediction-y).square().mean()
    for hook in hooks:hook.remove()
    gradients=torch.autograd.grad(loss,[p for _,p in opt.selected]);before={n:p.detach().clone() for n,p in opt.selected}
    for (_,p),gradient in zip(opt.selected,gradients):p.grad=gradient.detach().clone()
    durations={name:layer.step_size(captures[name][:8]).detach().clone() for name,layer in model.ode_layers()}
    pre_prediction=prediction.detach().clone();opt.projections.step()
    deltas={n:p.detach()-before[n] for n,p in opt.selected}
    energy=0.
    for name,layer in model.ode_layers():
        delta=deltas[name+'.proj.weight'];d=durations[name]
        energy+=float(sum((delta*row[None,:]).square().sum() for row in d))
    with torch.no_grad():after=model(x,t)
    if reference is None:
        reference=dict(prediction=pre_prediction,gradients=[g.detach().clone() for g in gradients],deltas=deltas,energy=energy,after=after)
        error=0.;covariance=0.;cosines=[1.]
    else:
        error=float((pre_prediction-reference['prediction']).abs().max())
        covariance=max(float((g/coefficient-original).norm()/original.norm().clamp_min(1e-30)) for g,original in zip(gradients,reference['gradients']))
        cosines=[float(torch.nn.functional.cosine_similarity(d.flatten(),reference['deltas'][n].flatten(),dim=0)) for n,d in deltas.items()]
        assert error<1e-9 and covariance<1e-7
    records.append(dict(mode=mode,pre_step_prediction_max_error=error,gradient_covariance_max_relative_error=covariance,conditional_update_norm_ratio=(energy/reference['energy'])**.5,mean_parameter_update_cosine=sum(cosines)/len(cosines),post_step_prediction_rms_change=float((after-pre_prediction).square().mean().sqrt()),post_step_prediction_max_difference_from_original=float((after-reference['after']).abs().max())))
result=dict(checkpoint_sha256=base.digest(file),model=cell['id'],rescaling=coefficient,passed=True,torch=torch.__version__,rows=records,scope='Copied frozen checkpoint, held validation pair loss, auxiliary parameters frozen, clipping omitted, P-only step using actual pinned five-BF16-NS Muon with covariantly transformed trained momentum; not end-to-end training or a quality claim.')
base.dump(a.save,result);print(json.dumps(result))
