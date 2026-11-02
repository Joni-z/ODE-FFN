"""Frozen affine-PD case: change P spectra while preserving the computed map."""
import argparse,importlib,sys,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--save',required=True);a=p.parse_args()
source=Path(a.source);sys.path.insert(0,str(source/'autoresearch_20261103'))
from trial import base,build_model
base.verify_sources();torch.set_num_threads(1)
file=Path(a.checkpoint);folder=file.parents[1];cfg=json.loads((folder/'config.json').read_text());cell=cfg['cell'];assert cell['family']=='pd_mlp_affine'
manifest=json.loads((Path(cfg['data_root'])/'manifest.json').read_text());spec=manifest['datasets'][cell['dataset']]
checkpoint=torch.load(file,map_location='cpu',weights_only=False)
assert checkpoint['config_sha256']==base.digest(folder/'config.json')
model=build_model(spec['dim'],cell['family']);base.train.restore_model(model,checkpoint);model=model.double().eval()
fixed=base.train.fixed_eval_data(Path(cfg['data_root'])/cell['dataset'],base.train.load_distribution(spec),spec['dim'],0,torch.device('cpu'),4096,256,'test')
x,t=fixed['x'][:64].double(),fixed['t'][:64].double()
inputs={};hooks=[]
for name,layer in model.ode_layers():
    hooks.append(layer.register_forward_pre_hook(lambda m,ins,key=name:inputs.__setitem__(key,ins[0].detach().clone())))
with torch.no_grad():before=model(x,t).clone()
for hook in hooks:hook.remove()
records=[]
for name,layer in model.ode_layers():
    h=inputs[name];c=torch.logspace(-1,1,64,dtype=torch.double)
    p0=layer.proj.weight.detach().clone();d0=layer.step_size(h).detach().clone();a0=p0*d0[0][None,:]
    j0=torch.autograd.functional.jacobian(lambda u:layer(u[None])[0],h[0])
    with torch.no_grad():
        layer.proj.weight.div_(c[None,:])
        layer.t_func[-1].weight.mul_(c[:,None]);layer.t_func[-1].bias.mul_(c).add_(30*(c-1))
    p1=layer.proj.weight.detach();d1=layer.step_size(h).detach();a1=p1*d1[0][None,:]
    j1=torch.autograd.functional.jacobian(lambda u:layer(u[None])[0],h[0])
    records.append(dict(layer=name,initial_p_sigma=float(torch.linalg.svdvals(p0)[0]),changed_p_sigma=float(torch.linalg.svdvals(p1)[0]),conditional_generator_max_error=float((a0-a1).abs().max()),full_layer_jacobian_max_error=float((j0-j1).abs().max())))
with torch.no_grad():after=model(x,t)
error=float((before-after).abs().max())
assert error<1e-8 and all(r['conditional_generator_max_error']<1e-9 and r['full_layer_jacobian_max_error']<1e-8 for r in records)
result=dict(passed=True,checkpoint_step=checkpoint['step'],checkpoint_sha256=base.digest(file),model=cell['id'],calculation_dtype='float64',full_prediction_max_error=error,layers=records,
    interpretation='Frozen reparameterization identity; not retraining, not a quality improvement, and not generally realizable by the bounded-sigmoid parent.')
base.dump(a.save,result);print(json.dumps(result))
