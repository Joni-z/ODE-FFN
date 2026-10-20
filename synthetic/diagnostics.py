"""Endpoint and time-series diagnostics for internal ODE matrices and actual activations."""
import numpy as np
import torch


def spectrum(weight):
    p=weight.detach().float().cpu().numpy().astype(np.float64)
    singular=np.linalg.svd(p,compute_uv=False)
    eigen=np.linalg.eigvals(p)
    total=float(np.sum(singular**2))
    return {'sigma_max':float(singular[0]),'sigma_median':float(np.median(singular)),
            'sigma_min':float(singular[-1]),
            'sigma_quantiles':[float(a) for a in np.quantile(singular,[0,.1,.25,.5,.75,.9,1])],
            'stable_rank':float(total/(singular[0]**2)),
            'eig_real_max':float(eigen.real.max()),
            'eig_real_min':float(eigen.real.min()),
            'spectral_radius':float(np.max(np.abs(eigen))),
            'nonnormality':float(np.linalg.norm(p.T@p-p@p.T)/(total+1e-12))}


@torch.no_grad()
def collect(model,x,t):
    """Probe actual hidden inputs to both solves; x,t remain fixed across checkpoints."""
    if not list(model.ode_layers()):
        return []
    inputs={}
    hooks=[]
    for name,layer in model.ode_layers():
        hooks.append(layer.register_forward_pre_hook(lambda module,args,key=name:inputs.__setitem__(key,
            (args[0].detach(), args[1].detach() if len(args)>1 and args[1] is not None else None))))
    try:
        model(x,t)
    finally:
        for hook in hooks:
            hook.remove()
    records=[]
    for name,layer in model.ode_layers():
        raw=spectrum(layer.proj.weight)
        effective=spectrum(layer.operator())
        records.append({'layer':name,'raw':raw,'effective':effective,
                        'gate_source':layer.gate_source,
                        'activation':layer.activation_stats(*inputs[name])})
    return records
