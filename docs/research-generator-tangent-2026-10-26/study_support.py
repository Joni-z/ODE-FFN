"""Immutable native predictor, antithetic fields and full-network adjoints."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch

BASE=Path('/Users/mr.z/ode-workspace-prep/architecture-search')
MODELS=BASE/'conservative_readout_20261026/output'
PROBES=BASE/'coordinate_dependence_20261026/output-v2'
PARITY=BASE/'parity_20261026_v2/output'
ARM='ode_k2_direct';DATASET='rotated_product_bimodal_16d';TIMES=(.1,.3,.5,.7,.9)
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def build():
    provenance=load(MODELS/'provenance.json');old=Path(provenance['source_root'])
    for name,digest in provenance['source_sha256'].items():assert sha(old/name)==digest
    sys.path.insert(0,str(old))
    from synthetic.conservative_readout import build as original
    from synthetic.train import restore_model
    setting=load(MODELS/'study.json')['arms'][ARM]
    checkpoint=MODELS/'runs'/DATASET/ARM/'step_0005000.pt'
    model=original(16,setting['family'],setting['readout']).eval()
    restore_model(model,torch.load(checkpoint,map_location='cpu',weights_only=False))
    for parameter in model.parameters():parameter.requires_grad_(False)
    return model.raw,checkpoint,provenance

def layers(model):
    found=list(model.ode_layers());assert len(found)==6
    assert len({id(m.proj.weight) for _,m in found})==6
    for name,m in found:
        assert m.orders==2 and m.gate_source=='input' and m.spectral_cap is None and m.clock_curvature==1.
    return found

def native_check(model,checkpoint):
    count=0
    for t in TIMES:
        with np.load(MODELS/'banks'/DATASET/f'panel_{t:.1f}.npz') as b,np.load(checkpoint.parent/'diagnostics'/f'panel_{t:.1f}.npz') as wanted:
            with torch.enable_grad():value=model(torch.from_numpy(b['x']).requires_grad_(True),torch.from_numpy(b['t'])).detach().numpy()
            np.testing.assert_array_equal(value,wanted['prediction']);count+=1
    return count

def adjoint_chunks(model,t):
    """Yield actual inputs/durations/full adjoints and native P gradients."""
    rotation=np.array(load(PROBES/'population.json')['rotation'])
    with np.load(PROBES/'banks'/f'time-{t:.1f}.npz') as bank:x=bank['A']@rotation.T;target=bank['oracle']@rotation.T
    with np.load(PARITY/'fields'/ARM/f'time-{t:.1f}.npz') as d:plus=d['plus'];minus=d['minus']
    with np.load(PARITY/'projections'/f'time-{t:.1f}.npz') as d:coeff=d['coefficients'][0];wanted=float(d['nonlinear_odd'][0,0])
    error=(plus-minus)/2-target-x@(coeff[:,:16]-coeff[:,-16:])
    np.testing.assert_allclose(np.mean(error*error),wanted,atol=1e-12)
    assert np.max(np.abs(x.T@error/len(x)))<1e-11
    found=layers(model);parameters=[m.proj.weight for name,m in found]
    for p in parameters:p.requires_grad_(True)
    state={};hooks=[]
    for name,layer in found:
        def logit_hook(m,args,y,key=name,l=layer):state.setdefault(key,{})['a']=torch.sigmoid(y/l.tau)*l.scale+l.shift
        def layer_hook(m,args,y,key=name):state.setdefault(key,{}).update(x=args[0],y=y)
        hooks.append(layer.t_func[-1].register_forward_hook(logit_hook));hooks.append(layer.register_forward_hook(layer_hook))
    n,d=x.shape;normalizer=n*d*len(TIMES)
    try:
        for sign,saved in [(1,plus),(-1,minus)]:
            for start in range(0,n,256):
                sl=slice(start,start+256);xx=torch.from_numpy(np.ascontiguousarray(sign*x[sl]));tt=torch.full((len(xx),1),t,dtype=xx.dtype)
                state.clear();prediction=model(xx,tt)
                np.testing.assert_array_equal(prediction.detach().numpy(),saved[sl])
                outputs=[state[name]['y'] for name,_ in found]
                for label in ['total','nonlinear']:
                    adjoint=((saved[sl]-sign*target[sl]) if label=='total' else sign*error[sl])/normalizer
                    grads=torch.autograd.grad(prediction,parameters+outputs,grad_outputs=torch.from_numpy(np.ascontiguousarray(adjoint)),retain_graph=label=='total')
                    records={}
                    for j,(name,layer) in enumerate(found):
                        records[name]=dict(x=state[name]['x'].detach().numpy(),a=state[name]['a'].detach().numpy(),u=grads[len(found)+j].numpy(),
                            native=grads[j].numpy(),p=layer.proj.weight.detach().numpy())
                    yield label,sign,start,records
    finally:
        for hook in hooks:hook.remove()

def input_paths(checkpoint,provenance):
    paths=[MODELS/'provenance.json',MODELS/'study.json',checkpoint,checkpoint.parent/'result.json',
        PROBES/'population.json',PROBES/'standardized-draws.npz',PARITY/'audit.json',PARITY/'summary.json',PARITY/'decision.json']
    paths.extend(Path(provenance['source_root'])/name for name in provenance['source_sha256'])
    for t in TIMES:paths.extend([MODELS/'banks'/DATASET/f'panel_{t:.1f}.npz',checkpoint.parent/'diagnostics'/f'panel_{t:.1f}.npz',
        PROBES/'banks'/f'time-{t:.1f}.npz',PARITY/'fields'/ARM/f'time-{t:.1f}.npz',PARITY/'projections'/f'time-{t:.1f}.npz'])
    return paths
