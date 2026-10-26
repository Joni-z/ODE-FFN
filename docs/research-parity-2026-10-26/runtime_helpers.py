"""Immutable predictor reconstruction and actual forward activation capture."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

BASE=Path('/Users/mr.z/ode-workspace-prep/architecture-search')
MODELS=BASE/'conservative_readout_20261026/output'
PROBES=BASE/'coordinate_dependence_20261026/output-v2'
GLU=BASE/'sigmoid_glu_20261025_m1/output/sigmoid_glu_product16_20261025_m1'
ARMS=('ode_k2_direct','swiglu128_direct','swiglu192_direct','swiglu384_direct','sigmoid_glu128')
TIMES=(.1,.3,.5,.7,.9);DATASET='rotated_product_bimodal_16d'
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def setup():
    provenance=load(MODELS/'provenance.json');old=Path(provenance['source_root'])
    for name,digest in provenance['source_sha256'].items():assert sha(old/name)==digest
    sys.path.insert(0,str(old));return provenance

def glu_checkpoint(label,kind):
    cfg=load(GLU/'configs'/f'{label}.json')
    return cfg,GLU/'runs'/cfg['experiment_id']/DATASET/kind/'seed000/checkpoints/step_0005000.pt'

def check_alias():
    _,path=glu_checkpoint('swiglu','swiglu');old=torch.load(path,map_location='cpu',weights_only=False)
    current=torch.load(MODELS/'runs'/DATASET/'swiglu128_direct/step_0005000.pt',map_location='cpu',weights_only=False)
    for group in ['model','runtime_buffers']:
        assert {'raw.'+k for k in old[group]}==set(current[group])
        for key,value in old[group].items():assert torch.equal(value,current[group]['raw.'+key]),key
    return dict(status='PASS',alias='historical paired Swi128 is current Swi128',historical_checkpoint=str(path),historical_sha256=sha(path))

def build(arm):
    from synthetic.models import VelocityNet
    from synthetic.conservative_readout import build as original
    from synthetic.train import restore_model
    if arm=='sigmoid_glu128':
        cfg,cp=glu_checkpoint('sigmoid_glu','glu');model=VelocityNet(16,'glu',**cfg['model_options']).eval()
        restore_model(model,torch.load(cp,map_location='cpu',weights_only=False));raw=model
    else:
        setting=load(MODELS/'study.json')['arms'][arm];cp=MODELS/'runs'/DATASET/arm/'step_0005000.pt'
        model=original(16,setting['family'],setting['readout']).eval()
        restore_model(model,torch.load(cp,map_location='cpu',weights_only=False));raw=model.raw
    for p in raw.parameters():p.requires_grad_(False)
    return raw,cp

def capture(model,x,t,arm):
    data={};outputs=[];hooks=[];metadata={}
    def append(key,part,value):data.setdefault(key,{}).setdefault(part,[]).append(value.detach().numpy().copy())
    for i,block in enumerate(model.blocks):
        prefix=f'b{i}.'
        if arm=='ode_k2_direct':
            modules=[('clock1',block.ffn.ode1.t_func[1]),('middle',block.ffn.act),('clock2',block.ffn.ode2.t_func[1])]
            for name,module in modules:
                key=prefix+name;metadata[key]=dict(kind='silu',width=64)
                hooks.append(module.register_forward_pre_hook(lambda m,a,k=key:append(k,'pre',a[0])))
                hooks.append(module.register_forward_hook(lambda m,a,y,k=key:append(k,'out',y)))
            for j,layer in enumerate([block.ffn.ode1,block.ffn.ode2]):
                key=prefix+f'duration{j+1}';metadata[key]=dict(kind='clock',width=1)
                def clock_hook(m,a,y,k=key,l=layer):append(k,'out',torch.sigmoid(y/l.tau)*l.scale+l.shift)
                hooks.append(layer.t_func[2].register_forward_hook(clock_hook))
        else:
            kind='sigmoid' if arm=='sigmoid_glu128' else 'silu';width=block.ffn.w3.in_features
            key=prefix+'gate';prodkey=prefix+'product';metadata[key]=dict(kind=kind,width=width)
            metadata[prodkey]=dict(kind='product_glu' if kind=='sigmoid' else 'product_swiglu',width=width)
            def projected(m,a,y,k=key,p=prodkey,activation=kind):
                gate,value=y.chunk(2,-1);out=torch.sigmoid(gate) if activation=='sigmoid' else F.silu(gate)
                append(k,'pre',gate);append(k,'out',out);append(p,'gate',gate);append(p,'value',value)
                append(p,'reconstructed',out*value)
            hooks.append(block.ffn.w12.register_forward_hook(projected))
            hooks.append(block.ffn.w3.register_forward_pre_hook(lambda m,a,k=prodkey:append(k,'out',a[0])))
    try:
        with torch.no_grad():
            for start in range(0,len(x),256):
                xx=torch.from_numpy(np.ascontiguousarray(x[start:start+256]));tt=torch.full((len(xx),1),t,dtype=xx.dtype)
                outputs.append(model(xx,tt).numpy())
    finally:
        for hook in hooks:hook.remove()
    arrays={key:{part:np.concatenate(values) for part,values in parts.items()} for key,parts in data.items()}
    for key,parts in arrays.items():
        info=metadata[key]
        if info['kind'].startswith('product'):
            # Preserve the live strided gate layout: a contiguous recalc can
            # choose a different vector kernel and differ by one float64 ULP.
            np.testing.assert_array_equal(parts.pop('reconstructed'),parts['out'])
    return np.concatenate(outputs),arrays,metadata
