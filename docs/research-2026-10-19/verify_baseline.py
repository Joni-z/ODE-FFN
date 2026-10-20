"""Regression checks for unchanged legacy models and official SwiGLU arithmetic."""
import importlib.util
import json
from pathlib import Path
import sys
import torch
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from synthetic.models import VelocityNet, SwiGLUFFN
spec=importlib.util.spec_from_file_location('legacy_models',Path(__file__).with_name('models-before-swiglu.py'))
old=importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
for kind in ('ffn','ode_original','ode_spectral_cap'):
    for order in (1,2):
        torch.manual_seed(0)
        a=old.VelocityNet(2,kind,orders=order)
        torch.manual_seed(0)
        b=VelocityNet(2,kind,orders=order)
        assert a.state_dict().keys()==b.state_dict().keys()
        for key,value in a.state_dict().items():
            torch.testing.assert_close(value,b.state_dict()[key],atol=0,rtol=0)
        x=torch.randn(8,2); t=torch.rand(8,1)
        # Compare hidden blocks as the zero output initialization would mask regressions.
        ha=a.input(x); hb=b.input(x)
        for ba,bb in zip(a.blocks,b.blocks):
            ha=ba(ha,a.time(t)); hb=bb(hb,b.time(t))
            torch.testing.assert_close(ha,hb,atol=0,rtol=0)
layer=SwiGLUFFN(64,128).double()
x=torch.randn(7,64,dtype=torch.float64,requires_grad=True)
g,v=layer.w12(x).chunk(2,-1)
expected=(torch.nn.functional.silu(g)*v)@layer.w3.weight.T+layer.w3.bias
torch.testing.assert_close(layer(x),expected)
layer(x).square().mean().backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in layer.parameters())
counts={k:VelocityNet(2,k).parameter_count() for k in ('ffn','swiglu','ode_original')}
assert abs(counts['swiglu']/counts['ode_original']-1)<0.005
print(json.dumps({'legacy_initialization_and_hidden_outputs':'bitwise equal for all six kind/order combinations','swiglu_gradients':'finite and nonzero','parameter_counts':counts},indent=2))
