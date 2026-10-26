"""Explicit raw-model traversal, with no hooks or calls to block/FFN forwards."""
import math
import numpy as np
import torch
from torch.nn import functional as F

def manual(model,x,t,arm):
    collected={};values=[];metadata={}
    def add(key,kind,**parts):
        first=next(iter(parts.values()));metadata[key]=dict(kind=kind,width=first.shape[-1])
        for part,value in parts.items():collected.setdefault(key,{}).setdefault(part,[]).append(value.numpy().copy())
    def solve(layer,z,prefix,index):
        assert layer.orders==2 and layer.gate_source=='input' and layer.spectral_cap is None
        assert layer.clock_curvature==1 and layer.operator_mode=='multiplicative'
        assert layer.rate_placement=='after_projection' and layer.forcing_mode=='none' and layer.solve_return=='map'
        pre=F.linear(z,layer.t_func[0].weight,layer.t_func[0].bias);hidden=F.silu(pre)
        add(prefix+f'clock{index}','silu',pre=pre,out=hidden)
        logit=F.linear(hidden,layer.t_func[2].weight,layer.t_func[2].bias)
        duration=torch.sigmoid(logit/layer.tau)*layer.scale+layer.shift
        add(prefix+f'duration{index}','clock',out=duration)
        total=z;term=z
        for order in range(1,3):
            term=F.linear(term,layer.proj.weight)*duration/order;total=total+term
        return total
    with torch.no_grad():
        for start in range(0,len(x),256):
            xx=torch.from_numpy(np.ascontiguousarray(x[start:start+256]));tt=torch.full((len(xx),1),t,dtype=xx.dtype)
            phases=tt*model.time.frequencies[None]*2*math.pi
            context=model.time.net(torch.cat((tt,torch.sin(phases),torch.cos(phases)),dim=1))
            h=F.linear(xx,model.input.weight,model.input.bias)
            for i,block in enumerate(model.blocks):
                prefix=f'b{i}.';scale,shift=F.linear(context,block.condition.weight,block.condition.bias).chunk(2,-1)
                z=F.layer_norm(h,block.norm.normalized_shape,block.norm.weight,block.norm.bias,block.norm.eps)*(1+scale)+shift
                if arm=='ode_k2_direct':
                    first=solve(block.ffn.ode1,z,prefix,1);middle=F.silu(first)
                    add(prefix+'middle','silu',pre=first,out=middle)
                    update=solve(block.ffn.ode2,middle,prefix,2)
                else:
                    gate,value=F.linear(z,block.ffn.w12.weight,block.ffn.w12.bias).chunk(2,-1)
                    activation=torch.sigmoid(gate) if arm=='sigmoid_glu128' else F.silu(gate)
                    kind='sigmoid' if arm=='sigmoid_glu128' else 'silu';product=activation*value
                    add(prefix+'gate',kind,pre=gate,out=activation)
                    add(prefix+'product','product_glu' if kind=='sigmoid' else 'product_swiglu',gate=gate,value=value,out=product)
                    update=F.linear(product,block.ffn.w3.weight,block.ffn.w3.bias)
                h=h+block.residual_scale*update
            norm=F.layer_norm(h,model.norm.normalized_shape,model.norm.weight,model.norm.bias,model.norm.eps)
            values.append(F.linear(norm,model.output.weight,model.output.bias).numpy())
    return np.concatenate(values),{k:{p:np.concatenate(v) for p,v in parts.items()} for k,parts in collected.items()},metadata
