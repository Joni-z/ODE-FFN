"""Separate complete forward algebra for endpoint/derivative replay."""
import torch
from torch.nn import functional as F

def forward(model,x,t,family):
    raw=model.raw;context=raw.time(t);h=F.linear(x,raw.input.weight,raw.input.bias)
    for b in raw.blocks:
        scale,shift=F.linear(context,b.condition.weight,b.condition.bias).chunk(2,-1)
        z=F.layer_norm(h,b.norm.normalized_shape,b.norm.weight,b.norm.bias,b.norm.eps)*(1+scale)+shift
        if family=='ode_k2':
            def solve(layer,state):
                pre=F.linear(state,layer.t_func[0].weight,layer.t_func[0].bias)
                logit=F.linear(F.silu(pre),layer.t_func[-1].weight,layer.t_func[-1].bias)
                a=torch.sigmoid(logit/layer.tau)*layer.scale+layer.shift
                first=F.linear(state,layer.proj.weight)*a
                second=F.linear(first,layer.proj.weight)*a/2
                return (state+first)+second
            update=solve(b.ffn.ode2,F.silu(solve(b.ffn.ode1,z)))
        else:
            gate,value=F.linear(z,b.ffn.w12.weight,b.ffn.w12.bias).chunk(2,-1)
            update=F.linear(F.silu(gate)*value,b.ffn.w3.weight,b.ffn.w3.bias)
        h=h+b.residual_scale*update
    z=F.layer_norm(h,raw.norm.normalized_shape,raw.norm.weight,raw.norm.bias,raw.norm.eps)
    return F.linear(z,raw.output.weight,raw.output.bias)
