"""Matched time-conditioned vector-field backbones for synthetic FM.

The ODEFFN below reproduces the professor's two ODELayer solves separated by SiLU.
The external residual/time conditioning is shared across all compared FFNs.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class OriginalODELayer(nn.Module):
    def __init__(self, width, gate_hidden, orders=2, tau=10.0, scale=0.8,
                 shift=0.2, spectral_cap=None, gate_source='input'):
        super().__init__()
        if orders < 1 or tau <= 0 or scale <= 0 or not 0 <= shift < shift + scale:
            raise ValueError('invalid ODELayer settings')
        if spectral_cap is not None and spectral_cap <= 0:
            raise ValueError('spectral_cap must be positive')
        self.orders, self.tau, self.scale, self.shift = orders, tau, scale, shift
        self.spectral_cap = spectral_cap
        if gate_source not in ('input', 'time', 'fixed', 'scalar'):
            raise ValueError('gate_source must be input, time, fixed or scalar')
        self.gate_source = gate_source
        self.t_func = nn.Sequential(nn.Linear(width, gate_hidden), nn.SiLU(),
                                    nn.Linear(gate_hidden, 1))
        self.proj = nn.Linear(width, width, bias=False)
        # Consume the legacy initialization draws so all common weights stay paired.
        # Remove the unused network rather than counting dormant parameters.
        if gate_source in ('fixed', 'scalar'):
            del self.t_func
        if gate_source == 'scalar':
            self.duration_logit = nn.Parameter(torch.zeros(()))

    def operator(self):
        weight = self.proj.weight
        if self.spectral_cap is None:
            return weight
        # Exact differentiable spectral-norm cap. Small (64x64) experiment only.
        sigma = torch.linalg.matrix_norm(weight.float(), ord=2)
        factor = torch.clamp(self.spectral_cap / sigma.clamp_min(1e-12), max=1.0)
        return weight * factor.to(weight.dtype)

    def step_size(self, x, context=None):
        if self.gate_source == 'fixed':
            return x.new_full((*x.shape[:-1], 1), self.shift + self.scale / 2)
        if self.gate_source == 'scalar':
            s = torch.sigmoid(self.duration_logit / self.tau) * self.scale + self.shift
            return s.expand(*x.shape[:-1], 1)
        gate_input = x
        if self.gate_source == 'time':
            if context is None or context.shape != x.shape:
                raise ValueError('time gate requires a matching time embedding')
            gate_input = context
        return torch.sigmoid(self.t_func(gate_input) / self.tau) * self.scale + self.shift

    def forward(self, x, context=None):
        s = self.step_size(x, context)
        p = self.operator()
        total, term = x, x
        for order in range(1, self.orders + 1):
            term = F.linear(term, p) * s / order
            total = total + term
        return total

    @torch.no_grad()
    def activation_stats(self, x, context=None):
        s = self.step_size(x, context)
        p = self.operator()
        rms = lambda a: a.square().mean().sqrt().item()
        denominator = max(rms(x), 1e-12)
        term, out = x, x
        stats = {'s_mean':s.mean().item(), 's_std':s.std(unbiased=False).item(),
                 's_min':s.min().item(), 's_max':s.max().item(),
                 'input_rms':denominator}
        for order in range(1,self.orders+1):
            term = F.linear(term,p)*s/order
            stats[f'term{order}_rms_over_input'] = rms(term)/denominator
            out = out+term
        stats['output_rms_over_input'] = rms(out)/denominator
        return stats


class OriginalODEFFN(nn.Module):
    def __init__(self, width, gate_hidden, **ode_options):
        super().__init__()
        self.gate_source = ode_options.get('gate_source', 'input')
        self.ode1 = OriginalODELayer(width,gate_hidden,**ode_options)
        self.act = nn.SiLU()
        self.ode2 = OriginalODELayer(width,gate_hidden,**ode_options)

    def forward(self, x, context=None):
        return self.ode2(self.act(self.ode1(x, context)), context)


class StandardFFN(nn.Module):
    def __init__(self,width,hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width,hidden),nn.SiLU(),nn.Linear(hidden,width))

    def forward(self,x):
        return self.net(x)


class SwiGLUFFN(nn.Module):
    """JiT-style SwiGLU, retaining the nominal 2/3 hidden-width convention.

    Only the FFN is ported; the toy backbone and initialization stay unchanged.
    Source: https://github.com/LTH14/JiT/blob/main/model_jit.py
    """
    def __init__(self, width, hidden):
        super().__init__()
        hidden = int(hidden * 2 / 3)
        if hidden < 1:
            raise ValueError('SwiGLU hidden width must be positive')
        self.w12 = nn.Linear(width, 2 * hidden)
        self.w3 = nn.Linear(hidden, width)

    def forward(self, x):
        gate, value = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(gate) * value)


class TimeEmbedding(nn.Module):
    def __init__(self,width):
        super().__init__()
        frequencies = torch.exp(torch.linspace(math.log(1),math.log(1000),16))
        self.register_buffer('frequencies',frequencies,persistent=False)
        self.net = nn.Sequential(nn.Linear(33,width),nn.SiLU(),nn.Linear(width,width))

    def forward(self,t):
        phases = t*self.frequencies[None]*2*math.pi
        return self.net(torch.cat((t,torch.sin(phases),torch.cos(phases)),dim=1))


class TimeConditionedBlock(nn.Module):
    def __init__(self,width,time_width,ffn,residual_scale):
        super().__init__()
        self.norm=nn.LayerNorm(width)
        self.condition=nn.Linear(time_width,width*2)
        self.ffn=ffn
        self.residual_scale=residual_scale

    def forward(self,x,temb):
        scale,shift=self.condition(temb).chunk(2,dim=-1)
        z=self.norm(x)*(1+scale)+shift
        if isinstance(self.ffn, OriginalODEFFN) and self.ffn.gate_source == 'time':
            return x+self.residual_scale*self.ffn(z, temb)
        return x+self.residual_scale*self.ffn(z)


class VelocityNet(nn.Module):
    def __init__(self,dim,kind,width=64,depth=3,gate_hidden=64,
                 ffn_hidden=128,orders=2,tau=10.0,scale=0.8,shift=0.2,
                 spectral_cap=1.0):
        super().__init__()
        if kind not in ('ffn','swiglu','ode_original','ode_spectral_cap','ode_time',
                        'ode_fixed','ode_scalar'):
            raise ValueError(f'unsupported model kind: {kind}')
        self.dim,self.kind,self.width,self.depth=dim,kind,width,depth
        self.input=nn.Linear(dim,width)
        self.time=TimeEmbedding(width)
        blocks=[]
        for _ in range(depth):
            if kind=='ffn':
                ffn=StandardFFN(width,ffn_hidden)
            elif kind=='swiglu':
                ffn=SwiGLUFFN(width,ffn_hidden)
            else:
                ffn=OriginalODEFFN(width,gate_hidden,orders=orders,tau=tau,
                                   scale=scale,shift=shift,
                                   gate_source={'ode_time':'time', 'ode_fixed':'fixed',
                                                'ode_scalar':'scalar'}.get(kind, 'input'),
                                   spectral_cap=spectral_cap if kind=='ode_spectral_cap' else None)
            blocks.append(TimeConditionedBlock(width,width,ffn,1/math.sqrt(depth)))
        self.blocks=nn.ModuleList(blocks)
        self.norm=nn.LayerNorm(width)
        self.output=nn.Linear(width,dim)
        # Exactly zero initial predicted velocity for every architecture.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self,x,t):
        if x.ndim != 2 or x.shape[1] != self.dim or t.shape != (len(x),1):
            raise ValueError('expected x=[batch,dim], t=[batch,1]')
        temb=self.time(t)
        h=self.input(x)
        for block in self.blocks:
            h=block(h,temb)
        return self.output(self.norm(h))

    def ode_layers(self):
        for name,module in self.named_modules():
            if isinstance(module,OriginalODELayer):
                yield name,module

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
