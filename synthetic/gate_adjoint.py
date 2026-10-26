"""Paired gate-sensitivity pathway removal for a cold mechanistic experiment."""
from __future__ import annotations

import torch

from synthetic.conservative_readout import build as parent_build
from synthetic.explicit_adjoint import ExplicitAdjointNet
from synthetic.models import OriginalODELayer


class GateAdjointNet(ExplicitAdjointNet):
    def __init__(self,raw,readout):
        if readout not in ('gradient','gate_omitted'):
            raise ValueError('unsupported gate-path readout')
        super().__init__(raw,include_clock=readout=='gradient',include_swiglu_gate=readout=='gradient')
        self.readout=readout
        self.mode='native'

    def forward(self,x,t,*,updates=3,step_size=None):
        if updates!=3 or step_size is not None:
            raise ValueError('preserve three native blocks')
        return super().forward(x,t)

    def ode_layers(self):
        for name,module in self.named_modules():
            if type(module) is OriginalODELayer:yield name,module


def build(dim,family,readout):
    if family not in ('ode_k2','swiglu128','swiglu192'):
        raise ValueError('unsupported family')
    return GateAdjointNet(parent_build(dim,family,'direct').raw,readout)


@torch.no_grad()
def visit_diagnostics(model,banks):
    """Capture actual scalar-gate inputs from the explicit forward sweep."""
    import numpy as np
    visits=[];hooks=[]
    for name,layer in model.ode_layers():
        def capture(module,args,*,name=name,layer=layer):
            visits.append((name,layer,args[0].detach().clone()))
        hooks.append(layer.t_func[0].register_forward_pre_hook(capture))
    try:
        with np.load(banks/'probe.npz') as data:
            model(torch.from_numpy(data['x'][:32]),torch.from_numpy(data['t'][:32]))
    finally:
        for hook in hooks:hook.remove()
    rows=[dict(layer=name,visit=1,activation=layer.activation_stats(hidden)) for name,layer,hidden in visits]
    assert len(rows)==len(list(model.ode_layers())) and len({r['layer'] for r in rows})==len(rows)
    return rows
