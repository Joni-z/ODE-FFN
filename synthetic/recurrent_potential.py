"""Same complete potential readout around shared or cloned hidden evolution."""
from synthetic.conservative_readout import ReadoutNet
from synthetic.recurrent_depth import build as recurrent_build


def build(dim,family,setting):
    if family not in ('ode_k2','swiglu128','swiglu192'):
        raise ValueError('unsupported family')
    layout,readout=setting.split('_',1)
    if layout not in ('shared','cloned','native') or readout not in ('direct','gradient'):
        raise ValueError('unsupported layout/readout factor')
    recurrent=recurrent_build(dim,family,layout)
    model=ReadoutNet(recurrent,dim,readout)
    model.layout=layout
    return model
