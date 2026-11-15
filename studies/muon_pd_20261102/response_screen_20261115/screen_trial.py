"""Response-only adapters around the source-pinned native trainer."""
from pathlib import Path
import math
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'order_response_20261115'))
from order_trial import base,torch,build_model as previous_builder
from train import ConditionalSpectrumLayer

TAG='response_screen_20261115'
RESPONSES=('isru','tanh','arctan','softsign','erf','silu','gelu','linear')

def diagonal(name,x):
    c=math.sqrt(1.2)
    if name=='isru':return torch.rsqrt(1+1.2*x.square())
    if name=='linear':return torch.ones_like(x)
    if name=='softsign':return 1/(1+c*x.abs())
    if name=='silu':return 2*torch.sigmoid(x)
    if name=='gelu':return 2*torch.special.ndtr(x)
    z=x*(c if name=='tanh' else math.pi*c/2 if name=='arctan' else math.sqrt(math.pi)*c/2)
    small=z.abs()<1e-3
    safe=torch.where(small,torch.ones_like(z),z)
    if name=='tanh':
        series=1-z.square()/3+2*z.pow(4)/15-17*z.pow(6)/315
        ratio=torch.tanh(z)/safe
    elif name=='arctan':
        series=1-z.square()/3+z.pow(4)/5-z.pow(6)/7
        ratio=torch.atan(z)/safe
    elif name=='erf':
        series=1-z.square()/3+z.pow(4)/10-z.pow(6)/42
        ratio=(math.sqrt(math.pi)/2)*torch.erf(z)/safe
    else:raise ValueError(name)
    return torch.where(small,series,ratio)

def response(name,x):
    c=math.sqrt(1.2)
    if name=='isru':return x*torch.rsqrt(1+1.2*x.square())
    if name=='tanh':return torch.tanh(c*x)/c
    if name=='arctan':return 2/(math.pi*c)*torch.atan(math.pi*c*x/2)
    if name=='softsign':return torch.nn.functional.softsign(c*x)/c
    if name=='erf':return torch.erf(math.sqrt(math.pi)*c*x/2)/c
    if name=='silu':return 2*torch.nn.functional.silu(x)
    if name=='gelu':return 2*torch.nn.functional.gelu(x,approximate='none')
    if name=='linear':return x
    raise ValueError(name)

class ResponseLayer(ConditionalSpectrumLayer):
    def step_size(self,x,context=None):return diagonal(self.response_name,x)

    @torch.no_grad()
    def activation_stats(self,x,context=None):
        stats=super().activation_stats(x,context)
        stats.update(direct_mode=self.response_name,gate_transform='direct_'+self.response_name)
        return stats

def build_model(dim,family):
    if not family.startswith('screen_'):return previous_builder(dim,family)
    name,order=family.removeprefix('screen_').rsplit('_k',1)
    order=int(order);assert name in RESPONSES and order in (1,8)
    model=previous_builder(dim,'pd_tanh_k1' if name=='tanh' else 'pd_isru_k1')
    for _,layer in model.ode_layers():
        assert layer.rate_placement=='before_projection'
        layer.orders=order
        if name not in ('isru','tanh'):
            layer.__class__=ResponseLayer;layer.response_name=name
    return model

if __name__=='__main__':
    base.execute(base.argument_parser().parse_args(),grid_path=HERE/'grid.json',
                 experiment_id=TAG,model_builder=build_model)
