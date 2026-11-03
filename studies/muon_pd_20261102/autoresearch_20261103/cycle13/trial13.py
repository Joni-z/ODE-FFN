"""One input diagonal plus the unchanged historical ordinary ISRU control."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cycle4'))
from trial4 import base,torch,build_model as previous_builder,ConditionalSpectrumLayer
from ordinary import FreeISRU

native_projection_parameters=base.projection_parameters

def projection_parameters(model):
    selected=native_projection_parameters(model)
    return selected+[(name,p) for name,p in model.named_parameters()
                     if '.ffn.' in name and name.endswith(('.matrix1','.matrix2'))]

# A name-only extension for the copied ordinary control. Native selections and
# the official optimizer implementations remain unchanged.
base.projection_parameters=projection_parameters

class ISRUFeatureLayer(ConditionalSpectrumLayer):
    def step_size(self,x,context=None):
        return torch.rsqrt(1+1.2*x.square())

    @torch.no_grad()
    def activation_stats(self,x,context=None):
        stats=super().activation_stats(x,context)
        derivative=(1+1.2*x.square()).pow(-1.5)
        stats.update(direct_mode='isru',gate_transform='direct_isru',
                     feature_derivative_mean=float(derivative.mean()),
                     feature_derivative_max=float(derivative.max()))
        return stats

def build_model(dim,family):
    if family not in ('pd_isru_k1','free_isru'):return previous_builder(dim,family)
    model=previous_builder(dim,'pd_constant_k1')
    if family=='free_isru':
        for block in model.blocks:block.ffn=FreeISRU(block.ffn.ode1)
    else:
        for _,layer in model.ode_layers():layer.__class__=ISRUFeatureLayer
    return model

if __name__=='__main__':
    base.execute(base.argument_parser().parse_args(),grid_path=Path(__file__).with_name('grid.json'),experiment_id='professor_autoresearch_cycle13_20261103',model_builder=build_model)
