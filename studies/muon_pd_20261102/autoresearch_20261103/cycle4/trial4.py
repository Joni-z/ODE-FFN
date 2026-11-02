"""K1 diagonal-feature contrasts; reuse the existing model/trainer/optimizers."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cycle3'))
from trial3 import base,build_model as previous_builder,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from trial import ConditionalSpectrumLayer

FAMILIES=['pd_constant_k1','pd_direct_bounded_k1','pd_direct_shift_k1']


class FeatureLayer(ConditionalSpectrumLayer):
    def step_size(self,x,context=None):
        if self.diagonal_mode=='constant':return torch.ones_like(x)
        if self.diagonal_mode=='shift':return x+1
        return .2+.8*torch.sigmoid(x/10)

    @torch.no_grad()
    def activation_stats(self,x,context=None):
        result=super().activation_stats(x,context)
        result['direct_mode']=self.diagonal_mode
        if self.diagonal_mode=='bounded':
            q=torch.sigmoid(x/10);derivative=.08*q*(1-q)
            result['clock_outer_saturated_fraction']=float(((q<.01)|(q>.99)).float().mean())
        else:derivative=torch.full_like(x,0. if self.diagonal_mode=='constant' else 1.)
        result['direct_input_derivative_mean']=float(derivative.mean())
        return result


def build_model(dim,family):
    if family not in FAMILIES:return previous_builder(dim,family)
    model=previous_builder(dim,'pd_direct_raw_k1')
    mode={'pd_constant_k1':'constant','pd_direct_bounded_k1':'bounded','pd_direct_shift_k1':'shift'}[family]
    for _,layer in model.ode_layers():
        layer.diagonal_mode=mode
        layer.__class__=FeatureLayer
    return model


if __name__=='__main__':
    base.execute(base.argument_parser().parse_args(),grid_path=Path(__file__).with_name('grid.json'),experiment_id='professor_autoresearch_cycle4_20261103',model_builder=build_model)
