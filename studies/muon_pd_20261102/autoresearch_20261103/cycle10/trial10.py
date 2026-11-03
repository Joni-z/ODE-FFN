"""D(h)-only radial/static contrast; unchanged native trainer and optimizer."""
from pathlib import Path
import math
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cycle3'))
from trial3 import base,build_model as previous_builder,torch
from trial2 import DirectLayer,ConditionalSpectrumLayer

class RadialLayer(DirectLayer):
    def step_size(self,x,context=None):
        if self.radial_mode=='radial':return x/(1+x.square().mean(dim=-1,keepdim=True)).sqrt()
        return x/math.sqrt(2)

    @torch.no_grad()
    def activation_stats(self,x,context=None):
        # Skip DirectLayer's raw/sigmoid-specific derivative metadata.
        stats=ConditionalSpectrumLayer.activation_stats(self,x,context)
        stats['direct_mode']=self.radial_mode
        stats['gate_transform']='direct_'+self.radial_mode
        denominator=(1+x.square().mean(dim=-1)).sqrt() if self.radial_mode=='radial' else x.new_full(x.shape[:-1],math.sqrt(2))
        stats['radial_denominator_mean']=float(denominator.mean())
        stats['radial_denominator_max']=float(denominator.max())
        return stats

def build_model(dim,family):
    if family not in ['pd_radial_k1','pd_scaled_raw_k1']:return previous_builder(dim,family)
    model=previous_builder(dim,'pd_direct_raw_k1')
    for _,layer in model.ode_layers():
        layer.__class__=RadialLayer
        layer.radial_mode='radial' if family=='pd_radial_k1' else 'static'
    return model

if __name__=='__main__':
    base.execute(base.argument_parser().parse_args(),grid_path=Path(__file__).with_name('grid.json'),experiment_id='professor_autoresearch_cycle10_20261103',model_builder=build_model)
