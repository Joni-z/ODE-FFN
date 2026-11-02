"""Matched-spectrum abs diagonal and ordinary native bilinear reference."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cycle3'))
from trial3 import base,build_model as previous_builder,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from trial import ConditionalSpectrumLayer
from synthetic.models import SwiGLUFFN

FAMILIES=['pd_direct_abs_k1','bilinear64']


class AbsoluteLayer(ConditionalSpectrumLayer):
    def step_size(self,x,context=None):return x.abs()

    @torch.no_grad()
    def activation_stats(self,x,context=None):
        result=super().activation_stats(x,context)
        result['direct_mode']='absolute'
        result['direct_input_derivative_abs_mean']=float((x!=0).float().mean())
        return result


class BilinearFFN(SwiGLUFFN):
    def forward(self,x):
        gate,value=self.w12(x).chunk(2,dim=-1)
        return self.w3(gate*value)


def build_model(dim,family):
    if family=='pd_direct_abs_k1':
        model=previous_builder(dim,'pd_direct_raw_k1')
        for _,layer in model.ode_layers():layer.__class__=AbsoluteLayer
        return model
    if family=='bilinear64':
        model=previous_builder(dim,'swiglu64')
        for block in model.blocks:block.ffn.__class__=BilinearFFN
        return model
    return previous_builder(dim,family)


if __name__=='__main__':
    base.execute(base.argument_parser().parse_args(),grid_path=Path(__file__).with_name('grid.json'),experiment_id='professor_autoresearch_cycle5_20261103',model_builder=build_model)
