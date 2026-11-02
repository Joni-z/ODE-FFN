"""Prospective direct-D and controller-rank counterfactuals; fixed parent trainer."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from trial import base,build_model as first_builder,ConditionalSpectrumLayer,torch,nn


class DirectLayer(ConditionalSpectrumLayer):
    def step_size(self,x,context=None):
        return x if self.direct_mode=='raw' else torch.sigmoid(x/10)*.8+.2

    @torch.no_grad()
    def activation_stats(self,x,context=None):
        stats=super().activation_stats(x,context)
        stats['direct_mode']=self.direct_mode
        if self.direct_mode=='raw':
            stats['clock_raw_logit_sensitivity_mean']=1.
            # The input is used directly, not passed through sigmoid logits.
            stats.pop('clock_outer_saturated_fraction',None)
        else:
            q=torch.sigmoid(x/10)
            stats['clock_raw_logit_sensitivity_mean']=float((.08*q*(1-q)).mean())
            stats['clock_outer_saturated_fraction']=float(((q<.01)|(q>.99)).float().mean())
        return stats


def build_model(dim,family):
    if family not in ['pd_factorized_linear','pd_direct_raw','pd_direct_bounded']:
        return first_builder(dim,family)
    model=base.build_model(dim,'vector_pd_k2')
    for _,layer in model.ode_layers():
        if family=='pd_factorized_linear':
            first,_,last=layer.t_func
            with torch.no_grad():
                q=torch.sigmoid(first.bias);slope=q*(1+first.bias*(1-q))
                original=last.weight.detach().clone()
                last.bias.add_(original@(torch.nn.functional.silu(first.bias)-slope*first.bias))
                last.weight.mul_(slope[None,:])
            layer.t_func[1]=nn.Identity()
            layer.__class__=ConditionalSpectrumLayer
        else:
            del layer.t_func
            layer.gate_source='provided'
            layer.direct_mode='raw' if family=='pd_direct_raw' else 'bounded'
            layer.__class__=DirectLayer
    return model

if __name__=='__main__':
    base.execute(base.argument_parser().parse_args(),grid_path=Path(__file__).with_name('grid.json'),experiment_id='professor_autoresearch_cycle2_20261103',model_builder=build_model)
