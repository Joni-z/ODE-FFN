"""Release only Taylor coefficients; reuse the pinned native training loop."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'response_screen_20261115'))
from screen_trial import base,torch,build_model as fixed_builder
from train import ConditionalSpectrumLayer

TAG='coefficient_20261116'

class FreeCoefficientLayer(ConditionalSpectrumLayer):
    def step_size(self,x,context=None):
        return torch.rsqrt(1+1.2*x.square())

    def forward(self,x,context=None,*,return_duration=False):
        d=self.step_size(x,context);p=self.operator()
        total,term=x,x
        for k in range(1,9):
            term=torch.nn.functional.linear(term*d,p)/k
            total=total+(term if k==1 else self.alpha[k-2]*term)
        return (total,d) if return_duration else total

    @torch.no_grad()
    def activation_stats(self,x,context=None):
        stats=super().activation_stats(x,context)
        d=self.step_size(x,context);p=self.operator();term=x;weighted=x
        effects=[]
        for k in range(1,9):
            term=torch.nn.functional.linear(term*d,p)/k
            contribution=term if k==1 else self.alpha[k-2]*term
            weighted=weighted+contribution
            effects.append(float(contribution.square().mean().sqrt()))
        stats.update(alpha=self.alpha.detach().cpu().tolist(),
                     weighted_term_rms=effects,weighted_output_rms=float(weighted.square().mean().sqrt()),
                     direct_mode='isru_free_coefficients')
        return stats

def build_model(dim,family):
    if family!='free_coeff_k8':return fixed_builder(dim,family)
    model=fixed_builder(dim,'screen_isru_k8')
    for _,layer in model.ode_layers():
        assert layer.orders==8 and layer.rate_placement=='before_projection'
        assert layer.solve_return=='map' and layer.forcing_mode=='none'
        layer.__class__=FreeCoefficientLayer
        layer.alpha=torch.nn.Parameter(torch.ones(7,dtype=layer.proj.weight.dtype))
    return model

if __name__=='__main__':
    base.execute(base.argument_parser().parse_args(),grid_path=HERE/'grid.json',
                 experiment_id=TAG,model_builder=build_model)
