"""Conditional second stage: same P work, response follows the hidden state."""
import json
from coefficient_trial import HERE,base,torch,fixed_builder,ConditionalSpectrumLayer

TAG='state_flow_20261116'

class StateFlowLayer(ConditionalSpectrumLayer):
    substeps=4
    def step_size(self,x,context=None):return torch.rsqrt(1+1.2*x.square())
    def solve(self,x,steps):
        y=x;p=self.operator();dt=1/steps
        def drift(v):return torch.nn.functional.linear(v*self.step_size(v),p)
        for _ in range(steps):y=y+dt*drift(y+.5*dt*drift(y))
        return y
    def forward(self,x,context=None,*,return_duration=False):
        y=self.solve(x,self.substeps)
        return (y,self.step_size(x,context)) if return_duration else y
    @torch.no_grad()
    def activation_stats(self,x,context=None):
        y=self(x);d=self.step_size(x)
        return dict(s_mean=float(d.mean()),s_min=float(d.min()),s_max=float(d.max()),
                    input_rms=float(x.square().mean().sqrt()),output_rms=float(y.square().mean().sqrt()),
                    increment_rms=float((y-x).square().mean().sqrt()),
                    direct_mode='state_isru_midpoint',internal_substeps=self.substeps,P_applications=2*self.substeps)

def build_model(dim,family):
    if family!='state_midpoint4':return fixed_builder(dim,family)
    model=fixed_builder(dim,'screen_isru_k8')
    for _,layer in model.ode_layers():layer.__class__=StateFlowLayer
    return model

if __name__=='__main__':
    base.execute(base.argument_parser().parse_args(),grid_path=HERE/'state-grid.json',experiment_id=TAG,model_builder=build_model)
