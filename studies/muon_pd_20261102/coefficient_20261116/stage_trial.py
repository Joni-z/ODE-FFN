"""Release midpoint stage coefficients at identical initial function/work."""
from state_trial import HERE,base,torch,build_model as state_builder,StateFlowLayer
TAG='free_stage_20261116'

class FreeStageLayer(StateFlowLayer):
    def solve(self,x,steps):
        y=x;p=self.operator();dt=1/steps
        def drift(v):return torch.nn.functional.linear(v*self.step_size(v),p)
        for i in range(steps):
            gamma=self.stage_gamma[min(3,4*i//steps)]
            y=y+dt*drift(y+(.5*dt*gamma)*drift(y))
        return y
    @torch.no_grad()
    def activation_stats(self,x,context=None):
        stats=super().activation_stats(x,context)
        stats.update(stage_gamma=self.stage_gamma.detach().tolist(),direct_mode='free_stage_midpoint')
        return stats

def build_model(dim,family):
    if family!='free_stage_midpoint4':return state_builder(dim,family)
    model=state_builder(dim,'state_midpoint4')
    for _,layer in model.ode_layers():
        layer.__class__=FreeStageLayer
        layer.stage_gamma=torch.nn.Parameter(torch.ones(4,dtype=layer.proj.weight.dtype))
    return model

if __name__=='__main__':
    base.execute(base.argument_parser().parse_args(),grid_path=HERE/'stage-grid.json',experiment_id=TAG,model_builder=build_model)
