"""Release a single generator into four stage matrices, with identical initial map."""
from state_trial import HERE,base,torch,build_model as state_builder,StateFlowLayer
TAG='operator_sharing_20261116'

class StageOperatorLayer(StateFlowLayer):
    def solve(self,x,steps):
        y=x;dt=1/steps;p=self.operator()
        for i in range(steps):
            stage=min(3,4*i//steps)
            weight=p if stage==0 else p+self.stage_delta[stage-1]
            def drift(v):return torch.nn.functional.linear(v*self.step_size(v),weight)
            y=y+dt*drift(y+.5*dt*drift(y))
        return y

    @torch.no_grad()
    def activation_stats(self,x,context=None):
        stats=super().activation_stats(x,context);p=self.operator()
        stats.update(direct_mode='stage_dependent_isru_midpoint',
                     delta_frobenius=[float(v.norm()) for v in self.stage_delta],
                     effective_operator_frobenius=[float(p.norm())]+[float((p+v).norm()) for v in self.stage_delta])
        return stats

def build_model(dim,family):
    if family!='stage_operator_midpoint4':return state_builder(dim,family)
    model=state_builder(dim,'state_midpoint4')
    for _,layer in model.ode_layers():
        layer.__class__=StageOperatorLayer
        layer.stage_delta=torch.nn.Parameter(torch.zeros((3,*layer.proj.weight.shape),dtype=layer.proj.weight.dtype))
    return model

if __name__=='__main__':
    base.execute(base.argument_parser().parse_args(),grid_path=HERE/'sharing-grid.json',experiment_id=TAG,model_builder=build_model)
