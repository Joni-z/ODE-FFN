"""Native model/trainer/Muon with a separately logged post-step constraint."""
from pathlib import Path
import math
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cycle3'))
from trial3 import base,build_model,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cycle7'))
from frozen_shape import variants

CADENCE=10
STRENGTH=1-math.exp(-CADENCE/10000)

class RelaxedOptimizer(base.ProjectionOptimizer):
    def __init__(self,model,cell,log_path=None):
        super().__init__(model,cell['optimizer'],cell['projection_lr'],None)
        self.relaxation=cell['relaxation'];self.outer_log=log_path
        assert self.relaxation in ['none','shape','scale']

    def step(self):
        logging=self.outer_log is not None and self.count+1 in base.SPECTRAL_STEPS
        if logging:
            before={n:p.detach().clone() for n,p in self.selected}
            gradients={n:p.grad.detach().clone() for n,p in self.selected if p.grad is not None}
        super().step()
        if logging:native={n:p.detach().clone() for n,p in self.selected}
        applied=self.relaxation!='none' and self.count%CADENCE==0
        if applied:
            with torch.no_grad():
                for _,p in self.selected:p.copy_(variants(p,STRENGTH)[self.relaxation=='scale'])
        if logging:
            matrices={}
            for name,p in self.selected:
                old=before[name].double();delta=p.detach().double()-old
                row=dict(weight=base.spectral(p),delta=base.spectral(delta),
                         native_delta=base.spectral(native[name].double()-old),
                         relaxation_delta=base.spectral(p.detach().double()-native[name].double()),
                         norm_energy=dict(radial_cross_term=float(2*(old*delta).sum()),update_squared_norm=float(delta.square().sum()),observed_weight_squared_norm_change=float(p.detach().double().square().sum()-old.square().sum())))
                if name in gradients:row['clipped_gradient']=base.spectral(gradients[name])
                momentum=self.projections.state[p].get('momentum_buffer')
                if momentum is not None:row['muon_momentum']=base.spectral(momentum)
                matrices[name]=row
            base.train.json_line(self.outer_log,dict(step=self.count,optimizer=self.name,relaxation=self.relaxation,relaxation_applied=applied,strength=STRENGTH if applied else 0.,cadence=CADENCE,matrices=matrices))

def optimizer_builder(model,cell,log_path):
    return RelaxedOptimizer(model,cell,log_path)

if __name__=='__main__':
    base.execute(base.argument_parser().parse_args(),grid_path=Path(__file__).with_name('grid.json'),experiment_id='professor_autoresearch_cycle8_20261103',model_builder=build_model,optimizer_builder=optimizer_builder)
