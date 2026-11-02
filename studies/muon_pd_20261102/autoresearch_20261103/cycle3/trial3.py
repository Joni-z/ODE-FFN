"""Prospective order-only ablation and native smaller SwiGLU budget control."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from trial import base,torch


def build_model(dim,family):
    if family=='scalar_k1':
        model=base.build_model(dim,'scalar_k2')
        for _,layer in model.ode_layers():layer.orders=1
        return model
    if family=='swiglu64':
        initial=torch.get_rng_state().clone()
        reference=base.build_model(dim,'scalar_k2')
        # Follow base.build_model's paired construction, using the unchanged
        # native SwiGLU with the sole nominal-width argument64.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(initial)
            model=base.VelocityNet(dim,'swiglu',width=64,depth=3,ffn_hidden=64)
        source=dict(reference.named_parameters());buffers=dict(reference.named_buffers())
        with torch.no_grad():
            for name,p in model.named_parameters():
                if '.ffn.' not in name:p.copy_(source[name])
            for name,b in model.named_buffers():
                if '.ffn.' not in name:b.copy_(buffers[name])
        return model
    return base.build_model(dim,family)

if __name__=='__main__':
    base.execute(base.argument_parser().parse_args(),grid_path=Path(__file__).with_name('grid.json'),experiment_id='professor_autoresearch_cycle3_20261103',model_builder=build_model)
