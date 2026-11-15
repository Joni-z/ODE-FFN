"""Order-only extension of the verified ISRU/tanh response comparison."""
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'followup_20261115'))
from train import base, torch, build_model as k1_builder

TAG='order_response_20261115'


def build_model(dim,family):
    if family not in ('pd_isru_k8','pd_tanh_k8'):
        return k1_builder(dim,family)
    model=k1_builder(dim,family[:-1]+'1')
    for _,layer in model.ode_layers():
        assert layer.orders==1 and layer.rate_placement=='before_projection'
        layer.orders=8
    return model


if __name__=='__main__':
    base.execute(base.argument_parser().parse_args(),grid_path=HERE/'grid.json',
                 experiment_id=TAG,model_builder=build_model)
