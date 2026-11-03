"""Missing rank32-controller/affine-output cell; native training unchanged."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cycle2'))
from trial2 import base,torch,build_model as controller_builder
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cycle3'))
from trial3 import build_model as ordinary_builder

def build_model(dim,family):
    if family=='swiglu64':return ordinary_builder(dim,family)
    if family!='pd_factorized_affine':return controller_builder(dim,family)
    model=controller_builder(dim,'pd_factorized_linear')
    for _,layer in model.ode_layers():layer.gate_transform='affine'
    return model

if __name__=='__main__':
    base.execute(base.argument_parser().parse_args(),grid_path=Path(__file__).with_name('grid.json'),experiment_id='professor_autoresearch_cycle11_20261103',model_builder=build_model)
