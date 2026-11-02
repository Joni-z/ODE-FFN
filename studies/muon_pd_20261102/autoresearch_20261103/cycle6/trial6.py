"""Budget-only confirmation; unchanged parent model and native trainer."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cycle3'))
from trial3 import base,build_model

if __name__=='__main__':
    # Extend the existing read-only logger, without changing optimizer steps.
    base.SPECTRAL_STEPS=set(base.SPECTRAL_STEPS)|{40000,60000,80000}
    base.execute(base.argument_parser().parse_args(),grid_path=Path(__file__).with_name('grid.json'),experiment_id='professor_autoresearch_cycle6_20261103',model_builder=build_model)
