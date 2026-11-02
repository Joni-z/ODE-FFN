"""Use the first-wave trainer/metrics unchanged through its explicit adapters."""
from pathlib import Path
from variants import base, build_model, optimizer

if __name__ == '__main__':
    base.execute(base.argument_parser().parse_args(),
                 grid_path=Path(__file__).with_name('grid.json'),
                 experiment_id='muon_pd_wave2_20261102',
                 model_builder=build_model, optimizer_builder=optimizer)
