"""Midpoint with an identity-initialized learned coordinate map for the diagonal."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'coefficient_20261116'))
from state_trial import StateFlowLayer, base, build_model as original, torch


class CrossResponse(StateFlowLayer):
    def step_size(self, x, context=None):
        projected = torch.nn.functional.linear(x, self.gate_matrix)
        return torch.rsqrt(1 + 1.2 * projected.square())


def build_model(dim, family):
    if family != 'cross_midpoint4':
        return original(dim, family)
    model = original(dim, 'state_midpoint4')
    for _, layer in model.ode_layers():
        layer.__class__ = CrossResponse
        layer.gate_matrix = torch.nn.Parameter(torch.eye(layer.proj.weight.shape[1]))
    return model


if __name__ == '__main__':
    torch.set_num_threads(1)
    base.execute(base.argument_parser().parse_args(), grid_path=HERE/'grid.json',
                 experiment_id='refresh_20261123', model_builder=build_model)
