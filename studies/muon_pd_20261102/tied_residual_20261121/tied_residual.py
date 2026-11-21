"""Train the Report 4 eight-step shared residual candidate using the existing trainer."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "coefficient_20261116"))
from state_trial import StateFlowLayer, base, build_model as build_midpoint, torch


class TiedResidualLayer(StateFlowLayer):
    substeps = 8

    def solve(self, x, steps):
        weight = self.operator()
        for _ in range(steps):
            response = x * self.step_size(x)
            x = x + torch.nn.functional.linear(response, weight) / steps
        return x

    @torch.no_grad()
    def activation_stats(self, x, context=None):
        stats = super().activation_stats(x, context)
        stats.update(direct_mode="tied_euler8", P_applications=self.substeps)
        return stats


def build_model(dim, family):
    if family != "tied_euler8":
        return build_midpoint(dim, family)
    model = build_midpoint(dim, "state_midpoint4")
    for _, layer in model.ode_layers():
        layer.__class__ = TiedResidualLayer
        layer.orders = 1
    return model


if __name__ == "__main__":
    torch.set_num_threads(1)
    base.execute(base.argument_parser().parse_args(), grid_path=HERE / "grid.json",
                 experiment_id="report4_tied_residual", model_builder=build_model)
