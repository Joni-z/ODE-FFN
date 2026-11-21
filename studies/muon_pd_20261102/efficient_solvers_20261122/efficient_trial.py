"""Keep the full-width ISRU field and change its internal numerical integrator."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "coefficient_20261116"))
from state_trial import StateFlowLayer, base, build_model as build_midpoint, torch
from torch.nn.functional import linear

CALLS = {"ralston2": 4, "adams2": 3, "ssp_rk3": 3}


class AlternateSolver(StateFlowLayer):
    substeps = 2

    def solve(self, x, steps):
        weight = self.operator()

        def drift(y):
            return linear(y * self.step_size(y), weight)

        if self.solver == "ralston2":
            dt = 1 / steps
            for _ in range(steps):
                first = drift(x)
                second = drift(x + (2 * dt / 3) * first)
                x = x + dt * (0.25 * first + 0.75 * second)
            return x
        if self.solver == "adams2":
            first = drift(x)
            y = x + 0.5 * drift(x + 0.25 * first)
            return y + 0.75 * drift(y) - 0.25 * first
        y = x + drift(x)
        z = 0.75 * x + 0.25 * (y + drift(y))
        return x / 3 + (2 / 3) * (z + drift(z))

    @torch.no_grad()
    def activation_stats(self, x, context=None):
        stats = super().activation_stats(x, context)
        stats.update(direct_mode=self.solver, P_applications=CALLS[self.solver])
        return stats


def build_model(dim, family):
    model = build_midpoint(dim, "state_midpoint4")
    for _, layer in model.ode_layers():
        layer.substeps = 2
        if family in CALLS:
            layer.__class__ = AlternateSolver
            layer.solver = family
        elif family != "state_midpoint2":
            raise ValueError(family)
    return model


if __name__ == "__main__":
    torch.set_num_threads(1)
    base.execute(base.argument_parser().parse_args(), grid_path=HERE / "grid.json",
                 experiment_id="report4_efficient_solvers", model_builder=build_model)
