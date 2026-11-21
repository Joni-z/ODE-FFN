"""Refresh and depth ablations, plus an exactly reduced nonlinear state flow."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "coefficient_20261116"))
from state_trial import StateFlowLayer, base, build_model as build_reference, torch
from torch.nn.functional import linear


class FrozenResponse(StateFlowLayer):
    def solve(self, x, steps):
        diagonal = self.step_size(x)
        weight = self.operator()
        dt = 1 / steps
        y = x
        for _ in range(steps):
            half = y + (dt / 2) * linear(y * diagonal, weight)
            y = y + dt * linear(half * diagonal, weight)
        return y


class ReducedState(StateFlowLayer):
    def factors(self):
        # Pack the two maps into the same number of weights as the square P.
        projection = self.operator()
        encode, decode_t = projection.chunk(2)
        decode = (6 ** 0.5) * decode_t.T
        return encode, decode

    def solve(self, x, steps):
        encode, decode = self.factors()
        coupling = encode @ decode
        latent = linear(x, encode)
        integral = torch.zeros_like(latent)
        dt = 1 / steps
        for _ in range(steps):
            current = latent + linear(integral, coupling)
            first = current * self.step_size(current)
            half = current + (dt / 2) * linear(first, coupling)
            integral = integral + dt * half * self.step_size(half)
        return x + linear(integral, decode)

    def activation_stats(self, x, context=None):
        stats = super().activation_stats(x, context)
        stats.update(direct_mode="reduced_state", latent_dim=x.shape[-1] // 2,
                     latent_matrix_calls=2 * self.substeps, P_applications=None)
        return stats


def build_model(dim, family):
    if family not in ("frozen_midpoint4", "state_midpoint2", "reduced_midpoint4"):
        return build_reference(dim, family)
    model = build_reference(dim, "state_midpoint4")
    for _, layer in model.ode_layers():
        if family == "frozen_midpoint4":
            layer.__class__ = FrozenResponse
        elif family == "state_midpoint2":
            layer.substeps = 2
        else:
            layer.__class__ = ReducedState
    return model


if __name__ == "__main__":
    torch.set_num_threads(1)
    base.execute(base.argument_parser().parse_args(), grid_path=HERE / "grid.json",
                 experiment_id="report4_reduced_state", model_builder=build_model)
