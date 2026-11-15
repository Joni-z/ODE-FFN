"""One response substitution; the source-pinned native trainer stays unchanged."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'autoresearch_20261103/cycle13'))
from trial13 import base, build_model as previous_builder, torch
from trial4 import ConditionalSpectrumLayer

TAG = 'professor_followup_20261115'


class TanhFeatureLayer(ConditionalSpectrumLayer):
    def step_size(self, x, context=None):
        z = 1.2**.5 * x
        small = z.abs() < 1e-3
        safe = torch.where(small, torch.ones_like(z), z)
        # Removable singularity: preserve derivatives at zero, including q'''(0).
        series = 1 - z.square()/3 + 2*z.pow(4)/15 - 17*z.pow(6)/315
        return torch.where(small, series, torch.tanh(z)/safe)

    @torch.no_grad()
    def activation_stats(self, x, context=None):
        stats = super().activation_stats(x, context)
        derivative = 1-torch.tanh(1.2**.5*x).square()
        stats.update(direct_mode='scaled_tanh',
                     feature_derivative_mean=float(derivative.mean()),
                     feature_derivative_max=float(derivative.max()))
        return stats


def build_model(dim, family):
    if family != 'pd_tanh_k1':
        return previous_builder(dim, family)
    model = previous_builder(dim, 'pd_isru_k1')
    for _, layer in model.ode_layers():
        assert layer.orders == 1 and layer.rate_placement == 'before_projection'
        layer.__class__ = TanhFeatureLayer
    return model


if __name__ == '__main__':
    base.execute(base.argument_parser().parse_args(), grid_path=HERE/'grid.json',
                 experiment_id=TAG, model_builder=build_model)
