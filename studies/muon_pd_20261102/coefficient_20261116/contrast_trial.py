"""Change stage coordinates only; retain the native trainer and ISRU dynamics."""
from sharing_trial import (
    HERE, base, torch, build_model as anchored_builder, StateFlowLayer,
)

TAG = 'mean_preserving_stages_20261116'
FAMILY = 'contrast_operator_midpoint4'


class MeanPreservingStageLayer(StateFlowLayer):
    def stage_weights(self) -> torch.Tensor:
        """Four matrices with mean P, using fixed orthonormal zero-sum contrasts."""
        p = self.operator()
        a, b, c = self.stage_delta.unbind(0)
        return torch.stack((p+.5*(a+b+c), p+.5*(a-b-c),
                            p+.5*(-a+b-c), p+.5*(-a-b+c)))

    def solve(self, x: torch.Tensor, steps: int) -> torch.Tensor:
        y, dt = x, 1/steps
        weights = self.stage_weights()
        for i in range(steps):
            weight = weights[min(3, 4*i//steps)]
            def drift(v: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.linear(v*self.step_size(v), weight)
            y = y+dt*drift(y+.5*dt*drift(y))
        return y

    @torch.no_grad()
    def activation_stats(self, x: torch.Tensor, context=None) -> dict:
        result = super().activation_stats(x, context)
        weights = self.stage_weights()
        result.update(direct_mode='mean_preserving_stage_isru_midpoint',
            contrast_frobenius=[float(v.norm()) for v in self.stage_delta],
            effective_operator_frobenius=[float(v.norm()) for v in weights],
            effective_mean_residual=float((weights.mean(0)-self.operator()).norm()))
        return result


def build_model(dim: int, family: str):
    if family != FAMILY:
        return anchored_builder(dim, family)
    model = anchored_builder(dim, 'stage_operator_midpoint4')
    for _, layer in model.ode_layers():
        layer.__class__ = MeanPreservingStageLayer
    return model


if __name__ == '__main__':
    base.execute(base.argument_parser().parse_args(), grid_path=HERE/'contrast-grid.json',
                 experiment_id=TAG, model_builder=build_model)
