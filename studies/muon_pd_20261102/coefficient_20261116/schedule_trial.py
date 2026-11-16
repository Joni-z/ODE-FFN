"""Matched P/B models: change only the chronological order of four stages."""
from state_trial import HERE, base, torch, build_model as state_builder, StateFlowLayer

TAG = 'schedule_order_20261116'
SCHEDULES = {
    'palindrome_operator_midpoint4': (1, -1, -1, 1),
    'alternating_operator_midpoint4': (1, -1, 1, -1),
}


class ScheduledOperatorLayer(StateFlowLayer):
    def stage_weights(self) -> torch.Tensor:
        p = self.operator()
        return torch.stack([p + sign*self.stage_offset for sign in self.schedule])

    def solve(self, x: torch.Tensor, steps: int) -> torch.Tensor:
        if steps < 4 or steps % 4:
            raise ValueError('refinement must preserve the four equal-duration stages')
        y, dt = x, 1/steps
        weights = self.stage_weights()
        for i in range(steps):
            weight = weights[4*i//steps]
            def drift(v: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.linear(v*self.step_size(v), weight)
            y = y + dt*drift(y + .5*dt*drift(y))
        return y

    @torch.no_grad()
    def activation_stats(self, x: torch.Tensor, context=None) -> dict:
        stats = super().activation_stats(x, context)
        stats.update(direct_mode='scheduled_operator_isru_midpoint',
            schedule=list(self.schedule), offset_frobenius=float(self.stage_offset.norm()),
            effective_operator_frobenius=[float(w.norm()) for w in self.stage_weights()])
        return stats


def build_model(dim: int, family: str):
    if family not in SCHEDULES:
        return state_builder(dim, family)
    model = state_builder(dim, 'state_midpoint4')
    for _, layer in model.ode_layers():
        layer.__class__ = ScheduledOperatorLayer
        layer.schedule = SCHEDULES[family]
        # zeros_like allocates B without consuming the shared initialization RNG.
        layer.stage_offset = torch.nn.Parameter(torch.zeros_like(layer.proj.weight))
    return model


if __name__ == '__main__':
    base.execute(base.argument_parser().parse_args(), grid_path=HERE/'schedule-grid.json',
                 experiment_id=TAG, model_builder=build_model)
