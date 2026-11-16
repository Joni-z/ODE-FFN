"""Same generator, response and eight matrix applications; change only solver."""
from state_trial import HERE, base, torch, build_model as state_builder, StateFlowLayer
from rk4_step import rk4_step_func

TAG = 'state_rk4_control_20261116'


class StateRK4Layer(StateFlowLayer):
    substeps = 2

    def solve(self, x, steps):
        if steps < 1:
            raise ValueError('RK4 substeps must be positive')
        y, matrix, dt = x, self.operator(), 1/steps
        def drift(time, state, **unused):
            return torch.nn.functional.linear(state*self.step_size(state), matrix)
        for i in range(steps):
            y = y+rk4_step_func(drift, i*dt, dt, (i+1)*dt, y)
        return y

    @torch.no_grad()
    def activation_stats(self, x, context=None):
        stats = super().activation_stats(x, context)
        stats.update(direct_mode='state_isru_classical_rk4', internal_substeps=self.substeps,
                     integrator_order=4, P_applications=4*self.substeps)
        return stats


def build_model(dim, family):
    if family != 'state_rk4_2':
        return state_builder(dim, family)
    model = state_builder(dim, 'state_midpoint4')
    for _, layer in model.ode_layers():
        layer.__class__ = StateRK4Layer
        layer.orders = 4
    return model


if __name__ == '__main__':
    base.execute(base.argument_parser().parse_args(), grid_path=HERE/'rk4-grid.json',
                 experiment_id=TAG, model_builder=build_model)
