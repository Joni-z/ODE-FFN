"""One-pass odd velocity fields with the same state-dependent midpoint FFN."""
from types import MethodType
from refresh_trial import HERE, base, build_model as standard, torch


def odd_velocity(self, x, t):
    if x.ndim != 2 or x.shape[1] != self.dim or t.shape != (len(x),1):
        raise ValueError('expected x=[batch,dim], t=[batch,1]')
    carrier = torch.nn.functional.linear(x, self.input.weight)
    features = carrier.square() + self.input.bias
    time = self.time(t)
    for block in self.blocks:
        features = block(features, time)
    if self.bounded_carrier:
        carrier = carrier * torch.rsqrt(1 + carrier.square())
    return self.output(carrier * self.norm(features))


def build_model(dim, family):
    bounded = family.startswith('bounded_odd_')
    family = family.removeprefix('bounded_') if bounded else family
    if not family.startswith('odd_'):
        return standard(dim, family)
    model = standard(dim, family.removeprefix('odd_'))
    model.bounded_carrier = bounded
    model.output.register_parameter('bias', None)
    model.forward = MethodType(odd_velocity, model)
    return model


if __name__ == '__main__':
    torch.set_num_threads(1)
    args = base.argument_parser().parse_args()
    control = '__odd_swiglu' in args.cell
    grid = 'odd-control-grid.json' if control else 'odd-grid.json'
    tag = 'odd_control_20261123' if control else 'odd_field_20261123'
    if '__bounded_odd_' in args.cell:
        grid, tag = 'bounded-grid.json', 'bounded_odd_20261123'
    base.execute(args, grid_path=HERE/grid, experiment_id=tag, model_builder=build_model)
