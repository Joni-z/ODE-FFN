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
    return self.output(carrier * self.norm(features))


def build_model(dim, family):
    if not family.startswith('odd_'):
        return standard(dim, family)
    model = standard(dim, family.removeprefix('odd_'))
    model.output.register_parameter('bias', None)
    model.forward = MethodType(odd_velocity, model)
    return model


if __name__ == '__main__':
    torch.set_num_threads(1)
    base.execute(base.argument_parser().parse_args(), grid_path=HERE/'odd-grid.json',
                 experiment_id='odd_field_20261123', model_builder=build_model)
