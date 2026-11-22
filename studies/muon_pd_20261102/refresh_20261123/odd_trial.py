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


def midpoint_block(self, x, temb):
    scale, shift = self.condition(temb).chunk(2, dim=-1)
    gain = 1 + scale
    def drift(state):
        return self.ffn(self.norm(state) * gain + shift)
    half = x + (0.5 * self.residual_scale) * drift(x)
    return x + self.residual_scale * drift(half)


def feedback_ffn(self, x):
    gate, value = self.w12(x).chunk(2, dim=-1)
    half = 0.5 * torch.nn.functional.silu(gate) * value
    response = half * torch.rsqrt(1 + half.square())
    updated_gate = gate + self.feedback * response
    return self.w3(torch.nn.functional.silu(updated_gate) * value)


def build_model(dim, family):
    bounded = family.startswith('bounded_odd_')
    family = family.removeprefix('bounded_') if bounded else family
    if not family.startswith('odd_'):
        return standard(dim, family)
    midpoint = family == 'odd_glu_midpoint192'
    feedback = family == 'odd_feedback192'
    model = standard(dim, 'swiglu192' if midpoint or feedback else family.removeprefix('odd_'))
    if feedback:
        for block in model.blocks:
            ffn = block.ffn
            ffn.feedback = torch.nn.Parameter(ffn.w3.weight.new_zeros(ffn.w3.in_features))
            ffn.forward = MethodType(feedback_ffn, ffn)
    if midpoint:
        for block in model.blocks:
            block.forward = MethodType(midpoint_block, block)
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
    if '__odd_glu_midpoint192__' in args.cell:
        grid, tag = 'glu-grid.json', 'glu_midpoint_20261123'
    if args.cell == 'gmm_ring8_2d__odd_swiglu192__adamw_0.0003':
        grid, tag = 'glu-euler-grid.json', 'glu_euler_ring_20261123'
    if '__odd_feedback192__' in args.cell:
        grid, tag = 'feedback-grid.json', 'gate_feedback_20261123'
    if '__bounded_odd_swiglu' in args.cell:
        grid, tag = 'bounded-control-grid.json', 'bounded_control_20261123'
    base.execute(args, grid_path=HERE/grid, experiment_id=tag, model_builder=build_model)
