"""Independent dense-Q, active derivative, pairing and response-rank fixtures."""
from __future__ import annotations

import copy
import numpy as np
import torch
from torch.nn import functional as F


def dense(layer, x):
    hidden = F.silu(F.linear(x, layer.t_func[0].weight, layer.t_func[0].bias))
    clock = torch.sigmoid(F.linear(hidden, layer.t_func[2].weight, layer.t_func[2].bias)/layer.tau)*layer.scale+layer.shift
    features = hidden if layer.frame_mode=='state' else torch.ones_like(hidden)
    g = F.linear(features, layer.frame_weight, layer.frame_bias)
    u = torch.cat((torch.ones_like(g[..., :1]), g), -1)
    width = x.shape[-1];identity = torch.eye(width, dtype=x.dtype)
    h = identity-2*u.unsqueeze(-1)*u.unsqueeze(-2)/u.square().sum(-1, keepdim=True).unsqueeze(-1)
    h0 = identity.clone();h0[0, 0] = -1
    q = h@h0;p = layer.proj.weight
    matrix = identity+clock.unsqueeze(-1)*p
    if layer.orders==2:
        matrix = matrix+.5*clock.square().unsqueeze(-1)*(p@p)
    return ((q@matrix@q.transpose(-1, -2))@x.unsqueeze(-1)).squeeze(-1), q


def verify(operator, original_class):
    torch.manual_seed(0)
    source = original_class(5, 7).double();x = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    before = torch.get_rng_state().clone();layer = operator.ConditionalFrameLayer(copy.deepcopy(source))
    assert torch.equal(before, torch.get_rng_state())
    torch.testing.assert_close(layer(x), source(x), rtol=0, atol=0)
    # A nonzero receiving vector exposes inherited backbone responses.
    receive = torch.linspace(-.4, .8, 5, dtype=torch.float64)
    a = (source(x)*receive).sum();b = (layer(x)*receive).sum()
    old = torch.autograd.grad(a, (x, *source.parameters()))
    inherited = dict(layer.named_parameters())
    new = torch.autograd.grad(b, (x, *(inherited[k] for k, _ in source.named_parameters())))
    for left, right in zip(old, new):torch.testing.assert_close(left, right, rtol=0, atol=0)
    maximum = 0.;sv_error = 0.;global_error = 0.;commutator_error = 0.;eigen_error = 0.
    for mode in ('state', 'global'):
        for order in (1, 2):
            tested = operator.ConditionalFrameLayer(copy.deepcopy(source), mode, order)
            with torch.no_grad():
                tested.frame_weight.normal_(0, .2);tested.frame_bias.normal_(0, .1)
            value = x.detach().clone().requires_grad_(True)
            actual = tested(value);expected, q = dense(tested, value)
            torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
            torch.testing.assert_close(q@q.transpose(-1, -2), torch.eye(5).expand(3, 5, 5).double(), atol=1e-12, rtol=0)
            for index in range(actual.numel()):
                left = torch.autograd.grad(actual.reshape(-1)[index], (value, *tested.parameters()), retain_graph=True)
                right = torch.autograd.grad(expected.reshape(-1)[index], (value, *tested.parameters()), retain_graph=True)
                for first, second in zip(left, right):
                    maximum = max(maximum, float((first-second).abs().max()))
                    torch.testing.assert_close(first, second, atol=1e-10, rtol=1e-10)
            p = tested.proj.weight.detach();changed = q.detach()@p@q.detach().transpose(-1, -2)
            sv_error = max(sv_error, float((torch.linalg.svdvals(changed)-torch.linalg.svdvals(p)).abs().max()))
            assert sv_error<1e-12
            expected_eigen = np.sort_complex(np.linalg.eigvals(p.numpy()))
            for altered in changed.numpy():
                error = float(np.max(np.abs(np.sort_complex(np.linalg.eigvals(altered))-expected_eigen)))
                eigen_error = max(eigen_error, error);assert error<1e-10
            direction = torch.randn_like(value)
            _, dq = torch.func.jvp(lambda v: dense(tested, v)[1], (value,), (direction,))
            clock, ds = torch.func.jvp(lambda v: tested.controls(v)[0], (value,), (direction,))
            omega = dq@q.transpose(-1, -2)
            middle = torch.eye(5, dtype=value.dtype)+clock.unsqueeze(-1)*p
            slope = p.expand(3, 5, 5)
            if order==2:
                middle = middle+.5*clock.square().unsqueeze(-1)*(p@p)
                slope = slope+clock.unsqueeze(-1)*(p@p)
            operator_matrix = q@middle@q.transpose(-1, -2)
            predicted = (operator_matrix@direction.unsqueeze(-1)).squeeze(-1)
            predicted += ((omega@operator_matrix-operator_matrix@omega)@value.unsqueeze(-1)).squeeze(-1)
            predicted += ((q@slope@q.transpose(-1, -2))@value.unsqueeze(-1)).squeeze(-1)*ds
            _, actual_direction = torch.func.jvp(tested, (value,), (direction,))
            error = float((predicted-actual_direction).detach().abs().max())
            commutator_error = max(commutator_error, error);assert error<1e-10
            if mode=='global':
                folded = copy.deepcopy(source);folded.orders = order
                with torch.no_grad():folded.proj.weight.copy_(changed[0])
                error = float((folded(value)-actual).detach().abs().max());global_error = max(global_error, error)
                assert error<1e-11
    witness = operator.ConditionalFrameLayer(copy.deepcopy(source))
    with torch.no_grad():
        witness.proj.weight.copy_(torch.diag(torch.linspace(.1, .5, 5, dtype=torch.float64)))
        witness.t_func[2].weight.zero_();witness.t_func[2].bias.zero_()
    anchor = torch.eye(5, dtype=torch.float64)[:1]
    response = torch.stack([torch.autograd.grad(witness(anchor)[0, i], witness.frame_bias, retain_graph=True)[0] for i in range(5)])
    p = torch.diag(witness.proj.weight.detach());m = 1+.6*p+.18*p*p
    expected = torch.zeros_like(response);expected[1:, :] = torch.diag(2*(m[0]-m[1:]))
    torch.testing.assert_close(response, expected, atol=1e-12, rtol=0)
    assert int(torch.linalg.matrix_rank(response))==4
    clock = ((witness.proj.weight+.6*witness.proj.weight@witness.proj.weight)@anchor[0]).detach()
    assert int(torch.linalg.matrix_rank(torch.cat((clock[:, None], response), 1)))==5
    return dict(status='PASS', active_all_output_state_parameter_gradient_max=maximum,
        conditional_singular_value_max=sv_error, conditional_eigenvalue_max=eigen_error,
        complete_commutator_differential_max=commutator_error, global_frame_absorption_max=global_error,
        identity_forward_and_inherited_gradients_bitwise=True, RNG_unchanged=True,
        local_frame_response_rank=4, local_clock_plus_frame_response_rank=5)
