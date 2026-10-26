import copy
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import solve_ivp
import torch
from torch import nn
from torch.nn import functional as F

from synthetic.models import OriginalODELayer, VectorODELayer
from synthetic.state_clock import StateClockLayer


class StateClockTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(1)

    def layer(self, solver='midpoint', width=3, hidden=4):
        return StateClockLayer(VectorODELayer(width, hidden, reference_gate_hidden=hidden,
                                             orders=1), solver).double()

    def active(self, solver='midpoint'):
        layer = self.layer(solver, 2, 3)
        with torch.no_grad():
            layer.proj.weight.copy_(torch.tensor([[.3, -.2], [.1, -.4]]))
            layer.t_func[0].weight.copy_(torch.tensor([[1., -.4], [.3, .8], [-.6, .2]]))
            layer.t_func[0].bias.copy_(torch.tensor([.2, -.1, .3]))
            layer.t_func[-1].weight.copy_(torch.tensor([[3., -2., 1.], [-1., 2., 4.]]))
            layer.t_func[-1].bias.copy_(torch.tensor([.2, -.3]))
        return layer

    def test_reference_state_rng_and_frozen_polynomial(self):
        reference = VectorODELayer(3, 4, reference_gate_hidden=4, orders=2).double()
        before = torch.get_rng_state().clone()
        expected = copy.deepcopy(reference)
        with torch.no_grad():
            expected.t_func[-1].weight.zero_(); expected.t_func[-1].bias.zero_()
        layer = StateClockLayer(reference, 'frozen')
        torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)
        self.assertEqual(set(layer.state_dict()), set(expected.state_dict()))
        self.assertEqual(set(layer._modules), {'proj', 't_func'})
        self.assertIs(layer.proj, reference.proj)
        self.assertIs(layer.t_func, reference.t_func)
        x = torch.randn(7, 3, dtype=torch.float64)
        torch.testing.assert_close(layer(x), expected(x), atol=0, rtol=0)
        # Frozen identity also holds after a genuinely input-dependent clock.
        with torch.no_grad():
            layer.t_func[-1].weight.fill_(.7)
            expected.load_state_dict(layer.state_dict())
        torch.testing.assert_close(layer(x), expected(x), atol=0, rtol=0)

    def test_constant_clock_matrix_exponential_and_error_order(self):
        x = torch.tensor([[.8, -.5]], dtype=torch.float64)
        matrix = torch.tensor([[.3, -.2], [.1, -.4]], dtype=torch.float64)
        errors = {name: [] for name in ('euler', 'frozen', 'midpoint')}
        for epsilon in (.2, .1, .05):
            for name in errors:
                layer = self.layer(name, 2)
                with torch.no_grad():layer.proj.weight.copy_(epsilon*matrix)
                generator = .6*epsilon*matrix
                exact = F.linear(x, torch.matrix_exp(generator))
                polynomial = torch.eye(2, dtype=x.dtype)+generator
                if name != 'euler':polynomial = polynomial+.5*generator@generator
                torch.testing.assert_close(layer(x), F.linear(x, polynomial), atol=2e-16, rtol=2e-15)
                errors[name].append(float((layer(x)-exact).norm().detach()))
        for name, values in errors.items():
            target = 4 if name == 'euler' else 8
            self.assertTrue(all(.85*target < left/right < 1.15*target for left, right in zip(values, values[1:])))

    def test_dynamic_explicit_midpoint_values_and_all_gradients(self):
        layer = self.active()
        x = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
        f = lambda z: layer.step_size(z)*F.linear(z, layer.proj.weight)
        explicit = x+f(x+.5*f(x))
        actual = layer(x)
        torch.testing.assert_close(actual, explicit, atol=3e-16, rtol=3e-15)
        parameters = (x, *layer.parameters())
        cotangent = torch.tensor([[.1, -.3], [.7, .8], [-.2, .4], [.9, -.6]], dtype=x.dtype)
        first = torch.autograd.grad((actual*cotangent).sum(), parameters)
        second = torch.autograd.grad((explicit*cotangent).sum(), parameters)
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, atol=2e-15, rtol=2e-13)

    def test_clock_taylor_coefficient_and_complete_input_jacobian(self):
        layer = self.active()
        x = torch.tensor([.8, -.5], dtype=torch.float64)
        p = layer.proj.weight
        s = layer.step_size(x)
        first = s*(p@x)
        gate_j = torch.autograd.functional.jacobian(layer.step_size, x)
        jvp = layer.clock_jvp(x, first)
        torch.testing.assert_close(jvp, gate_j@first, atol=1e-16, rtol=2e-14)
        f = lambda z: layer.step_size(z)*(p@z)
        jf = lambda z: torch.diag(layer.step_size(z))@p+torch.diag(p@z)@torch.autograd.functional.jacobian(layer.step_size, z)
        taylor = .5*(s*(p@first)+(p@x)*jvp)
        torch.testing.assert_close(taylor, .5*torch.autograd.functional.jvp(f, x, first)[1], atol=1e-16, rtol=2e-14)
        midpoint = x+.5*first
        expected = torch.eye(2, dtype=x.dtype)+jf(midpoint)@(torch.eye(2, dtype=x.dtype)+.5*jf(x))
        torch.testing.assert_close(torch.autograd.functional.jacobian(layer, x), expected, atol=3e-16, rtol=2e-14)
        frozen = copy.deepcopy(layer); frozen.solver = 'frozen'
        a = torch.diag(s)@p
        b = torch.diag(p@x)@gate_j
        expected_fr = torch.eye(2, dtype=x.dtype)+a+b+.5*(torch.diag(p@first)@gate_j+a@(a+b))
        torch.testing.assert_close(torch.autograd.functional.jacobian(frozen, x), expected_fr, atol=3e-16, rtol=2e-14)

    def test_local_error_vs_actual_state_dependent_ode(self):
        layer = self.active()
        x = torch.tensor([.8, -.5], dtype=torch.float64)
        matrix = layer.proj.weight.detach().clone()
        errors = {'frozen': [], 'midpoint': []}
        for epsilon in (.2, .1, .05, .025):
            with torch.no_grad():layer.proj.weight.copy_(epsilon*matrix)
            def rhs(_, state):
                with torch.no_grad():
                    z = torch.from_numpy(state)
                    return (layer.step_size(z)*F.linear(z, layer.proj.weight)).numpy()
            result = solve_ivp(rhs, (0., 1.), x.numpy(), method='DOP853', rtol=2e-13, atol=2e-15)
            self.assertTrue(result.success)
            exact = torch.from_numpy(result.y[:, -1])
            for solver in errors:
                layer.solver = solver
                errors[solver].append(float((layer(x)-exact).norm().detach()))
        for solver, values in errors.items():
            target = 4 if solver == 'frozen' else 8
            self.assertTrue(all(.8*target < a/b < 1.2*target for a, b in zip(values[-3:], values[-2:])), (solver, values))

    def test_paired_functions_P_input_and_controller_tangents(self):
        frozen = self.layer('frozen')
        dynamic = copy.deepcopy(frozen); dynamic.solver = 'midpoint'; dynamic.controller_calls = 2
        x = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        # A nonzero receiving head exposes controller responses hidden by the
        # complete VelocityNet's formally zero initial output head.
        head = torch.tensor([.4, -.7, .2], dtype=x.dtype)
        torch.testing.assert_close(frozen(x), dynamic(x), atol=0, rtol=0)
        a = torch.autograd.grad((frozen(x)*head).sum(), (x, frozen.proj.weight, *frozen.t_func.parameters()))
        b = torch.autograd.grad((dynamic(x)*head).sum(), (x, dynamic.proj.weight, *dynamic.t_func.parameters()))
        for index in (0, 1, 2, 3, 5):
            torch.testing.assert_close(a[index], b[index], atol=2e-16, rtol=2e-14)
        self.assertGreater(float((a[4]-b[4]).norm()), 1e-6)
        self.assertEqual(int(torch.count_nonzero(a[2])), 0)
        # Exact linearized difference: diag(P*h_mid)[q(h_mid)-q(h)].
        ph = F.linear(x.detach(), frozen.proj.weight.detach())
        middle = x.detach()+.3*ph
        receiving = F.linear(middle, frozen.proj.weight.detach())*head
        features = F.silu(frozen.t_func[0](middle))-F.silu(frozen.t_func[0](x.detach()))
        expected = (.8/(4*10))*(receiving.T@features)
        torch.testing.assert_close(b[4]-a[4], expected, atol=2e-16, rtol=3e-13)

    def test_executed_dense_cost_statistics_and_roundtrip(self):
        original_linear = F.linear
        for solver, expected in (('euler', 8192), ('frozen', 12288), ('midpoint', 16384)):
            layer = self.layer(solver, 64, 32)
            count = [0]
            def counted(x, weight, bias=None):
                count[0] += (x.numel()//x.shape[-1])*weight.numel()
                return original_linear(x, weight, bias)
            x = torch.randn(1, 64, dtype=torch.float64)
            with patch.object(F, 'linear', counted), torch.no_grad():layer(x)
            self.assertEqual(count[0], expected)
            self.assertEqual(sum(p.numel() for p in layer.parameters()), 8288)
            before = copy.deepcopy(layer.state_dict())
            stats = layer.activation_stats(x)
            json.dumps(stats, allow_nan=False)
            self.assertEqual(stats['gate_change_rms'], 0.)
            self.assertEqual(stats['clock_taylor_probe_rms'], 0.)
            self.assertEqual(stats['midpoint_clock_is_executed_stage'], solver == 'midpoint')
            for key, value in before.items():torch.testing.assert_close(layer.state_dict()[key], value, atol=0, rtol=0)
            buffer = io.BytesIO(); torch.save(before, buffer); buffer.seek(0)
            restored = self.layer(solver, 64, 32)
            restored.load_state_dict(torch.load(buffer, weights_only=True))
            torch.testing.assert_close(restored(x), layer(x), atol=0, rtol=0)

    def test_invalid_references_solver_and_inputs(self):
        with self.assertRaises(ValueError):StateClockLayer(OriginalODELayer(3, 4))
        with self.assertRaises(ValueError):self.layer('rk4')
        for kwargs in (dict(gate_transform='affine'), dict(operator_mode='additive'), dict(spectral_cap=1.)):
            reference = VectorODELayer(3, 4, orders=1, **kwargs)
            with self.assertRaises(ValueError):StateClockLayer(reference)
        layer = self.layer()
        for x in (torch.zeros(2, 4), torch.zeros(3, dtype=torch.int64), torch.tensor(0.)):
            with self.assertRaises(ValueError):layer(x)
        with self.assertRaises(ValueError):layer.activation_stats(torch.zeros(0, 3))


if __name__ == '__main__':
    unittest.main()
