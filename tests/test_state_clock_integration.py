import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from scripts.run_architecture_study import macs
from synthetic.diagnostics import collect
from synthetic.models import VelocityNet


class StateClockIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(1)

    @staticmethod
    def options():
        return dict(width=12, depth=2, gate_hidden=6, gate_reference_hidden=12, ffn_hidden=24)

    def test_legacy_constant_clock_states_rng_functions_and_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for solver, order in [('euler', 1), ('frozen', 2)]:
                with self.subTest(dtype=dtype, solver=solver):
                    torch.manual_seed(0)
                    old = VelocityNet(3, 'ode_vector', orders=order, **self.options()).to(dtype)
                    rng = torch.get_rng_state().clone()
                    with torch.no_grad():
                        for _, layer in old.ode_layers():
                            layer.t_func[-1].weight.zero_()
                            layer.t_func[-1].bias.zero_()
                        old.output.weight.normal_(std=.2)
                    x, t = torch.randn(7, 3, dtype=dtype), torch.rand(7, 1, dtype=dtype)
                    torch.manual_seed(0)
                    new = VelocityNet(3, 'ode_state_clock', state_clock_solver=solver,
                                      **self.options()).to(dtype)
                    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
                    with torch.no_grad():
                        new.output.weight.copy_(old.output.weight)
                    for name, value in old.state_dict().items():
                        torch.testing.assert_close(new.state_dict()[name], value, atol=0, rtol=0)
                    torch.testing.assert_close(new(x, t), old(x, t), atol=0, rtol=0)
                    old(x, t).square().sum().backward()
                    new(x, t).square().sum().backward()
                    for name, parameter in old.named_parameters():
                        torch.testing.assert_close(dict(new.named_parameters())[name].grad,
                                                   parameter.grad, atol=0, rtol=0)

    def test_frozen_midpoint_zero_head_pair_and_nonzero_head_tangent_limit(self):
        for dtype in (torch.float32, torch.float64):
            torch.manual_seed(0)
            frozen = VelocityNet(3, 'ode_state_clock', state_clock_solver='frozen', **self.options()).to(dtype)
            rng = torch.get_rng_state().clone()
            torch.manual_seed(0)
            dynamic = VelocityNet(3, 'ode_state_clock', state_clock_solver='midpoint', **self.options()).to(dtype)
            torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
            for name, value in frozen.state_dict().items():
                torch.testing.assert_close(dynamic.state_dict()[name], value, atol=0, rtol=0)
            x, t = torch.randn(7, 3, dtype=dtype), torch.rand(7, 1, dtype=dtype)
            target = torch.randn(7, 3, dtype=dtype)
            for model in (frozen, dynamic):
                (model(x, t)-target).square().sum().backward()
            for name, parameter in frozen.named_parameters():
                torch.testing.assert_close(dict(dynamic.named_parameters())[name].grad,
                                           parameter.grad, atol=0, rtol=0)
            with torch.no_grad():
                frozen.output.weight.normal_(std=.2)
                dynamic.output.weight.copy_(frozen.output.weight)
            torch.testing.assert_close(dynamic(x, t), frozen(x, t), atol=0, rtol=0)
            for model in (frozen, dynamic):
                model.zero_grad()
                (model(x, t)-target).square().sum().backward()
            controller_difference = 0.
            for name, parameter in frozen.named_parameters():
                current = dict(dynamic.named_parameters())[name]
                if '.t_func.2.weight' in name:
                    controller_difference = max(controller_difference,
                                                float((current.grad-parameter.grad).detach().abs().max()))
                elif '.t_func.' not in name:
                    tolerance = 3e-6 if dtype == torch.float32 else 3e-13
                    torch.testing.assert_close(current.grad, parameter.grad, atol=tolerance, rtol=tolerance)
            self.assertGreater(controller_difference, 1e-8)
            if dtype == torch.float64:
                a = torch.autograd.functional.jacobian(lambda z: frozen(z[None], t[:1])[0], x[0])
                b = torch.autograd.functional.jacobian(lambda z: dynamic(z[None], t[:1])[0], x[0])
                torch.testing.assert_close(a, b, atol=3e-13, rtol=3e-13)

    def test_actual_macs_parameter_sharing_and_stage_logging(self):
        models = {solver: VelocityNet(16, 'ode_state_clock', gate_hidden=32, state_clock_solver=solver)
                  for solver in ['euler', 'frozen', 'midpoint']}
        for solver, expected in [('euler', 81984), ('frozen', 106560), ('midpoint', 131136)]:
            model = models[solver]
            self.assertEqual(model.parameter_count(), 83664)
            self.assertEqual(macs(model, 16), expected)
            executed, linear = [0], F.linear
            def instrument(x, weight, bias=None):
                executed[0] += (x.numel()//x.shape[-1])*weight.numel()
                return linear(x, weight, bias)
            with patch.object(F, 'linear', instrument), torch.no_grad():
                model(torch.zeros(1, 16), torch.zeros(1, 1))
            self.assertEqual(executed[0], expected)
            self.assertEqual(len(list(model.ode_layers())), 0)
            self.assertEqual(len(list(model.state_clock_layers())), 6)
            self.assertEqual(len(list(model.parameters())), len({id(p) for p in model.parameters()}))
            with torch.no_grad():
                for _, layer in model.state_clock_layers():
                    layer.t_func[-1].weight.normal_(std=.1)
            stats = collect(model, torch.randn(8, 16), torch.rand(8, 1))
            self.assertEqual(len(stats), 6)
            for row in stats:
                self.assertEqual(row['operator_type'], 'state_clock_vector_field')
                self.assertEqual(row['solver'], solver)
                self.assertEqual(row['activation']['midpoint_clock_is_executed_stage'], solver == 'midpoint')
                self.assertGreater(row['activation']['gate_change_abs_max'], 0.)

    def test_unused_solver_and_legacy_interventions_rejected(self):
        for change in [dict(state_clock_solver='unknown'), dict(orders=1), dict(ode_gate_reparam=8.),
                       dict(ode_middle_activation='identity'), dict(ode_clock_mode='reuse'),
                       dict(diagonal_solver='exp'), dict(forced_gate_clock='static')]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                VelocityNet(2, 'ode_state_clock', **change)
        with self.assertRaises(ValueError):
            VelocityNet(2, 'swiglu', state_clock_solver='frozen')


if __name__ == '__main__':
    unittest.main()
