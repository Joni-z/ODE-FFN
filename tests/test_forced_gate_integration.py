import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.func import functional_call, jacrev

from scripts.run_architecture_study import macs
from synthetic.diagnostics import collect
from synthetic.models import ForcedGateSwiGLUFFN, SwiGLUFFN, VelocityNet


class ForcedGateIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def test_initial_full_functions_common_gradients_and_rng_match_swiglu(self):
        for dtype in (torch.float32, torch.float64):
            torch.manual_seed(0)
            base = VelocityNet(3, 'swiglu', width=12, depth=2, ffn_hidden=24).to(dtype)
            rng = torch.get_rng_state().clone()
            with torch.no_grad():
                base.output.weight.normal_(std=.2)
            x, t = torch.randn(7, 3, dtype=dtype), torch.rand(7, 1, dtype=dtype)
            base(x, t).square().sum().backward()
            for solver, clock in [('exact', 'adaptive'), ('k2', 'adaptive'), ('exact', 'static')]:
                with self.subTest(dtype=dtype, solver=solver, clock=clock):
                    torch.manual_seed(0)
                    model = VelocityNet(3, 'ode_forced_gate', width=12, depth=2, ffn_hidden=24,
                                        forced_gate_solver=solver, forced_gate_clock=clock).to(dtype)
                    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
                    with torch.no_grad():
                        model.output.weight.copy_(base.output.weight)
                    for name, value in base.state_dict().items():
                        torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
                    torch.testing.assert_close(model(x, t), base(x, t), atol=0, rtol=0)
                    model(x, t).square().sum().backward()
                    for name, parameter in base.named_parameters():
                        torch.testing.assert_close(dict(model.named_parameters())[name].grad,
                                                   parameter.grad, atol=0, rtol=0)
                    if dtype == torch.float64:
                        jb = torch.autograd.functional.jacobian(lambda z: base(z[None], t[:1])[0], x[0])
                        jm = torch.autograd.functional.jacobian(lambda z: model(z[None], t[:1])[0], x[0])
                        torch.testing.assert_close(jm, jb, atol=2e-13, rtol=2e-13)

    def test_k2_exact_complete_initial_parameter_jacobians(self):
        torch.manual_seed(0)
        low = ForcedGateSwiGLUFFN(3, 6, solver='k2').double()
        torch.manual_seed(0)
        exact = ForcedGateSwiGLUFFN(3, 6, solver='exact').double()
        x = torch.randn(5, 3, dtype=torch.float64)
        def parameter_jacobian(model):
            return jacrev(lambda p: functional_call(model, p, (x,)))(dict(model.named_parameters()))
        a, b = parameter_jacobian(low), parameter_jacobian(exact)
        self.assertEqual(set(a), set(b))
        for name in a:
            torch.testing.assert_close(a[name], b[name], atol=2e-13, rtol=2e-13)
        self.assertGreater(float(a['flow.raw_eigenvalues'].detach().abs().max()), 0.)

    def test_static_gain_absorbs_into_output_columns_and_bias(self):
        model = ForcedGateSwiGLUFFN(5, 12, clock='static').double()
        with torch.no_grad():
            model.flow.raw_eigenvalues.copy_(torch.linspace(-2, 2, 8))
        base = SwiGLUFFN(5, 12).double()
        base.w12.load_state_dict(model.w12.state_dict())
        with torch.no_grad():
            gain = model.flow.gain(torch.zeros(1, 8, dtype=torch.float64))[0]
            base.w3.weight.copy_(model.w3.weight * gain)
            base.w3.bias.copy_(model.w3.bias)
        x = torch.randn(13, 5, dtype=torch.float64)
        torch.testing.assert_close(base(x), model(x), atol=2e-13, rtol=2e-13)
        jb = torch.autograd.functional.jacobian(base, x)
        jm = torch.autograd.functional.jacobian(model, x)
        torch.testing.assert_close(jm, jb, atol=2e-13, rtol=2e-13)

    def test_actual_dense_work_counts_and_live_forced_statistics(self):
        base = VelocityNet(16, 'swiglu')
        model = VelocityNet(16, 'ode_forced_gate')
        self.assertEqual(model.parameter_count() - base.parameter_count(), 3 * 85)
        self.assertEqual(macs(model, 16), macs(base, 16))
        executed, linear = [0], F.linear
        def instrument(x, weight, bias=None):
            executed[0] += (x.numel() // x.shape[-1]) * weight.numel()
            return linear(x, weight, bias)
        with patch.object(F, 'linear', instrument), torch.no_grad():
            model(torch.zeros(1, 16), torch.zeros(1, 1))
        self.assertEqual(executed[0], macs(model, 16))
        self.assertEqual(len(list(model.ode_layers())), 0)
        self.assertEqual(len(list(model.forced_gate_layers())), 3)
        with torch.no_grad():
            for _, flow in model.forced_gate_layers():
                flow.raw_eigenvalues.fill_(.1)
        records = collect(model, torch.randn(8, 16), torch.rand(8, 1))
        self.assertEqual(len(records), 3)
        for record in records:
            self.assertEqual(record['operator_type'], 'conditional_forced_gate')
            self.assertGreater(record['P_sigma_max'], 0.)
            self.assertEqual(len(record['P_eigenvalues']), 85)
            self.assertGreater(record['forcing_abs_max'], 0.)

    def test_rejects_unused_or_combined_interventions(self):
        for change in [dict(forced_gate_solver='unknown'), dict(forced_gate_clock='unknown'),
                       dict(ode_gate_reparam=8.), dict(ode_middle_activation='identity'),
                       dict(ode_operator_mode='additive'), dict(diagonal_solver='exp')]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                VelocityNet(2, 'ode_forced_gate', **change)
        with self.assertRaises(ValueError):
            VelocityNet(2, 'swiglu', forced_gate_clock='static')


if __name__ == '__main__':
    unittest.main()
