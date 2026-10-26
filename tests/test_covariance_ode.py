import unittest

import torch

from scripts.run_architecture_study import macs
from synthetic.diagnostics import collect
from synthetic.models import CovarianceODELayer, VelocityNet


class CovarianceODETests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.options = dict(orders=1, gate_hidden=32, gate_reference_hidden=64)

    def model(self, mode):
        torch.manual_seed(0)
        return VelocityNet(2, 'ode_covariance', **self.options, ode_covariance_mode=mode)

    def test_paired_models_keep_surrounding_tensors_and_initial_function(self):
        tied = self.model('tied')
        rng = torch.get_rng_state()
        untied = self.model('untied')
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
        linear = self.model('linearized')
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
        torch.manual_seed(0)
        reference = VelocityNet(2, 'ode_vector', **self.options)
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
        for name, value in tied.state_dict().items():
            torch.testing.assert_close(value, untied.state_dict()[name], atol=0, rtol=0)
            torch.testing.assert_close(value, linear.state_dict()[name], atol=0, rtol=0)
            if '.field.' not in name:
                torch.testing.assert_close(value, reference.state_dict()[name], atol=0, rtol=0)
        x, t = torch.randn(11, 2), torch.rand(11, 1)
        # Reveal the hidden function instead of comparing the zero output head.
        head = torch.randn_like(tied.output.weight)
        with torch.no_grad():
            tied.output.weight.copy_(head)
            untied.output.weight.copy_(head)
        torch.testing.assert_close(tied(x, t), untied(x, t), atol=0, rtol=0)
        self.assertEqual(tied.parameter_count(), 69576)
        self.assertEqual(untied.parameter_count(), 81864)
        self.assertEqual(linear.parameter_count(), tied.parameter_count())
        self.assertEqual(macs(tied, 2), 80192)
        self.assertEqual(macs(untied, 2), macs(tied, 2))
        self.assertEqual(macs(linear, 2), macs(tied, 2))
        # State-dict restoration also covers independent decoder parameters.
        restored = self.model('untied')
        restored.load_state_dict(untied.state_dict())
        torch.testing.assert_close(restored(x, t), untied(x, t), atol=0, rtol=0)

    def test_update_jacobian_includes_gate_derivative_and_is_not_conservative(self):
        torch.manual_seed(0)
        layer = CovarianceODELayer(4, 3, 4, heads=1, atoms=3).double()
        with torch.no_grad():
            layer.field.gamma.fill_(-.2)
            layer.t_func[-1].weight.mul_(20)
        x = torch.randn(4, dtype=torch.float64)
        duration = layer.step_size(x)
        field = layer.field(x)
        derivative = layer.field.field_jacobian(x)
        gate_derivative = torch.func.jacrev(layer.step_size)(x)
        expected = torch.eye(4, dtype=x.dtype) + duration[:, None] * derivative + field[:, None] * gate_derivative
        actual = torch.func.jacrev(layer)(x)
        torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-12)
        self.assertGreater(float((actual - actual.T).detach().abs().max()), .0001)

    def test_diagnostics_log_actual_field_matrices_and_gradients_flow(self):
        model = self.model('tied')
        x, t = torch.randn(8, 2), torch.rand(8, 1)
        with torch.no_grad():
            model.output.weight.normal_()
        model(x, t).square().mean().backward()
        for _, layer in model.covariance_layers():
            for parameter in (layer.field.weight, layer.field.gamma, layer.t_func[0].weight):
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        records = collect(model, x, t)
        self.assertEqual(len(records), 6)
        self.assertEqual(list(model.ode_layers()), [])
        for record in records:
            self.assertEqual(record['operator_type'], 'covariance_internal_field')
            self.assertNotIn('raw', record)
            self.assertLess(record['probe_skew_frobenius_mean'], 1e-7)


if __name__ == '__main__':
    unittest.main()
