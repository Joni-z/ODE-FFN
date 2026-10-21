"""Analytic oracle and anchored-gate derivative contracts."""
import unittest

import numpy as np
import torch

from scripts.check_gate_geometry import (diagnostic_batch, gate_context,
                                         oracle_geometry, trajectory_states)
from synthetic.data import suite
from synthetic.models import VelocityNet


class GateGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def model(self, kind: str) -> VelocityNet:
        model = VelocityNet(2, kind, width=8, depth=2, gate_hidden=8, ffn_hidden=16).double().eval()
        with torch.no_grad():
            model.output.weight.normal_(std=.2)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def test_oracle_jacobian_includes_responsibility_derivative(self) -> None:
        x = np.random.default_rng(0).normal(size=(8, 2))
        for name in ('gaussian_aniso_2d', 'gmm_ring8_2d', 'gmm_aniso4_2d'):
            dist = suite()[name]
            for t in (.1, .5, .9):
                actual = oracle_geometry(dist, x, t)['oracle_jacobian']
                columns = []
                for axis in range(2):
                    delta = np.zeros_like(x)
                    delta[:, axis] = 1e-5
                    columns.append((dist.velocity(x+delta, t)-dist.velocity(x-delta, t))/2e-5)
                np.testing.assert_allclose(actual, np.stack(columns, axis=-1), atol=2e-6, rtol=2e-6)

    def test_anchored_map_matches_native_value_and_finite_differences(self) -> None:
        model = self.model('ode_original')
        x, t = torch.randn(6, 2, dtype=torch.float64), torch.full((6, 1), .5, dtype=torch.float64)
        result = diagnostic_batch(model, x, t, finite_difference=True)
        correction = result['jacobian']-result['anchored_jacobian']
        self.assertGreater(np.linalg.norm(correction), 1e-8)
        self.assertEqual(result['anchor_max_error'], 0)
        self.assertTrue(all('step_size' not in layer.__dict__ for _, layer in model.ode_layers()))

    def test_constant_and_baseline_zero_controls(self) -> None:
        x, t = torch.randn(5, 2, dtype=torch.float64), torch.full((5, 1), .3, dtype=torch.float64)
        for kind in ('ode_fixed', 'ode_scalar', 'ffn', 'swiglu'):
            result = diagnostic_batch(self.model(kind), x, t, finite_difference=True)
            np.testing.assert_array_equal(result['jacobian'], result['anchored_jacobian'])

    def test_gate_method_restored_after_exception(self) -> None:
        model = self.model('ode_original')
        with self.assertRaisesRegex(RuntimeError, 'test'):
            with gate_context(model, {}, capture=True):
                raise RuntimeError('test')
        self.assertTrue(all('step_size' not in layer.__dict__ for _, layer in model.ode_layers()))

    def test_trajectory_capture_uses_requested_grid(self) -> None:
        model = self.model('ffn')
        source = torch.randn(4, 2, dtype=torch.float64)
        first = trajectory_states(model, source, (.1, .5, .9), 100)
        second = trajectory_states(model, source, (.1, .5, .9), 100)
        self.assertEqual(set(first), {.1, .5, .9})
        for t in first:
            torch.testing.assert_close(first[t], second[t], atol=0, rtol=0)
        with self.assertRaises(ValueError):
            trajectory_states(model, source, (.123,), 100)


if __name__ == '__main__':
    unittest.main()
