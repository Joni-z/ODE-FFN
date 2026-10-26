import copy
import io
import json
import math
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from synthetic.forced_gate import ForcedGateResponse, phi1


class ForcedGateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(1)

    def test_phi1_zero_boundary_values_first_and_second_derivatives(self):
        locations = [-1., -.050001, -.05, -.049999, -1e-7, -1e-15,
                     0., 1e-15, 1e-7, .049999, .05, .050001, 1.]
        # High-order series is an independent reference on this bounded interval.
        for dtype in (torch.float32, torch.float64):
            z = torch.tensor(locations, dtype=dtype, requires_grad=True)
            value = phi1(z)
            first = torch.autograd.grad(value.sum(), z, create_graph=True)[0]
            second = torch.autograd.grad(first.sum(), z)[0]
            zz = z.detach().double()
            ref = sum(zz.pow(k)*(1/math.factorial(k+1)) for k in range(24))
            ref_first = sum(zz.pow(k)*((k+1)/math.factorial(k+2)) for k in range(23))
            ref_second = sum(zz.pow(k)*((k+2)*(k+1)/math.factorial(k+3)) for k in range(22))
            # Float32 quotient Hessians lose precision just outside .05
            # (observed maximum error 4.81e-5); values/first derivatives keep
            # stricter tolerances. At zero the series gives the exact limit.
            tolerances = (2e-7, 8e-6, 8e-5) if dtype == torch.float32 else (3e-12,)*3
            for (actual, expected), tolerance in zip(
                    ((value, ref), (first, ref_first), (second, ref_second)), tolerances):
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual.double(), expected, atol=tolerance, rtol=tolerance)
            index = locations.index(0.)
            self.assertEqual(float(value[index].detach()), 1.)
            self.assertEqual(float(first[index].detach()), .5)
            self.assertAlmostEqual(float(second[index]), 1/3, places=6 if dtype == torch.float32 else 15)

    def test_augmented_matrix_response_and_input_derivatives(self):
        a = torch.tensor([[-8., -.7, .1], [1., 3., 8.]], dtype=torch.float64)
        for solver in ('k2', 'exact'):
            for clock in ('adaptive', 'static'):
                model = ForcedGateResponse(3, solver, clock).double()
                with torch.no_grad():
                    model.raw_eigenvalues.copy_(torch.tensor([-1.3, .4, 1.1]))
                baseline = F.silu(a)
                c = model.eigenvalues()*model.duration(a)
                matrix = torch.zeros(*a.shape, 2, 2, dtype=a.dtype)
                matrix[..., 0, 0] = c
                matrix[..., 0, 1] = baseline
                reference = (torch.matrix_exp(matrix) if solver == 'exact'
                             else torch.eye(2, dtype=a.dtype)+matrix+.5*matrix@matrix)[..., 0, 1]
                torch.testing.assert_close(model(a), reference, atol=2e-14, rtol=2e-13)
                xx = a.clone().requires_grad_(True)
                actual = torch.autograd.grad(model(xx).sum(), xx)[0]
                torch.testing.assert_close(actual, model.derivative(a), atol=3e-13, rtol=3e-13)

    def test_initial_baseline_input_gradient_rng_and_new_eta_tangent(self):
        for dtype in (torch.float32, torch.float64):
            for solver, clock in (('k2', 'adaptive'), ('exact', 'adaptive'), ('exact', 'static')):
                before = torch.get_rng_state().clone()
                model = ForcedGateResponse(4, solver, clock).to(dtype)
                torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)
                a = torch.randn(7, 4, dtype=dtype, requires_grad=True)
                output = model(a)
                torch.testing.assert_close(output, F.silu(a), atol=0, rtol=0)
                actual = torch.autograd.grad(output.sum(), a, retain_graph=True)[0]
                expected = torch.autograd.grad(F.silu(a).sum(), a)[0]
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                eta_gradient = torch.autograd.grad(output.sum(), model.raw_eigenvalues)[0]
                target = (.5*F.silu(a.detach())*model.duration(a.detach())).sum(0)
                tolerance = 2e-7 if dtype == torch.float32 else 2e-15
                torch.testing.assert_close(eta_gradient, target, atol=tolerance, rtol=tolerance)

    def test_adaptive_initial_full_parameter_jacobians_match(self):
        class ToySwi(nn.Module):
            def __init__(self, solver):
                super().__init__()
                self.w12 = nn.Linear(3, 8)
                self.w3 = nn.Linear(4, 2)
                self.response = ForcedGateResponse(4, solver)
            def forward(self, x):
                gate, value = self.w12(x).chunk(2, -1)
                return self.w3(self.response(gate)*value)
        models = []
        for solver in ('k2', 'exact'):
            torch.manual_seed(0)
            models.append(ToySwi(solver).double())
        x = torch.randn(3, 3, dtype=torch.float64)
        jacobians = []
        for model in models:
            values = model(x).reshape(-1)
            parameters = tuple(model.parameters())
            rows = [torch.cat([g.reshape(-1) for g in torch.autograd.grad(
                value, parameters, retain_graph=True)]) for value in values]
            jacobians.append(torch.stack(rows))
        torch.testing.assert_close(models[0](x), models[1](x), atol=0, rtol=0)
        torch.testing.assert_close(jacobians[0], jacobians[1], atol=0, rtol=0)
        torch.testing.assert_close(jacobians[0]@jacobians[0].T,
                                   jacobians[1]@jacobians[1].T, atol=0, rtol=0)

    def test_relative_gains_statistics_and_state_round_trip(self):
        a = torch.linspace(-40, 40, 103, dtype=torch.float64)[:, None].expand(-1, 3)
        for solver in ('k2', 'exact'):
            model = ForcedGateResponse(3, solver).double()
            with torch.no_grad():model.raw_eigenvalues.copy_(torch.tensor([-30., 0., 30.]))
            gain = model.gain(a)
            lower, upper = ((.5, 1.5) if solver == 'k2' else (1-math.exp(-1), math.e-1))
            self.assertGreaterEqual(float(gain.min().detach()), lower-1e-15)
            self.assertLessEqual(float(gain.max().detach()), upper+1e-15)
            before = copy.deepcopy(model.state_dict())
            stats = model.activation_stats(a)
            json.dumps(stats, allow_nan=False)
            self.assertEqual(stats['scope'], 'observed_gate_response_only')
            torch.testing.assert_close(model.state_dict()['raw_eigenvalues'], before['raw_eigenvalues'], atol=0, rtol=0)
            buffer = io.BytesIO();torch.save(before, buffer);buffer.seek(0)
            restored = ForcedGateResponse(3, solver).double()
            restored.load_state_dict(torch.load(buffer, weights_only=True))
            torch.testing.assert_close(restored(a), model(a), atol=0, rtol=0)

    def test_invalid_settings_and_input_shapes(self):
        for options in (dict(features=0), dict(features=True), dict(features=2.5),
                        dict(features=3, solver='k1'), dict(features=3, clock='time')):
            with self.assertRaises(ValueError):ForcedGateResponse(**options)
        model = ForcedGateResponse(3)
        for value in (torch.zeros(2, 4), torch.zeros(3, dtype=torch.int64), torch.tensor(0.)):
            with self.assertRaises(ValueError):model(value)
        with self.assertRaises(ValueError):model.activation_stats(torch.zeros(0, 3))

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS unavailable')
    def test_mps_float32_statistics_do_not_require_device_float64(self):
        model = ForcedGateResponse(3).to('mps')
        a = torch.linspace(-3, 3, 12, device='mps').reshape(4, 3).requires_grad_(True)
        value = model(a)
        gradients = torch.autograd.grad(value.sum(), (a, model.raw_eigenvalues))
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        json.dumps(model.activation_stats(a), allow_nan=False)


if __name__ == '__main__':
    unittest.main()
