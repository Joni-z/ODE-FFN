"""Independent mathematical and initialization checks for the new prototype."""
import copy
import math
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from scripts.run_architecture_study import macs
from synthetic.diagonal import DiagonalFlow, DiagonalODEFFN
from synthetic.models import OriginalODELayer, StandardFFN


def parameter_jacobian(model, x):
    outputs = model(x).flatten()
    parameters = tuple(model.parameters())
    return torch.stack([
        torch.cat([gradient.flatten() for gradient in torch.autograd.grad(
            value, parameters, retain_graph=True)]) for value in outputs
    ])


def silu_derivative(x):
    probability = torch.sigmoid(x)
    return probability * (1 + x * (1 - probability))


class DiagonalFlowTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(1)

    def test_exponential_agrees_with_frozen_diagonal_matrix_exponential(self):
        x = torch.randn(3, 4, dtype=torch.float64)
        for clock in ("adaptive", "static"):
            with self.subTest(clock=clock):
                layer = DiagonalFlow(4, "exp", clock, .8).double()
                with torch.no_grad():
                    layer.raw_eigenvalues.copy_(torch.tensor([-.9, -.2, .3, 1.1]))
                durations = .2 + .8 * torch.sigmoid(x) if clock == "adaptive" else torch.full_like(x, .6)
                generator = torch.diag_embed(layer.eigenvalues() * durations)
                expected = (torch.matrix_exp(generator) @ x.unsqueeze(-1)).squeeze(-1)
                torch.testing.assert_close(layer(x), expected, atol=2e-15, rtol=2e-13)
                self.assertFalse(isinstance(layer, OriginalODELayer))

    def test_input_jacobians_include_clock_response_for_all_solvers(self):
        x = torch.tensor([-2.1, -.7, .4, 1.8], dtype=torch.float64)
        for solver in ("k1", "k2", "exp"):
            for clock in ("adaptive", "static"):
                with self.subTest(solver=solver, clock=clock):
                    layer = DiagonalFlow(4, solver, clock, .9).double()
                    with torch.no_grad():
                        layer.raw_eigenvalues.copy_(torch.tensor([-.9, -.3, .2, .7]))
                    probability = torch.sigmoid(x)
                    duration = .2 + .8 * probability if clock == "adaptive" else torch.full_like(x, .6)
                    ds = .8 * probability * (1-probability) if clock == "adaptive" else torch.zeros_like(x)
                    rate = layer.eigenvalues()
                    y = rate * duration
                    polynomial = {"k1": 1+y, "k2": 1+y+y*y/2, "exp": y.exp()}[solver]
                    polynomial_prime = {"k1": torch.ones_like(y), "k2": 1+y, "exp": y.exp()}[solver]
                    expected = polynomial + x * polynomial_prime * rate * ds
                    actual = torch.autograd.functional.jacobian(layer, x)
                    torch.testing.assert_close(actual, torch.diag(expected), atol=2e-15, rtol=2e-13)
                    torch.testing.assert_close(layer.derivative(x), expected, atol=2e-15, rtol=2e-13)
                    if clock == "adaptive":
                        self.assertGreater(float((expected-polynomial).detach().abs().max()), .01)

                    ffn = DiagonalODEFFN(3, 4, solver, clock, .9).double()
                    with torch.no_grad():
                        ffn.ode1.raw_eigenvalues.copy_(layer.raw_eigenvalues)
                        ffn.ode2.raw_eigenvalues.copy_(-layer.raw_eigenvalues)
                    state = x[:3]
                    z = ffn.net[0](state)
                    first = ffn.ode1(z)
                    middle = F.silu(first)
                    diagonal = ffn.ode2.derivative(middle) * silu_derivative(first) * ffn.ode1.derivative(z)
                    full_expected = ffn.net[2].weight @ torch.diag(diagonal) @ ffn.net[0].weight
                    full_actual = torch.autograd.functional.jacobian(ffn, state)
                    torch.testing.assert_close(full_actual, full_expected, atol=3e-15, rtol=3e-13)

    def test_conditional_exponential_is_not_recomputed_nonlinear_flow(self):
        layer = DiagonalFlow(1, "exp").double()
        with torch.no_grad():
            layer.raw_eigenvalues.fill_(math.atanh(.7))
        x = torch.tensor([1.2], dtype=torch.float64)
        rate = layer.eigenvalues()
        half = x * torch.exp(.5 * rate * layer.duration(x))
        reclosed_half = half * torch.exp(.5 * rate * layer.duration(half))
        self.assertGreater(float((reclosed_half-layer(x)).detach().abs().max()), .005)

    def test_positive_derivative_bounds_and_explicit_k1_fold(self):
        # Regression checks of the analytic global bounds, not a finite-grid proof.
        # |z sigmoid'(z)| < 1/4 gives exp >= .8/e and K2 >= .1 for beta<=1.
        x = torch.linspace(-40, 40, 2001, dtype=torch.float64).unsqueeze(-1).expand(-1, 5)
        raw = torch.tensor([-30., -1., 0., 1., 30.], dtype=torch.float64)
        for solver, lower in (("k2", .1), ("exp", .8/math.e)):
            layer = DiagonalFlow(5, solver).double()
            with torch.no_grad():
                layer.raw_eigenvalues.copy_(raw)
            self.assertGreaterEqual(float(layer.derivative(x).detach().min()), lower)
            self.assertLessEqual(float(layer.eigenvalues().detach().abs().max()), 1.)
        k1 = DiagonalFlow(1, "k1").double()
        with torch.no_grad():
            k1.raw_eigenvalues.fill_(math.atanh(-.99))
        witness = torch.tensor([2.4], dtype=torch.float64)
        self.assertLess(float(k1.derivative(witness).detach()), -.06)
        self.assertLess(float(torch.autograd.functional.jacobian(k1, witness).detach()), -.06)
        self.assertGreater(float(k1(witness).detach()), 0.)

    def test_zero_eigenvalues_match_standard_ffn_keys_values_and_rng(self):
        for solver in ("k1", "k2", "exp"):
            for clock in ("adaptive", "static"):
                for dtype in (torch.float32, torch.float64):
                    with self.subTest(solver=solver, clock=clock, dtype=dtype):
                        torch.manual_seed(0)
                        baseline = StandardFFN(4, 7).to(dtype)
                        baseline_rng = torch.get_rng_state().clone()
                        torch.manual_seed(0)
                        candidate = DiagonalODEFFN(4, 7, solver, clock).to(dtype)
                        torch.testing.assert_close(torch.get_rng_state(), baseline_rng, atol=0, rtol=0)
                        shared = baseline.state_dict()
                        state = candidate.state_dict()
                        self.assertEqual(set(state)-set(shared), {"ode1.raw_eigenvalues", "ode2.raw_eigenvalues"})
                        for key, value in shared.items():
                            torch.testing.assert_close(state[key], value, atol=0, rtol=0)
                        x = torch.randn(2, 3, 4, dtype=dtype)
                        torch.testing.assert_close(candidate(x), baseline(x), atol=0, rtol=0)
                        rng_before = torch.get_rng_state().clone()
                        DiagonalFlow(7, solver, clock)
                        torch.testing.assert_close(torch.get_rng_state(), rng_before, atol=0, rtol=0)

    def test_adaptive_initial_full_parameter_jacobian_and_ntk_equality(self):
        models = []
        for solver in ("k1", "k2", "exp"):
            torch.manual_seed(0)
            models.append(DiagonalODEFFN(3, 4, solver, "adaptive", .8).double())
        x = torch.randn(3, 3, dtype=torch.float64)
        expected_names = list(dict(models[0].named_parameters()))
        jacobians = []
        for model in models:
            self.assertEqual(list(dict(model.named_parameters())), expected_names)
            self.assertGreater(float(model.net[2].weight.detach().norm()), .1)
            jacobians.append(parameter_jacobian(model, x))
        # Nonzero output projection makes the new eigenvalue tangent nontrivial.
        self.assertGreater(float(jacobians[0][:, -8:].detach().norm()), .01)
        for jac in jacobians[1:]:
            torch.testing.assert_close(jac, jacobians[0], atol=2e-15, rtol=2e-13)
            torch.testing.assert_close(jac @ jac.T, jacobians[0] @ jacobians[0].T, atol=3e-15, rtol=3e-13)

    def test_static_k2_absorbs_exactly_into_affine_projections(self):
        candidate = DiagonalODEFFN(3, 5, "k2", "static", .9).double()
        with torch.no_grad():
            candidate.ode1.raw_eigenvalues.copy_(torch.linspace(-1.2, .7, 5))
            candidate.ode2.raw_eigenvalues.copy_(torch.linspace(.8, -.9, 5))
        baseline = StandardFFN(3, 5).double()
        rate1, rate2 = candidate.ode1.eigenvalues(), candidate.ode2.eigenvalues()
        d1, d2 = 1+.6*rate1+.5*(.6*rate1).square(), 1+.6*rate2+.5*(.6*rate2).square()
        with torch.no_grad():
            baseline.net[0].weight.copy_(d1[:, None] * candidate.net[0].weight)
            baseline.net[0].bias.copy_(d1 * candidate.net[0].bias)
            baseline.net[2].weight.copy_(candidate.net[2].weight * d2[None, :])
            baseline.net[2].bias.copy_(candidate.net[2].bias)
        self.assertGreater(float(candidate.net[0].bias.detach().abs().sum()), .01)
        self.assertGreater(float(candidate.net[2].bias.detach().abs().sum()), .01)
        x = torch.randn(5, 3, dtype=torch.float64)
        torch.testing.assert_close(candidate(x), baseline(x), atol=3e-15, rtol=3e-13)
        torch.testing.assert_close(torch.autograd.functional.jacobian(candidate, x),
                                   torch.autograd.functional.jacobian(baseline, x), atol=3e-15, rtol=3e-13)

    def test_real_parameter_and_dense_mac_counts(self):
        width, hidden = 64, 128
        candidate = DiagonalODEFFN(width, hidden)
        baseline = StandardFFN(width, hidden)
        self.assertEqual(sum(p.numel() for p in candidate.parameters()), 2*width*hidden + 3*hidden + width)
        self.assertEqual(sum(p.numel() for p in candidate.parameters()) - sum(p.numel() for p in baseline.parameters()), 2*hidden)
        self.assertEqual(len([m for m in candidate.modules() if isinstance(m, nn.Linear)]), 2)
        self.assertEqual(list(dict(candidate.ode1.named_parameters())), ["raw_eigenvalues"])
        count, calls = 0, 0
        linear = F.linear
        def measured(x, weight, bias=None):
            nonlocal count, calls
            count += (x.numel() // x.shape[-1]) * weight.numel()
            calls += 1
            return linear(x, weight, bias)
        with patch.object(F, "linear", measured), torch.no_grad():
            candidate(torch.zeros(2, 3, width))
        self.assertEqual(calls, 2)
        self.assertEqual(count, 6*2*width*hidden)
        class Envelope(nn.Module):
            def __init__(self, ffn):
                super().__init__()
                self.ffn = ffn
            def forward(self, x, t):
                return self.ffn(x)
            def ode_layers(self):
                return []
        self.assertEqual(macs(Envelope(candidate), width), 2*width*hidden)

    def test_stats_are_observational_and_state_round_trip_is_exact(self):
        layer = DiagonalFlow(3, "exp", beta=.7).double()
        with torch.no_grad():
            layer.raw_eigenvalues.copy_(torch.tensor([-.5, .2, .8]))
        x = torch.tensor([[-1., 0., 2.], [.1, -.3, 1.]], dtype=torch.float64, requires_grad=True)
        state, rng = copy.deepcopy(layer.state_dict()), torch.get_rng_state().clone()
        stats = layer.activation_stats(x)
        self.assertEqual(stats["scope"], "observed_coordinatewise_map_only")
        self.assertAlmostEqual(stats["derivative_min"], float(layer.derivative(x).detach().min()), places=14)
        self.assertAlmostEqual(stats["duration_mean"], float(layer.duration(x).detach().mean()), places=14)
        self.assertTrue(all(math.isfinite(v) for v in stats.values() if isinstance(v, (float, int))))
        self.assertIsNone(x.grad)
        self.assertIsNone(layer.raw_eigenvalues.grad)
        torch.testing.assert_close(layer.state_dict()["raw_eigenvalues"], state["raw_eigenvalues"], atol=0, rtol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
        restored = DiagonalFlow(3, "exp", beta=.7).double()
        restored.load_state_dict(state)
        torch.testing.assert_close(restored(x), layer(x), atol=0, rtol=0)

    def test_invalid_settings_and_shapes(self):
        for features in (0, -1, 1.2, True):
            with self.subTest(features=features), self.assertRaises(ValueError):
                DiagonalFlow(features)
        for options in (dict(solver="k3"), dict(clock="learned"), dict(beta=0), dict(beta=-1),
                        dict(beta=1.01), dict(beta=math.nan), dict(beta=math.inf), dict(beta=True)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                DiagonalFlow(3, **options)
            with self.subTest(ffn_options=options), self.assertRaises(ValueError):
                DiagonalODEFFN(3, 4, **options)
        for width, hidden in ((0, 4), (3, 0), (True, 4), (3, 1.2)):
            with self.assertRaises(ValueError):
                DiagonalODEFFN(width, hidden)
        layer = DiagonalFlow(3)
        for x in (torch.ones(()), torch.ones(2, 4), torch.ones(2, 3, dtype=torch.int64)):
            with self.assertRaises(ValueError):
                layer(x)
        with self.assertRaises(ValueError):
            layer.activation_stats(torch.empty(0, 3))


if __name__ == "__main__":
    unittest.main()
