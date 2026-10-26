"""Meaningful derivative and parameter-reuse checks for the local prototype."""
from __future__ import annotations

import copy
import unittest

import torch
from torch.nn import functional as F

from synthetic.local_adjoint import LocalAdjointODEFFN, TiedSiLUFFN, build
from synthetic.models import OriginalODELayer


class LocalAdjointTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def test_exact_vjp_and_mixed_parameter_derivatives(self) -> None:
        layer = OriginalODELayer(4, 7, orders=2).double()
        model = LocalAdjointODEFFN(layer).double()
        reference = copy.deepcopy(layer)
        x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        xx = x.detach().clone().requires_grad_(True)
        actual = model(x)
        encoded = reference(xx)
        expected = torch.autograd.grad(encoded, xx, F.silu(encoded), create_graph=True)[0]
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
        target = torch.randn_like(actual)
        ga = torch.autograd.grad((actual-target).square().mean(), (x, *model.parameters()))
        gb = torch.autograd.grad((expected-target).square().mean(), (xx, *reference.parameters()))
        for a, b in zip(ga, gb):
            torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-11)
        self.assertEqual(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in layer.parameters()))

    def test_local_symmetry_and_omitted_clock_difference(self) -> None:
        layer = OriginalODELayer(4, 6, orders=2).double()
        full = LocalAdjointODEFFN(layer).double()
        omitted = LocalAdjointODEFFN(layer, include_clock=False).double()
        x = torch.randn(4, dtype=torch.float64, requires_grad=True)
        jac = torch.autograd.functional.jacobian(full, x)
        torch.testing.assert_close(jac, jac.T, rtol=1e-10, atol=1e-11)
        duration = full.ode1.step_size(x)
        clock_grad = torch.autograd.grad(duration, x, create_graph=True)[0]
        p = full.ode1.proj.weight
        ph = F.linear(x, p)
        response = ph + duration*F.linear(ph, p)
        q = F.silu(full.ode1(x))
        expected = clock_grad * torch.dot(response, q)
        torch.testing.assert_close(full(x)-omitted(x), expected, rtol=1e-10, atol=1e-11)
        self.assertGreater(float(expected.detach().norm()), 1e-8)

    def test_tied_ridge_gradient(self) -> None:
        model = TiedSiLUFFN(4, 9).double()
        x = torch.randn(4, dtype=torch.float64, requires_grad=True)
        jac = torch.autograd.functional.jacobian(model, x)
        torch.testing.assert_close(jac, jac.T, rtol=1e-11, atol=1e-12)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 4*9+9)

    def test_whole_model_derivative_and_reference_pairing(self) -> None:
        original = build(3, "original").double()
        model = build(3, "local_full").double()
        expected = original.state_dict()
        for name, value in model.state_dict().items():
            if ".ffn." not in name:
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        for a, b in zip(original.raw.blocks, model.raw.blocks):
            for key, value in b.ffn.ode1.state_dict().items():
                torch.testing.assert_close(value, a.ffn.ode1.state_dict()[key], rtol=0, atol=0)
        # A nonzero head exposes the new FFN's input/parameter responses.
        with torch.no_grad():
            model.raw.output.weight.copy_(torch.linspace(-.03,.04,model.raw.output.weight.numel()).reshape_as(model.raw.output.weight))
        x = torch.randn(2,3,dtype=torch.float64)
        t = torch.full((2,1),.4,dtype=torch.float64)
        p = model.raw.blocks[0].ffn.ode1.proj.weight
        direction = torch.linspace(-1,1,p.numel(),dtype=p.dtype).reshape_as(p)
        direction /= direction.norm()
        gradient = torch.autograd.grad(model(x,t).square().mean(),p)[0]
        analytic = float((gradient*direction).sum())
        saved = p.detach().clone(); eps=1e-5
        with torch.no_grad():
            p.copy_(saved+eps*direction); plus=float(model(x,t).square().mean())
            p.copy_(saved-eps*direction); minus=float(model(x,t).square().mean())
            p.copy_(saved)
        self.assertLess(abs((plus-minus)/(2*eps)-analytic),1e-8)


if __name__ == "__main__":
    unittest.main()
