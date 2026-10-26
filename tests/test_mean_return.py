"""Test the observable internal K2 term and actual-model counterfactuals."""
import copy
import math
import unittest

import torch
from torch.nn import functional as F

from synthetic.mean_return import MeanReturnNet, second_order_terms
from synthetic.models import OriginalODELayer
from synthetic.recurrent_depth import build


def nonzero_head(model):
    with torch.no_grad():
        w = model.raw.output.weight
        w.copy_(torch.linspace(-.3, .3, w.numel(), dtype=w.dtype).reshape_as(w))


class MeanReturnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_local_k2_formula_with_state_dependent_gate(self):
        torch.manual_seed(0)
        layer = OriginalODELayer(5, 7).double()
        z = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
        changed = copy.deepcopy(layer)
        with torch.no_grad():
            changed.proj.weight.sub_(changed.proj.weight.mean(dim=0, keepdim=True))
        original, projected = layer(z), changed(z)
        difference = original - projected
        centered = difference - difference.mean(-1, keepdim=True)
        returning = second_order_terms(layer, z)['returning']
        torch.testing.assert_close(centered, returning, rtol=1e-12, atol=1e-12)
        a = torch.autograd.functional.jacobian(lambda v: layer(v)-changed(v), z)
        a = a - a.mean(dim=1, keepdim=True)
        b = torch.autograd.functional.jacobian(lambda v: second_order_terms(layer, v)['returning'], z)
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-12)

    def test_k1_uniform_row_gauge_and_k2_observable_return(self):
        p = torch.tensor([[.4, -.2, .1], [.3, .5, -.6], [-.4, .2, .7]], dtype=torch.float64)
        u = torch.ones(3, dtype=torch.float64)/math.sqrt(3)
        v = torch.tensor([.8, -.2, .4], dtype=torch.float64)
        c = torch.eye(3, dtype=torch.float64)-torch.outer(u, u)
        changed = p + torch.outer(u, v)
        torch.testing.assert_close(c@(changed-p), torch.zeros_like(p), rtol=0, atol=1e-15)
        expected = torch.outer(c@p@u, v)
        torch.testing.assert_close(c@(changed@changed-p@p), expected, rtol=1e-13, atol=1e-15)
        self.assertGreater(float(expected.norm()), .1)

    def test_native_and_mean_control_actual_models(self):
        x = torch.tensor([[.2, -.6], [1., .4]], dtype=torch.float64)
        t = torch.tensor([[.25], [.75]], dtype=torch.float64)
        for family in ('ode_k2', 'swiglu192', 'swiglu128'):
            parent = build(2, family, 'native').double()
            nonzero_head(parent)
            native, shifted = MeanReturnNet(parent), MeanReturnNet(parent, 'drop_uniform')
            self.assertTrue(torch.equal(native(x, t), parent(x, t)))
            torch.testing.assert_close(native(x,t), shifted(x,t), rtol=1e-12, atol=1e-12)
            a = torch.autograd.functional.jacobian(lambda v: native(v,t), x)
            b = torch.autograd.functional.jacobian(lambda v: shifted(v,t), x)
            torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-12)

    def test_complete_network_cp_equivalence_and_order_reference(self):
        parent = build(2, 'ode_k2', 'native').double()
        nonzero_head(parent)
        x = torch.tensor([[.2, -.6], [1., .4]], dtype=torch.float64)
        t = torch.tensor([[.25], [.75]], dtype=torch.float64)
        for variant in ('drop_return', 'drop_secondorder'):
            changed = copy.deepcopy(parent)
            with torch.no_grad():
                for block in changed.raw.blocks:
                    layer = block.ffn.ode2
                    if variant == 'drop_return':
                        layer.proj.weight.sub_(layer.proj.weight.mean(dim=0, keepdim=True))
                    else:
                        layer.orders = 1
            source = MeanReturnNet(parent, variant)
            torch.testing.assert_close(source(x,t), changed(x,t), rtol=1e-12, atol=1e-12)
            a = torch.autograd.functional.jacobian(lambda v: source(v,t), x)
            b = torch.autograd.functional.jacobian(lambda v: changed(v,t), x)
            torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-12)

    def test_gate_derivatives_and_trace_are_preserved(self):
        parent = build(2, 'ode_k2', 'native').double()
        nonzero_head(parent)
        x = torch.tensor([[.2, -.6]], dtype=torch.float64, requires_grad=True)
        t = torch.tensor([[.4]], dtype=torch.float64)
        for variant in ('native', 'drop_return', 'drop_secondorder'):
            model = MeanReturnNet(parent, variant)
            trace = []
            self.assertTrue(torch.equal(model(x,t), model(x,t,capture=trace)))
            self.assertEqual(len(trace), 3)
            self.assertTrue(torch.autograd.gradcheck(lambda v: model(v,t), (x,), fast_mode=True))


if __name__ == '__main__':
    unittest.main()
