import copy
import unittest

import torch

from scripts.run_architecture_study import macs
from synthetic.models import OriginalODELayer, VectorODELayer, VelocityNet


class RateOrderTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def test_scalar_commutation_includes_full_gate_derivatives(self):
        after=OriginalODELayer(5,7,orders=1).double()
        before=copy.deepcopy(after)
        before.rate_placement='before_projection'
        x=torch.randn(11,5,dtype=torch.float64)
        target=torch.randn_like(x)
        torch.testing.assert_close(after(x),before(x),atol=3e-16,rtol=3e-15)
        (after(x)-target).square().mean().backward()
        (before(x)-target).square().mean().backward()
        for a,b in zip(after.parameters(),before.parameters()):
            torch.testing.assert_close(a.grad,b.grad,atol=3e-16,rtol=3e-14)

    def test_frozen_generators_are_similar_but_vector_actions_differ(self):
        after=VectorODELayer(5,4,reference_gate_hidden=5,orders=1).double()
        before=copy.deepcopy(after);before.rate_placement='before_projection'
        with torch.no_grad():
            after.t_func[-1].weight.mul_(20)
            before.load_state_dict(after.state_dict())
        x=torch.randn(9,5,dtype=torch.float64)
        duration=after.step_size(x)
        d=torch.diag_embed(duration)
        p=after.proj.weight
        left=d@p;right=p@d
        torch.testing.assert_close(right,torch.linalg.solve(d,left@d),atol=2e-16,rtol=2e-14)
        for order in (1,2,3):
            torch.testing.assert_close(torch.linalg.matrix_power(right,order),
                torch.linalg.solve(d,torch.linalg.matrix_power(left,order)@d),atol=2e-16,rtol=2e-14)
        expected=x+torch.einsum('bij,bj->bi',right,x)
        torch.testing.assert_close(before(x),expected,atol=2e-16,rtol=2e-14)
        self.assertGreater(float((before(x)-after(x)).detach().abs().max()),1e-3)

    def test_before_projection_complete_jacobian_and_weight_gradient(self):
        layer=VectorODELayer(4,3,reference_gate_hidden=4,orders=1,
                             rate_placement='before_projection').double()
        x=torch.randn(4,dtype=torch.float64,requires_grad=True)
        gate=lambda z:layer.step_size(z[None])[0]
        duration=gate(x)
        js=torch.autograd.functional.jacobian(gate,x)
        p=layer.proj.weight
        expected=torch.eye(4,dtype=torch.float64)+p@torch.diag(duration)+p@torch.diag(x)@js
        actual=torch.autograd.functional.jacobian(lambda z:layer(z[None])[0],x)
        torch.testing.assert_close(actual,expected,atol=2e-15,rtol=2e-13)
        q=torch.randn_like(x)
        gradient=torch.autograd.grad((layer(x[None])[0]*q).sum(),layer.proj.weight)[0]
        torch.testing.assert_close(gradient,torch.outer(q,duration*x),atol=2e-15,rtol=2e-13)
        stats=layer.activation_stats(x[None])
        ratio=float(((layer(x[None])-x).square().mean().sqrt()/x.square().mean().sqrt()).detach())
        self.assertAlmostEqual(stats['term1_rms_over_input'],ratio,places=13)

    def test_network_state_rng_budget_pairing_and_configuration_guards(self):
        options=dict(width=16,depth=2,orders=1,gate_hidden=8,gate_reference_hidden=16)
        after=VelocityNet(3,'ode_vector',**options)
        state=torch.get_rng_state()
        torch.manual_seed(0)
        before=VelocityNet(3,'ode_vector',**options,ode_rate_placement='before_projection')
        torch.testing.assert_close(state,torch.get_rng_state(),atol=0,rtol=0)
        self.assertEqual(macs(after,3),macs(before,3))
        self.assertEqual(after.parameter_count(),before.parameter_count())
        for name,value in after.state_dict().items():
            torch.testing.assert_close(value,before.state_dict()[name],atol=0,rtol=0)
        self.assertTrue(all(layer.rate_placement=='before_projection' for _,layer in before.ode_layers()))
        with self.assertRaises(ValueError):
            VelocityNet(3,'ode_vector',**options,ode_rate_placement='before_projection',ode_gate_reparam=8.)
        with self.assertRaises(ValueError):
            VectorODELayer(4,3,orders=2,rate_placement='before_projection')


if __name__=='__main__':
    unittest.main()
