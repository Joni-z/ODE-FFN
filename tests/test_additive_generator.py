import copy
import unittest

import torch

from scripts.run_architecture_study import macs
from synthetic.models import OriginalODELayer,VectorODELayer,VelocityNet


class AdditiveGeneratorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0);torch.set_num_threads(1)

    def test_complete_jacobian_and_P_gradient(self):
        for cls in (OriginalODELayer,VectorODELayer):
            layer=cls(4,3,orders=1,operator_mode='additive').double()
            x=torch.randn(4,dtype=torch.float64)
            duration=layer.step_size(x[None])[0].expand(4)
            js=torch.autograd.functional.jacobian(lambda z:layer.step_size(z[None])[0].expand(4),x)
            center=layer.shift+layer.scale/2
            expected=torch.eye(4,dtype=torch.float64)+center*layer.proj.weight+torch.diag(duration-center)+torch.diag(x)@js
            actual=torch.autograd.functional.jacobian(lambda z:layer(z[None])[0],x)
            torch.testing.assert_close(actual,expected,atol=2e-15,rtol=2e-13)
            q=torch.randn_like(x)
            gradient=torch.autograd.grad((layer(x[None])[0]*q).sum(),layer.proj.weight)[0]
            torch.testing.assert_close(gradient,center*torch.outer(q,x),atol=2e-15,rtol=2e-13)
            stats=layer.activation_stats(x[None])
            increment=layer(x[None])-x
            ratio=float((increment.square().mean().sqrt()/x.square().mean().sqrt()).detach())
            self.assertAlmostEqual(stats['term1_rms_over_input'],ratio,places=13)

    def test_common_initial_functions_gradients_rng_and_budget(self):
        for kind,hidden in (('ode_original',16),('ode_vector',8)):
            options=dict(width=16,depth=2,orders=1,gate_hidden=hidden,gate_reference_hidden=16,ode_solve_init='constant_paired')
            torch.manual_seed(0);parent=VelocityNet(3,kind,**options);rng=torch.get_rng_state()
            torch.manual_seed(0);candidate=VelocityNet(3,kind,**options,ode_operator_mode='additive')
            torch.testing.assert_close(torch.get_rng_state(),rng,atol=0,rtol=0)
            for name,value in parent.state_dict().items():torch.testing.assert_close(value,candidate.state_dict()[name],atol=0,rtol=0)
            self.assertEqual(macs(parent,3),macs(candidate,3));self.assertEqual(parent.parameter_count(),candidate.parameter_count())
            with torch.no_grad():
                head=torch.randn_like(parent.output.weight)*.2
                parent.output.weight.copy_(head);candidate.output.weight.copy_(head)
            x,t=torch.randn(53,3),torch.rand(53,1)
            torch.testing.assert_close(parent(x,t),candidate(x,t),atol=0,rtol=0)
            parent(x,t).square().mean().backward();candidate(x,t).square().mean().backward()
            other=dict(candidate.named_parameters())
            for name,value in parent.named_parameters():
                if '.t_func.2.' not in name:
                    torch.testing.assert_close(value.grad,other[name].grad,atol=2e-7,rtol=2e-6)

    def test_actual_finite_controller_has_a_direction_switching_witness(self):
        layer=OriginalODELayer(1,2,orders=1,operator_mode='additive').double()
        with torch.no_grad():
            layer.proj.weight.zero_()
            layer.t_func[0].weight.copy_(torch.tensor([[1.],[-1.]],dtype=torch.float64));layer.t_func[0].bias.zero_()
            layer.t_func[-1].weight.fill_(-2*layer.tau*.25);layer.t_func[-1].bias.fill_(2*layer.tau*.25)
        x=torch.tensor([[.25],[1.],[2.],[5.]],dtype=torch.float64)
        coefficient=.4*torch.tanh(.25*(1-x*torch.tanh(x/2)))
        torch.testing.assert_close(layer(x),x*(1+coefficient),atol=2e-15,rtol=2e-14)
        self.assertGreater(float(coefficient[0]),0);self.assertLess(float(coefficient[-1]),0)
        grid=torch.linspace(-20,20,2001,dtype=torch.float64)[:,None].requires_grad_(True)
        derivative=torch.autograd.grad(layer(grid).sum(),grid)[0]
        self.assertGreaterEqual(float(derivative.min()),.2)

    def test_controller_response_survives_zero_P(self):
        parent=OriginalODELayer(2,3,orders=1).double()
        with torch.no_grad():
            parent.proj.weight.zero_();parent.t_func[-1].weight.zero_();parent.t_func[-1].bias.zero_()
        candidate=copy.deepcopy(parent);candidate.operator_mode='additive'
        x=torch.tensor([[1.,2.]],dtype=torch.float64)
        torch.testing.assert_close(parent(x),candidate(x),atol=0,rtol=0)
        parent(x).sum().backward();candidate(x).sum().backward()
        self.assertEqual(torch.count_nonzero(parent.t_func[-1].bias.grad),0)
        self.assertGreater(float(candidate.t_func[-1].bias.grad.norm()),0)

    def test_unsupported_combinations_are_rejected(self):
        for options in (dict(orders=2),dict(orders=1,ode_solve_return='increment'),
                        dict(orders=1,ode_rate_placement='before_projection'),dict(orders=1,ode_gate_reparam=8.)):
            with self.assertRaises(ValueError):VelocityNet(2,'ode_vector',ode_operator_mode='additive',**options)


if __name__=='__main__':
    unittest.main()
