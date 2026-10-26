"""Mathematical controls only; no production architecture option is added."""
import copy
import unittest

import torch
from torch.nn import functional as F

from synthetic.models import OriginalODELayer, VectorODELayer


def solve(layer, x, gamma):
    return gamma*x + layer.step_size(x)*F.linear(x,layer.proj.weight)


class ZeroOrderAlgebraTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def test_constant_diagonal_rates_have_exact_map_increment_correspondence(self):
        first=VectorODELayer(5,3,reference_gate_hidden=5,orders=1).double()
        second=copy.deepcopy(first)
        duration=torch.linspace(.2,.9,5,dtype=torch.float64)
        for layer in (first,second):
            layer.step_size=lambda x,d=duration:d.expand_as(x)
        with torch.no_grad():
            second.proj.weight.add_(torch.diag(1/duration))
        x=torch.randn(17,5,dtype=torch.float64)
        torch.testing.assert_close(solve(first,x,1),solve(second,x,0),atol=1e-15,rtol=1e-14)

    def test_state_dependent_complete_jacobian(self):
        for cls in (OriginalODELayer,VectorODELayer):
            layer=cls(4,3,orders=1).double()
            x=torch.randn(4,dtype=torch.float64)
            duration=layer.step_size(x[None])[0].expand(4)
            js=torch.autograd.functional.jacobian(lambda h:layer.step_size(h[None])[0].expand(4),x)
            for gamma in (0,1):
                jac=torch.autograd.functional.jacobian(lambda h:solve(layer,h[None],gamma)[0],x)
                p=layer.proj.weight
                expected=gamma*torch.eye(4,dtype=torch.float64)+torch.diag(duration)@p+torch.diag(p@x)@js
                torch.testing.assert_close(jac,expected,atol=2e-15,rtol=2e-13)

    def test_both_increment_solves_have_joint_zero_P_degeneracy(self):
        first=VectorODELayer(4,3,orders=1).double()
        second=VectorODELayer(4,3,orders=1).double()
        with torch.no_grad():
            first.proj.weight.zero_();second.proj.weight.zero_()
        x=torch.randn(9,4,dtype=torch.float64)
        for gamma in (0,1):
            output=solve(second,F.silu(solve(first,x,gamma)),gamma)
            torch.testing.assert_close(output,F.silu(x) if gamma else torch.zeros_like(x),atol=0,rtol=0)
            gradients=torch.autograd.grad(output.sum(),(first.proj.weight,second.proj.weight))
            if gamma:
                self.assertTrue(all(g.norm()>0 for g in gradients))
            else:
                self.assertTrue(all(torch.count_nonzero(g)==0 for g in gradients))

    def test_constant_rate_shift_preserves_clipped_Adam_updates(self):
        first=OriginalODELayer(4,3,orders=1).double()
        second=copy.deepcopy(first)
        for layer in (first,second):
            layer.step_size=lambda x:x.new_full((len(x),1),.6)
        shift=torch.eye(4,dtype=torch.float64)/.6
        with torch.no_grad():second.proj.weight.add_(shift)
        a=torch.optim.Adam([first.proj.weight],lr=.003)
        b=torch.optim.Adam([second.proj.weight],lr=.003)
        x,target=torch.randn(13,4,dtype=torch.float64),torch.randn(13,4,dtype=torch.float64)
        clipped=0
        for _ in range(8):
            for layer,gamma,optimizer in ((first,1,a),(second,0,b)):
                optimizer.zero_grad();(solve(layer,x,gamma)-target).square().mean().backward()
                norm=torch.nn.utils.clip_grad_norm_([layer.proj.weight],.1)
                clipped+=int(norm>.1)
                optimizer.step()
            torch.testing.assert_close(second.proj.weight,first.proj.weight+shift,atol=1e-15,rtol=1e-14)
        self.assertEqual(clipped,16)


if __name__=='__main__':
    unittest.main()
