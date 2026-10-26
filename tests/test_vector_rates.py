"""Operator semantics, derivatives and paired initialization of vector rates."""
import unittest
import torch
from synthetic.models import OriginalODELayer,VectorODELayer,VelocityNet


class VectorRateTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def test_frozen_generator_power_and_noncommuting_difference(self) -> None:
        layer=VectorODELayer(5,3,reference_gate_hidden=5).double()
        x=torch.randn(7,5,dtype=torch.float64)
        with torch.no_grad():layer.t_func[-1].weight.mul_(20)
        s,p=layer.step_size(x),layer.proj.weight
        a=s.unsqueeze(-1)*p.unsqueeze(0)
        expected=x+torch.einsum('bij,bj->bi',a,x)+.5*torch.einsum('bij,bjk,bk->bi',a,a,x)
        torch.testing.assert_close(layer(x),expected,atol=1e-14,rtol=1e-12)
        wrong=x+s*(x@p.T)+.5*s.square()*(x@p.T@p.T)
        self.assertGreater(float((expected-wrong).abs().max().detach()),1e-5)

    def test_uniform_rates_recover_scalar_and_derivative_formula(self) -> None:
        layer=VectorODELayer(4,3,reference_gate_hidden=4).double()
        x=torch.randn(4,dtype=torch.float64,requires_grad=True)
        def gate(z):return layer.step_size(z[None])[0]
        s=gate(x)
        js=torch.autograd.functional.jacobian(gate,x)
        p=layer.proj.weight
        q=p@x
        d=torch.diag(s)
        a=d@p
        u=torch.diag(q)+.5*torch.diag(p@d@q)+.5*d@p@torch.diag(q)
        expected=torch.eye(4,dtype=torch.float64)+a+.5*a@a+u@js
        actual=torch.autograd.functional.jacobian(lambda z:layer(z[None])[0],x)
        torch.testing.assert_close(actual,expected,atol=1e-13,rtol=1e-12)
        scalar=OriginalODELayer(4,4).double()
        with torch.no_grad():scalar.proj.weight.copy_(p)
        scalar.step_size=lambda z,context=None:z.new_full((len(z),1),.7)
        layer.step_size=lambda z,context=None:z.new_full(z.shape,.7)
        xx=torch.randn(6,4,dtype=torch.float64)
        torch.testing.assert_close(layer(xx),scalar(xx),atol=0,rtol=0)

    def test_non_gate_parameters_and_rng_match_scalar_reference(self) -> None:
        models=[]
        states=[]
        for kind,hidden in [('ode_original',8),('ode_vector',4)]:
            torch.manual_seed(0)
            models.append(VelocityNet(2,kind,width=8,depth=2,gate_hidden=hidden,gate_reference_hidden=8))
            states.append(torch.get_rng_state())
        a,b=models
        torch.testing.assert_close(states[0],states[1],atol=0,rtol=0)
        for key,value in a.state_dict().items():
            if '.t_func.' not in key:
                torch.testing.assert_close(value,b.state_dict()[key],atol=0,rtol=0)
        for _,layer in b.ode_layers():self.assertEqual(layer.t_func[-1].out_features,8)


if __name__=='__main__':unittest.main()
