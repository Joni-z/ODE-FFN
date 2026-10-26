"""Composition semantics and causal controls for the middle-activation study."""
import unittest
import torch
from synthetic.models import OriginalODEFFN,VectorODEFFN,VelocityNet


class CompositionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def test_zero_generator_limit_and_remaining_gate_nonlinearity(self) -> None:
        layer=VectorODEFFN(4,3,reference_gate_hidden=4,orders=1,
                           middle_activation='identity').double()
        x=torch.randn(7,4,dtype=torch.float64)
        with torch.no_grad():
            layer.ode1.proj.weight.zero_()
            layer.ode2.proj.weight.zero_()
        torch.testing.assert_close(layer(x),x,rtol=0,atol=0)
        with torch.no_grad():
            layer.ode1.proj.weight.copy_(torch.eye(4,dtype=torch.float64))
            layer.ode1.t_func[-1].weight.mul_(20)
        # Removing the middle SiLU does not turn an input-gated layer linear.
        self.assertGreater(float((layer(2*x)-2*layer(x)).abs().max().detach()),1e-4)

    def test_gate_only_composition_has_exact_chain_jacobian(self) -> None:
        layer=OriginalODEFFN(4,4,orders=1,middle_activation='identity').double()
        x=torch.randn(4,dtype=torch.float64,requires_grad=True)
        first=lambda z:layer.ode1(z[None])[0]
        second=lambda z:layer.ode2(z[None])[0]
        j1=torch.autograd.functional.jacobian(first,x)
        j2=torch.autograd.functional.jacobian(second,first(x))
        actual=torch.autograd.functional.jacobian(lambda z:layer(z[None])[0],x)
        torch.testing.assert_close(actual,j2@j1,atol=1e-13,rtol=1e-12)

    def test_initial_parameters_and_rng_are_paired_for_activation_factor(self) -> None:
        for kind,hidden in [('ode_original',8),('ode_vector',4)]:
            models=[];states=[]
            for activation in ('silu','identity'):
                torch.manual_seed(0)
                models.append(VelocityNet(2,kind,width=8,depth=2,gate_hidden=hidden,
                    orders=1,gate_reference_hidden=8,ode_middle_activation=activation))
                states.append(torch.get_rng_state())
            self.assertEqual(models[0].parameter_count(),models[1].parameter_count())
            torch.testing.assert_close(states[0],states[1],atol=0,rtol=0)
            for key,value in models[0].state_dict().items():
                torch.testing.assert_close(value,models[1].state_dict()[key],atol=0,rtol=0)


if __name__=='__main__':unittest.main()
