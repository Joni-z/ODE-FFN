"""Matched local clock geometry and unchanged common parameter initialization."""
import unittest
import torch
from synthetic.models import OriginalODELayer,VectorODELayer,VelocityNet
from scripts.run_architecture_study import macs


class ClockRangeTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1);torch.manual_seed(0)

    def test_center_slope_second_derivative_and_cubic_matching(self) -> None:
        settings=[{},dict(shift=0.,scale=1.2,tau=15.),dict(gate_transform='affine')]
        layers=[OriginalODELayer(4,4,orders=1,**s).double() for s in settings]
        for layer in layers:
            zero=torch.zeros((),dtype=torch.float64,requires_grad=True)
            value=layer.duration_from_logit(zero)
            derivative=torch.autograd.functional.jacobian(layer.duration_from_logit,zero)
            curvature=torch.autograd.functional.hessian(layer.duration_from_logit,zero)
            torch.testing.assert_close(value,torch.tensor(.6,dtype=torch.float64),atol=1e-14,rtol=0)
            torch.testing.assert_close(derivative,torch.tensor(.02,dtype=torch.float64),atol=1e-14,rtol=0)
            torch.testing.assert_close(curvature,torch.zeros_like(curvature),atol=1e-14,rtol=0)
        differences=[]
        for g in [.1,.05]:
            z=torch.tensor(g,dtype=torch.float64)
            differences.append(float((layers[0].duration_from_logit(z)-layers[1].duration_from_logit(z)).abs()))
        self.assertGreater(differences[0]/differences[1],7.9)
        self.assertLess(differences[0]/differences[1],8.1)

    def test_affine_allows_signed_rates_and_preserves_frozen_generator(self) -> None:
        layer=VectorODELayer(4,3,reference_gate_hidden=4,orders=1,gate_transform='affine').double()
        values=layer.duration_from_logit(torch.tensor([-100.,0.,100.],dtype=torch.float64))
        self.assertLess(float(values[0]),0.)
        self.assertGreater(float(values[-1]),1.)
        x=torch.randn(7,4,dtype=torch.float64)
        with torch.no_grad():layer.t_func[-1].bias.fill_(-100.)
        s=layer.step_size(x)
        torch.testing.assert_close(layer(x),x+s*layer.proj(x),atol=0,rtol=0)
        layer(x).square().mean().backward()
        self.assertTrue(torch.isfinite(layer.proj.weight.grad).all())
        self.assertGreater(float(layer.t_func[-1].bias.grad.abs().sum()),0.)

    def test_initial_tensors_rng_and_macs_match_all_range_arms(self) -> None:
        models=[];rng=[]
        for options in [{},dict(shift=0.,scale=1.2,tau=15.),dict(ode_gate_transform='affine')]:
            torch.manual_seed(0)
            models.append(VelocityNet(2,'ode_vector',gate_hidden=32,orders=1,**options))
            rng.append(torch.get_rng_state())
        for model,state in zip(models[1:],rng[1:]):
            torch.testing.assert_close(state,rng[0],atol=0,rtol=0)
            for name,value in models[0].state_dict().items():
                torch.testing.assert_close(value,model.state_dict()[name],atol=0,rtol=0)
            self.assertEqual(macs(model,2),macs(models[0],2))
        stats=models[-1].blocks[0].ffn.ode1.activation_stats(torch.randn(16,64))
        self.assertEqual(stats['clock_outer_saturated_fraction'],0.)
        self.assertAlmostEqual(stats['clock_logit_sensitivity_over_initial'],1.,places=6)


if __name__=='__main__':unittest.main()
