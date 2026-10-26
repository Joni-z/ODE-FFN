import copy
import unittest

import torch

from scripts.run_architecture_study import macs
from synthetic.models import OriginalODELayer, VelocityNet


class ClockCoordinateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def test_initial_function_rng_budget_and_parameter_response(self):
        options=dict(orders=1,gate_hidden=32,gate_reference_hidden=64)
        reference=VelocityNet(16,'ode_vector',**options)
        rng=torch.get_rng_state()
        torch.manual_seed(0)
        reparam=VelocityNet(16,'ode_vector',**options,ode_gate_reparam=8.)
        torch.testing.assert_close(torch.get_rng_state(),rng,atol=0,rtol=0)
        self.assertEqual(reference.parameter_count(),reparam.parameter_count())
        self.assertEqual(macs(reference,16),macs(reparam,16))
        for name,value in reference.state_dict().items():
            paired=reparam.state_dict()[name]
            torch.testing.assert_close(value,paired*8 if '.t_func.2.' in name else paired,atol=0,rtol=0)
        x,t=torch.randn(19,16),torch.rand(19,1)
        head=torch.randn_like(reference.output.weight)
        with torch.no_grad():
            reference.output.weight.copy_(head)
            reparam.output.weight.copy_(head)
        old,new=reference(x,t),reparam(x,t)
        torch.testing.assert_close(old,new,atol=0,rtol=0)
        target=torch.randn_like(old)
        (old-target).square().mean().backward()
        (new-target).square().mean().backward()
        for name,parameter in reference.named_parameters():
            actual=dict(reparam.named_parameters())[name].grad
            expected=parameter.grad*(8 if '.t_func.2.' in name else 1)
            torch.testing.assert_close(actual,expected,atol=2e-7,rtol=2e-6)

    def test_adam_coordinate_equivalence_requires_lr_epsilon_and_clip_metric(self):
        old=OriginalODELayer(2,3,orders=1).double()
        new=copy.deepcopy(old)
        rho=1/8
        new.tau*=rho
        with torch.no_grad():
            new.t_func[-1].weight.mul_(rho)
            new.t_func[-1].bias.mul_(rho)
        old_optimizer=torch.optim.Adam(old.parameters(),lr=.003,eps=1e-8)
        last=[];other=[]
        for name,parameter in new.named_parameters():
            (last if name.startswith('t_func.2.') else other).append(parameter)
        new_optimizer=torch.optim.Adam([dict(params=last,lr=rho*.003,eps=1e-8/rho),
                                       dict(params=other,lr=.003,eps=1e-8)])
        x=torch.randn(13,2,dtype=torch.float64)
        target=torch.randn_like(x)
        clipped=0
        for _ in range(20):
            old_optimizer.zero_grad();new_optimizer.zero_grad()
            torch.testing.assert_close(old(x),new(x),atol=2e-13,rtol=2e-13)
            (old(x)-target).square().mean().backward()
            (new(x)-target).square().mean().backward()
            norm=torch.nn.utils.clip_grad_norm_(old.parameters(),.03)
            pulled_back_norm=torch.stack([(p.grad*(rho if name.startswith('t_func.2.') else 1)).square().sum()
                for name,p in new.named_parameters()]).sum().sqrt()
            torch.testing.assert_close(norm,pulled_back_norm,atol=2e-13,rtol=2e-13)
            coefficient=(.03/(pulled_back_norm+1e-6)).clamp(max=1.)
            for parameter in new.parameters():parameter.grad.mul_(coefficient)
            clipped+=int(norm>.03)
            old_optimizer.step();new_optimizer.step()
            for name,parameter in old.named_parameters():
                paired=dict(new.named_parameters())[name]
                torch.testing.assert_close(parameter,paired/rho if name.startswith('t_func.2.') else paired,
                                           atol=3e-13,rtol=3e-13)
        self.assertGreater(clipped,0)


if __name__=='__main__':
    unittest.main()
