import unittest

import torch
from torch.nn import functional as F

from scripts.run_architecture_study import macs
from synthetic.models import OriginalODEFFN, VectorODEFFN, VelocityNet


PATTERNS=('map_map','map_increment','increment_map','increment_increment')


class IdentityLocationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0);torch.set_num_threads(1)

    def test_complete_two_solve_chain_jacobian(self):
        for cls in (OriginalODEFFN,VectorODEFFN):
            for pattern in PATTERNS:
                modes=pattern.split('_')
                block=cls(4,3,orders=1,solve_returns=modes).double()
                x=torch.randn(4,dtype=torch.float64)
                first=block.ode1(x[None])[0]
                middle=F.silu(first)
                factors=[]
                for layer,h,mode in ((block.ode1,x,modes[0]),(block.ode2,middle,modes[1])):
                    duration=layer.step_size(h[None])[0].expand(4)
                    js=torch.autograd.functional.jacobian(lambda z:layer.step_size(z[None])[0].expand(4),h)
                    p=layer.proj.weight
                    factors.append((int(mode=='map')*torch.eye(4,dtype=torch.float64)+
                                    torch.diag(duration)@p+torch.diag(p@h)@js))
                probability=torch.sigmoid(first)
                derivative=probability*(1+first*(1-probability))
                expected=factors[1]@torch.diag(derivative)@factors[0]
                actual=torch.autograd.functional.jacobian(lambda z:block(z[None])[0],x)
                torch.testing.assert_close(actual,expected,atol=2e-15,rtol=2e-13)

    def test_zero_P_gradient_bootstrap_depends_on_identity_location(self):
        expected={'map_map':(True,True),'map_increment':(False,True),
                  'increment_map':(True,False),'increment_increment':(False,False)}
        for pattern in PATTERNS:
            block=OriginalODEFFN(4,3,orders=1,solve_returns=pattern.split('_')).double()
            with torch.no_grad():block.ode1.proj.weight.zero_();block.ode2.proj.weight.zero_()
            x=torch.randn(11,4,dtype=torch.float64)
            gradients=torch.autograd.grad(block(x).sum(),(block.ode1.proj.weight,block.ode2.proj.weight))
            self.assertEqual(tuple(bool(g.norm()>0) for g in gradients),expected[pattern])

    def test_all_eight_patterns_share_initial_hidden_function_and_RNG(self):
        previous=torch.get_default_dtype();torch.set_default_dtype(torch.float64)
        try:
            models=[];rng=[]
            for kind,hidden in (('ode_original',16),('ode_vector',8)):
                for pattern in PATTERNS:
                    torch.manual_seed(0)
                    model=VelocityNet(3,kind,width=16,depth=2,orders=1,gate_hidden=hidden,
                        gate_reference_hidden=16,ode_solve_init='constant_paired',ode_solve_pattern=pattern)
                    models.append(model);rng.append(torch.get_rng_state())
            head=torch.randn_like(models[0].output.weight)*.2
            x,t=torch.randn(47,3),torch.rand(47,1)
            for model,state in zip(models,rng):
                with torch.no_grad():model.output.weight.copy_(head)
                torch.testing.assert_close(state,rng[0],atol=0,rtol=0)
                torch.testing.assert_close(model(x,t),models[0](x,t),atol=2e-13,rtol=2e-13)
            for group in (models[:4],models[4:]):
                self.assertEqual(len({m.parameter_count() for m in group}),1)
                self.assertEqual(len({macs(m,3) for m in group}),1)
        finally:
            torch.set_default_dtype(previous)

    def test_conflicting_uniform_override_is_rejected(self):
        with self.assertRaises(ValueError):
            VelocityNet(2,'ode_vector',orders=1,ode_solve_pattern='map_increment',ode_solve_return='increment')


if __name__=='__main__':
    unittest.main()
