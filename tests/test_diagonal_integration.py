import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from scripts.run_architecture_study import macs
from synthetic.diagnostics import collect
from synthetic.models import VelocityNet


class DiagonalIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0);torch.set_num_threads(1)

    def test_whole_velocity_initial_functions_jacobians_rng_and_projections(self):
        for dtype in (torch.float32,torch.float64):
            for dim in (2,16):
                torch.manual_seed(0)
                base=VelocityNet(dim,'ffn',width=16,depth=2,ffn_hidden=32).to(dtype)
                expected_rng=torch.get_rng_state().clone()
                with torch.no_grad():base.output.weight.normal_(std=.2)
                x,t=torch.randn(9,dim,dtype=dtype),torch.rand(9,1,dtype=dtype)
                expected=base(x,t).detach()
                for solver,clock in [('k1','adaptive'),('k2','adaptive'),('exp','adaptive'),('k2','static')]:
                    torch.manual_seed(0)
                    model=VelocityNet(dim,'ode_diagonal',width=16,depth=2,ffn_hidden=32,
                        diagonal_solver=solver,diagonal_clock=clock).to(dtype)
                    torch.testing.assert_close(torch.get_rng_state(),expected_rng,atol=0,rtol=0)
                    with torch.no_grad():model.output.weight.copy_(base.output.weight)
                    current=model.state_dict()
                    for name,value in base.state_dict().items():
                        torch.testing.assert_close(current[name],value,atol=0,rtol=0)
                    torch.testing.assert_close(model(x,t),expected,atol=0,rtol=0)
                    if dtype==torch.float64:
                        a=torch.autograd.functional.jacobian(lambda z:base(z[None],t[:1])[0],x[0])
                        b=torch.autograd.functional.jacobian(lambda z:model(z[None],t[:1])[0],x[0])
                        torch.testing.assert_close(a,b,atol=2e-13,rtol=2e-13)

    def test_real_macs_parameter_counts_and_diagonal_collection(self):
        model=VelocityNet(16,'ode_diagonal',ffn_hidden=128)
        base=VelocityNet(16,'ffn',ffn_hidden=128)
        self.assertEqual(model.parameter_count()-base.parameter_count(),3*2*128)
        self.assertEqual(macs(model,16),macs(base,16))
        executed=0;linear=F.linear
        def instrument(x,weight,bias=None):
            nonlocal executed
            executed+=(x.numel()//x.shape[-1])*weight.numel()
            return linear(x,weight,bias)
        with patch.object(F,'linear',instrument),torch.no_grad():
            model(torch.zeros(1,16),torch.zeros(1,1))
        self.assertEqual(executed,macs(model,16))
        self.assertEqual(len(list(model.ode_layers())),0)
        self.assertEqual(len(list(model.diagonal_layers())),6)
        stats=collect(model,torch.randn(8,16),torch.rand(8,1))
        self.assertEqual(len(stats),6)
        for record in stats:
            self.assertEqual(record['operator_type'],'conditional_diagonal_flow')
            self.assertEqual(record['P_sigma_max'],0.)
            self.assertEqual(len(record['P_eigenvalues']),128)
            self.assertEqual(record['activation']['negative_derivative_fraction'],0.)

    def test_legacy_interventions_and_unused_diagonal_options_rejected(self):
        for change in (dict(ode_solve_init='constant_paired'),dict(ode_feature_mode='shared_projection'),
                dict(ode_operator_mode='additive'),dict(ode_middle_activation='identity'),
                dict(ode_gate_reparam=8.),dict(diagonal_beta=2.),dict(diagonal_clock='unknown')):
            with self.subTest(change=change),self.assertRaises(ValueError):
                VelocityNet(2,'ode_diagonal',**change)
        with self.assertRaises(ValueError):VelocityNet(2,'swiglu',diagonal_solver='exp')


if __name__=='__main__':unittest.main()
