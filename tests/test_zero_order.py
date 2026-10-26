import copy
import unittest

import torch
from torch.nn import functional as F

from scripts.run_architecture_study import macs
from synthetic.models import OriginalODELayer, VectorODELayer, VelocityNet


class ZeroOrderTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def test_production_increment_uses_the_same_P_duration_and_activation(self):
        for cls in (OriginalODELayer,VectorODELayer):
            layer=cls(5,4,orders=1).double()
            increment=copy.deepcopy(layer);increment.solve_return='increment'
            x=torch.randn(17,5,dtype=torch.float64)
            expected=layer.step_size(x)*F.linear(x,layer.proj.weight)
            torch.testing.assert_close(increment(x),expected,atol=0,rtol=0)
            torch.testing.assert_close(layer(x),x+expected,atol=0,rtol=0)
            stats=increment.activation_stats(x)
            self.assertEqual(stats['solve_return'],'increment')
            self.assertAlmostEqual(stats['output_rms_over_input'],stats['term1_rms_over_input'],places=14)

    def test_legacy_map_increment_rng_parameters_and_macs_match(self):
        for kind,hidden in (('ode_original',16),('ode_vector',8)):
            options=dict(width=16,depth=2,orders=1,gate_hidden=hidden,gate_reference_hidden=16)
            torch.manual_seed(0);parent=VelocityNet(3,kind,**options);rng=torch.get_rng_state()
            torch.manual_seed(0);increment=VelocityNet(3,kind,**options,ode_solve_return='increment')
            torch.testing.assert_close(rng,torch.get_rng_state(),atol=0,rtol=0)
            for name,value in parent.state_dict().items():
                torch.testing.assert_close(value,increment.state_dict()[name],atol=0,rtol=0)
            self.assertEqual(macs(parent,3),macs(increment,3))
            self.assertEqual(parent.parameter_count(),increment.parameter_count())
            self.assertTrue(all(isinstance(b.ffn.act,torch.nn.SiLU) for b in increment.blocks))

    def test_constant_initializer_matches_complete_nonzero_head_functions(self):
        previous=torch.get_default_dtype()
        try:
            for dtype,tolerance in ((torch.float64,2e-13),(torch.float32,3e-6)):
                torch.set_default_dtype(dtype)
                for kind,hidden in (('ode_original',16),('ode_vector',8)):
                    options=dict(width=16,depth=2,orders=1,gate_hidden=hidden,
                        gate_reference_hidden=16,ode_solve_init='constant_paired')
                    torch.manual_seed(0);parent=VelocityNet(3,kind,**options);rng=torch.get_rng_state()
                    torch.manual_seed(0);increment=VelocityNet(3,kind,**options,ode_solve_return='increment')
                    torch.testing.assert_close(rng,torch.get_rng_state(),atol=0,rtol=0)
                    with torch.no_grad():
                        head=torch.randn_like(parent.output.weight)*.2
                        parent.output.weight.copy_(head);increment.output.weight.copy_(head)
                    x,t=torch.randn(61,3),torch.rand(61,1)
                    torch.testing.assert_close(parent(x,t),increment(x,t),atol=tolerance,rtol=tolerance)
        finally:
            torch.set_default_dtype(previous)

    def test_paired_constant_layer_changes_only_last_controller_gradient_initially(self):
        for cls in (OriginalODELayer,VectorODELayer):
            parent=cls(4,3,orders=1).double()
            with torch.no_grad():
                parent.t_func[-1].weight.zero_();parent.t_func[-1].bias.zero_()
            increment=copy.deepcopy(parent);increment.solve_return='increment'
            center=parent.shift+parent.scale/2
            with torch.no_grad():increment.proj.weight.add_(torch.eye(4,dtype=torch.float64)/center)
            x=torch.randn(13,4,dtype=torch.float64,requires_grad=True)
            q=torch.randn_like(x)
            torch.testing.assert_close(parent(x),increment(x),atol=1e-15,rtol=1e-14)
            ((parent(x))*q).sum().backward();((increment(x))*q).sum().backward()
            torch.testing.assert_close(parent.proj.weight.grad,increment.proj.weight.grad,atol=1e-15,rtol=1e-14)
            features=F.silu(parent.t_func[0](x)).detach()
            contribution=q*x.detach()/center
            if parent.t_func[-1].out_features==1:contribution=contribution.sum(dim=1,keepdim=True)
            sensitivity=parent.scale/(4*parent.tau)
            expected_weight=sensitivity*contribution.T@features
            expected_bias=sensitivity*contribution.sum(dim=0)
            torch.testing.assert_close(increment.t_func[-1].weight.grad-parent.t_func[-1].weight.grad,
                                       expected_weight,atol=2e-15,rtol=2e-13)
            torch.testing.assert_close(increment.t_func[-1].bias.grad-parent.t_func[-1].bias.grad,
                                       expected_bias,atol=2e-15,rtol=2e-13)
            self.assertEqual(torch.count_nonzero(parent.t_func[0].weight.grad),0)
            self.assertEqual(torch.count_nonzero(increment.t_func[0].weight.grad),0)

    def test_rejected_variant_combinations(self):
        for options in (dict(orders=2),dict(orders=1,ode_rate_placement='before_projection'),
                        dict(orders=1,ode_gate_reparam=8.),dict(orders=1,ode_middle_activation='identity')):
            with self.assertRaises(ValueError):
                VelocityNet(2,'ode_vector',ode_solve_return='increment',**options)


if __name__=='__main__':
    unittest.main()
