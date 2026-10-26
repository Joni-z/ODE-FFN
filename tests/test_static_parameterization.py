"""Function and initialization controls for static matrix parameterization."""
import unittest
import torch
from synthetic.models import VelocityNet,CollapsedODEFFN


class StaticParameterizationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)

    def test_hidden_functions_common_weights_and_rng_match(self) -> None:
        models=[]
        states=[]
        for kind in ('ode_fixed','ffn_poly_init'):
            torch.manual_seed(0)
            models.append(VelocityNet(2,kind,width=8,depth=2,gate_hidden=8).double())
            states.append(torch.get_rng_state())
        a,b=models
        self.assertEqual(a.parameter_count(),b.parameter_count())
        torch.testing.assert_close(states[0],states[1],atol=0,rtol=0)
        for key,value in a.state_dict().items():
            if '.ffn.' not in key:
                torch.testing.assert_close(value,b.state_dict()[key],atol=0,rtol=0)
        x=torch.randn(13,8,dtype=torch.float64)
        for left,right in zip(a.blocks,b.blocks):
            torch.testing.assert_close(left.ffn(x),right.ffn(x),atol=5e-7,rtol=5e-7)
        with torch.no_grad():
            a.output.weight.normal_()
            b.output.weight.copy_(a.output.weight)
        data=torch.randn(19,2,dtype=torch.float64)
        t=torch.rand(19,1,dtype=torch.float64)
        torch.testing.assert_close(a(data,t),b(data,t),atol=2e-6,rtol=2e-6)
        self.assertTrue(all(isinstance(block.ffn,CollapsedODEFFN) for block in b.blocks))


if __name__=='__main__':
    unittest.main()
