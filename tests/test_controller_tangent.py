import unittest

import numpy as np
import torch

from scripts.controller_tangent import kernels,replace_output
from synthetic.models import VelocityNet
from scripts.diagnose_controller_tangent import ridge_summary


class ControllerTangentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0);torch.set_num_threads(1)

    def model(self,kind,mode='multiplicative'):
        model=VelocityNet(2,kind,width=4,depth=2,orders=1,gate_hidden=3,gate_reference_hidden=4,
            ffn_hidden=6,ode_operator_mode=mode).double()
        with torch.no_grad():model.output.weight.normal_(std=.3)
        return model.eval()

    @staticmethod
    def direct(model,x,t,filter_name):
        prediction=model(x,t).reshape(-1)
        parameters=[p for name,p in model.named_parameters() if filter_name(name)]
        jacobian=torch.stack([torch.cat([g.flatten() for g in torch.autograd.grad(
            v,parameters,retain_graph=True)]) for v in prediction])
        return jacobian@jacobian.T

    def test_exact_parameter_kernel_includes_all_downstream_gates(self):
        for kind,mode in [('ode_original','multiplicative'),('ode_vector','multiplicative'),
                          ('ode_original','additive'),('ode_vector','additive'),('swiglu','multiplicative')]:
            model=self.model(kind,mode);x,t=torch.randn(4,2,dtype=torch.float64),torch.rand(4,1,dtype=torch.float64)
            calculated=kernels(model,x,t)['components']
            direct=self.direct(model,x,t,lambda name:'.ffn.' in name)
            torch.testing.assert_close(calculated['full_ffn'],direct,atol=2e-14,rtol=2e-12)
            if kind!='swiglu':
                direct=self.direct(model,x,t,lambda name:'.t_func.' in name)
                torch.testing.assert_close(calculated['controller'],direct,atol=2e-14,rtol=2e-12)
                direct=self.direct(model,x,t,lambda name:'.proj.' in name)
                torch.testing.assert_close(calculated['projection_P'],direct,atol=2e-14,rtol=2e-12)

    def test_output_replacement_restored_after_exception(self):
        model=self.model('ode_vector');x,t=torch.randn(3,2,dtype=torch.float64),torch.rand(3,1,dtype=torch.float64)
        expected=model(x,t).detach()
        with self.assertRaises(RuntimeError):
            with replace_output(model.blocks[0].ffn.ode1,torch.zeros(3,4,dtype=torch.float64)):
                raise RuntimeError('deliberate scope exit')
        torch.testing.assert_close(model(x,t),expected,atol=0,rtol=0)

    def test_zero_head_has_zero_response_without_nan(self):
        model=VelocityNet(2,'ode_vector',width=4,depth=1,orders=1,gate_hidden=3,gate_reference_hidden=4).double().eval()
        result=kernels(model,torch.randn(3,2,dtype=torch.float64),torch.rand(3,1,dtype=torch.float64))
        for matrix in result['components'].values():
            self.assertEqual(torch.count_nonzero(matrix),0);self.assertTrue(torch.isfinite(matrix).all())

    def test_rejects_shared_parameters_and_training_mode(self):
        x,t=torch.randn(3,2,dtype=torch.float64),torch.rand(3,1,dtype=torch.float64)
        model=self.model('ode_vector');model.train()
        with self.assertRaises(ValueError):kernels(model,x,t)
        self.assertTrue(model.training)
        model.eval()
        model.blocks[1].ffn.ode1.proj.weight=model.blocks[0].ffn.ode1.proj.weight
        with self.assertRaises(ValueError):kernels(model,x,t)

    def test_rejects_internal_value_controller_parameter_sharing(self):
        model=VelocityNet(2,'ode_vector',width=4,depth=1,orders=1,gate_hidden=4,gate_reference_hidden=4).double().eval()
        layer=model.blocks[0].ffn.ode1
        layer.t_func[0].weight=layer.proj.weight
        with self.assertRaises(ValueError):
            kernels(model,torch.randn(3,2,dtype=torch.float64),torch.rand(3,1,dtype=torch.float64))

    def test_saturated_sigmoid_parameter_kernel_matches_autograd(self):
        model=self.model('ode_vector')
        with torch.no_grad():
            for _,layer in model.ode_layers():layer.t_func[-1].bias.fill_(-30*layer.tau)
        x,t=torch.randn(3,2,dtype=torch.float64),torch.rand(3,1,dtype=torch.float64)
        calculated=kernels(model,x,t)['components']['controller']
        direct=self.direct(model,x,t,lambda name:'.t_func.' in name)
        torch.testing.assert_close(calculated,direct,atol=0,rtol=2e-11)

    def test_ridge_dual_matches_primal_on_shared_features(self):
        rng=np.random.default_rng(0)
        feature=rng.normal(size=(12,5));residual=rng.normal(size=(6,2))
        fit=np.array([0,2,4]);evaluation=np.array([1,3,5])
        result,arrays=ridge_summary(feature@feature.T,residual,fit,evaluation)
        calibration=np.concatenate([np.arange(i*2,i*2+2) for i in fit])
        check=np.concatenate([np.arange(i*2,i*2+2) for i in evaluation])
        a,b=feature[calibration],feature[check]
        scale=np.sum(a*a)/len(a)
        for row in result['rows']:
            penalty=row['penalty']
            update=np.linalg.solve(a.T@a+penalty*scale*np.eye(5),a.T@residual[fit].reshape(-1))
            np.testing.assert_allclose(arrays[f'evaluation_correction_{penalty}'].reshape(-1),b@update,rtol=2e-10,atol=2e-10)
            np.testing.assert_allclose(row['feature_update_norm'],np.linalg.norm(update),rtol=2e-10,atol=2e-10)


if __name__=='__main__':unittest.main()
