"""Research controls for a known GLU, including a test-only oracle construction."""
import itertools
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from scripts.run_architecture_study import macs
from synthetic.data import RotatedGaussianMixture, suite
from synthetic.models import SigmoidGLUFFN, VelocityNet


class SigmoidGLUTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(1)

    def pair(self, dim, dtype=torch.float64):
        models, states = [], []
        for kind in ('swiglu', 'glu'):
            torch.manual_seed(0)
            models.append(VelocityNet(dim, kind, ffn_hidden=128).to(dtype))
            states.append(torch.get_rng_state().clone())
        torch.testing.assert_close(states[0], states[1], atol=0, rtol=0)
        return models

    def test_native_tensor_buffer_rng_parameter_and_executed_MAC_pairing(self):
        for dim in (2, 16):
            models = self.pair(dim, torch.float32)
            for collection in ('state_dict', 'named_buffers'):
                left = dict(getattr(models[0], collection)())
                right = dict(getattr(models[1], collection)())
                self.assertEqual(set(left), set(right))
                for name in left:
                    torch.testing.assert_close(left[name], right[name], atol=0, rtol=0)
            self.assertEqual(models[0].parameter_count(), models[1].parameter_count())
            measured = []
            linear = F.linear
            for model in models:
                count = [0]
                def counted(x, weight, bias=None):
                    count[0] += (x.numel()//x.shape[-1])*weight.numel()
                    return linear(x, weight, bias)
                with patch.object(F, 'linear', counted), torch.no_grad():
                    model(torch.zeros(1, dim), torch.zeros(1, 1))
                measured.append(count[0])
                self.assertEqual(count[0], macs(model, dim))
            self.assertEqual(measured[0], measured[1])
            if dim == 16:
                self.assertEqual(models[1].parameter_count(), 83598)
                self.assertEqual(measured[1], 81792)

    def test_zero_head_masks_functions_but_not_head_parameter_response(self):
        swi, glu = self.pair(2)
        x = torch.tensor([[-1., .3], [.2, -.6], [1.1, .7]], dtype=torch.float64)
        t = torch.tensor([[.2], [.5], [.8]], dtype=x.dtype)
        gradients = []
        for model in (swi, glu):
            prediction = model(x, t)
            self.assertEqual(int(torch.count_nonzero(prediction)), 0)
            head, backbone = torch.autograd.grad(prediction.sum(), (model.output.weight, model.input.weight))
            gradients.append(head)
            self.assertEqual(int(torch.count_nonzero(backbone)), 0)
        self.assertGreater(float((gradients[0]-gradients[1]).norm()), 1e-5)
        # The same nonzero receiving head exposes the different hidden maps.
        receiving = torch.linspace(-.1, .1, 128, dtype=x.dtype).reshape(2, 64)
        for model in (swi, glu):
            with torch.no_grad():model.output.weight.copy_(receiving)
        self.assertGreater(float((swi(x, t)-glu(x, t)).detach().norm()), 1e-7)
        self.assertGreater(float((torch.autograd.functional.jacobian(lambda z: swi(z, t), x)
                                  -torch.autograd.functional.jacobian(lambda z: glu(z, t), x)).norm()), 1e-7)

    def test_fixed_time_raw_GLU_representation_against_population_oracles(self):
        # These analytic weights prove existence in a raw-input GLU only. They
        # are never used by the trainer, formal initialization, or a time-shared
        # model. The 3D oracle enumerates all eight Gaussian components rather
        # than reusing the product/tanh implementation.
        rng = np.random.default_rng(0)
        rotation, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        locations = np.array([.8, 1.1, 1.5])
        variances = np.array([.04, .08, .16])
        signs = np.array(list(itertools.product((-1., 1.), repeat=3)))
        enumerated = RotatedGaussianMixture(signs*locations, np.tile(variances, (8, 1)), np.ones(8), rotation)
        product16 = suite()['rotated_product_bimodal_16d']
        cases = ((enumerated, locations, variances, rotation),
                 (product16, product16.locations, product16.variances, product16.rotation))
        for oracle, mu, variance, rotation in cases:
            dim = len(mu)
            x = torch.from_numpy(rng.standard_normal((13, dim))*1.3)
            for time in (.1, .5, .9):
                a = 1-time
                denominator = a*a+time*time*variance
                linear = torch.from_numpy((time*variance-a)/denominator)
                amplitude = torch.from_numpy(a*mu/denominator)
                slope = torch.from_numpy(time*mu/denominator)
                r = torch.from_numpy(rotation)
                # Nominal3d gives exactly2d actual GLU channels; this is a
                # mathematical unit example, not an experimental width arm.
                model = SigmoidGLUFFN(dim, 3*dim).double()
                hidden = model.w3.in_features
                self.assertEqual(hidden, 2*dim)
                with torch.no_grad():
                    for parameter in model.parameters():parameter.zero_()
                    model.w12.weight[dim:2*dim].copy_(2*slope[:, None]*r.T)
                    model.w12.weight[hidden:hidden+dim].copy_(r.T)
                    model.w12.bias[hidden+dim:hidden+2*dim].fill_(1.)
                    model.w3.weight[:, :dim].copy_(2*r*linear[None, :])
                    model.w3.weight[:, dim:2*dim].copy_(2*r*amplitude[None, :])
                    model.w3.bias.copy_(-(r@amplitude))
                expected = torch.from_numpy(oracle.velocity(x.numpy(), time))
                torch.testing.assert_close(model(x), expected, atol=2e-13, rtol=2e-13)

    def test_native_input_jacobian_and_all_parameter_gradients(self):
        model = SigmoidGLUFFN(4, 6).double()
        x = torch.tensor([-.8, .1, .7, 1.2], dtype=torch.float64, requires_grad=True)
        gate_weight, value_weight = model.w12.weight.chunk(2)
        gate_bias, value_bias = model.w12.bias.chunk(2)
        probability = torch.sigmoid(F.linear(x, gate_weight, gate_bias))
        value = F.linear(x, value_weight, value_bias)
        expected = model.w3.weight@(
            torch.diag(value*probability*(1-probability))@gate_weight
            +torch.diag(probability)@value_weight)
        actual = torch.autograd.functional.jacobian(model, x)
        torch.testing.assert_close(actual, expected, atol=2e-16, rtol=2e-14)
        gradients = torch.autograd.grad(model(x).square().sum(), (x, *model.parameters()))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))


if __name__ == '__main__':
    unittest.main()
