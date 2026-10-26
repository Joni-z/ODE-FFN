"""Mathematical and comparison invariants for the single bias-completeness test."""
import copy
import json
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from scripts.run_architecture_study import macs
from synthetic.models import VectorODELayer, VelocityNet


class AffineForcingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(1)

    def models(self, dim=2, dtype=torch.float64):
        models, rng = [], []
        for mode in ('none', 'static', 'conditional'):
            torch.manual_seed(0)
            models.append(VelocityNet(dim, 'ode_vector', gate_hidden=32, orders=1,
                          ode_solve_init='constant_paired', ode_forcing_mode=mode).to(dtype))
            rng.append(torch.get_rng_state().clone())
        for state in rng[1:]:torch.testing.assert_close(state, rng[0], atol=0, rtol=0)
        return models

    def active(self, mode='conditional'):
        layer = VectorODELayer(3, 4, reference_gate_hidden=4, orders=1,
                               forcing_mode=mode).double()
        with torch.no_grad():
            layer.proj.weight.copy_(torch.tensor([[.4, -.2, .1], [.1, .5, -.1], [-.2, .1, .6]]))
            layer.t_func[0].weight.copy_(torch.tensor([[.7, -.2, .3], [-.1, .8, .2], [.4, .3, -.6], [-.2, .5, .7]]))
            layer.t_func[0].bias.copy_(torch.tensor([.1, -.3, .2, .4]))
            layer.t_func[-1].weight.copy_(torch.tensor([[3., -2., 1., .4], [-1., 2., 4., -.2], [.7, -3., .8, 2.]]))
            layer.t_func[-1].bias.copy_(torch.tensor([.4, -.3, .2]))
            layer.forcing.copy_(torch.tensor([.8, -.5, .3]))
        return layer

    def test_native_common_tensors_buffers_rng_and_executed_cost(self):
        original_linear = F.linear
        for dim in (2, 16):
            parent, static, affine = self.models(dim, torch.float32)
            state, buffers = parent.state_dict(), dict(parent.named_buffers())
            self.assertFalse(any(name.endswith('.forcing') for name in state))
            expected_parameters, expected_macs = ((81858, 80192) if dim == 2 else (83664, 81984))
            self.assertEqual(parent.parameter_count(), expected_parameters)
            for model in (static, affine):
                new = model.state_dict()
                extra = set(new)-set(state)
                self.assertEqual(len(extra), 6)
                self.assertTrue(all(name.endswith('.forcing') and new[name].shape == (64,) for name in extra))
                self.assertTrue(all(torch.count_nonzero(new[name]) == 0 for name in extra))
                for name in state:torch.testing.assert_close(new[name], state[name], atol=0, rtol=0)
                other_buffers = dict(model.named_buffers())
                self.assertEqual(set(buffers), set(other_buffers))
                for name in buffers:torch.testing.assert_close(buffers[name], other_buffers[name], atol=0, rtol=0)
                self.assertEqual(model.parameter_count(), expected_parameters+384)
            for model in (parent, static, affine):
                count = [0]
                def counted(x, weight, bias=None):
                    count[0] += (x.numel()//x.shape[-1])*weight.numel()
                    return original_linear(x, weight, bias)
                with patch.object(F, 'linear', counted), torch.no_grad():
                    prediction = model(torch.zeros(1, dim), torch.zeros(1, 1))
                self.assertEqual(int(torch.count_nonzero(prediction)), 0)
                self.assertEqual(count[0], expected_macs)
                self.assertEqual(count[0], macs(model, dim))

    def test_whole_model_initial_full_parameter_responses_with_nonzero_head(self):
        # Every scalar output response is checked for every parameter, rather
        # than relying on one cotangent or the formal zero-head masking.
        for dtype, atol, rtol in ((torch.float32, 2e-6, 2e-5), (torch.float64, 2e-13, 2e-12)):
            parent, static, affine = self.models(dtype=dtype)
            x = torch.tensor([[-.8, .3], [.7, -.2]], dtype=dtype)
            t = torch.tensor([[.2], [.7]], dtype=dtype)
            for model in (parent, static, affine):
                with torch.no_grad():
                    model.output.weight.copy_(torch.linspace(-.2, .3, 128, dtype=dtype).reshape(2, 64))
            outputs = [model(x, t) for model in (parent, static, affine)]
            for y in outputs[1:]:torch.testing.assert_close(y, outputs[0], atol=0, rtol=0)
            parameter_maps = [dict(model.named_parameters()) for model in (parent, static, affine)]
            self.assertEqual(set(parameter_maps[1]), set(parameter_maps[2]))
            active_bias = False
            for index in range(outputs[0].numel()):
                gradients = [dict(zip(parameters, torch.autograd.grad(y.flatten()[index], tuple(parameters.values()), retain_graph=True)))
                             for parameters, y in zip(parameter_maps, outputs)]
                for name, value in gradients[0].items():
                    torch.testing.assert_close(gradients[1][name], value, atol=atol, rtol=rtol)
                for name, value in gradients[1].items():
                    torch.testing.assert_close(gradients[2][name], value, atol=atol, rtol=rtol)
                    if name.endswith('.forcing') and float(value.norm()) > 1e-5:active_bias = True
            self.assertTrue(active_bias, 'nonzero head must expose the added parameter responses')
            jacobians = [torch.autograd.functional.jacobian(lambda z: model(z, t), x) for model in (static, affine)]
            torch.testing.assert_close(jacobians[0], jacobians[1], atol=atol, rtol=rtol)

    def test_active_input_and_all_parameter_derivatives(self):
        affine = self.active(); static = copy.deepcopy(affine); static.forcing_mode = 'static'
        x = torch.tensor([.8, -.4, .6], dtype=torch.float64, requires_grad=True)
        r = torch.tensor([.3, -.7, .4], dtype=x.dtype)
        jacobians = []
        for layer in (static, affine):
            first, _, last = layer.t_func
            u = first(x); hidden = F.silu(u); inner = torch.sigmoid(u)
            inner_slope = inner+u*inner*(1-inner)
            probability = torch.sigmoid(last(hidden)/layer.tau)
            s = layer.shift+layer.scale*probability
            slope = layer.scale/layer.tau*probability*(1-probability)
            gate_j = torch.diag(slope)@last.weight@torch.diag(inner_slope)@first.weight
            ph = layer.proj.weight@x
            receiving = ph+(layer.forcing if layer.forcing_mode == 'conditional' else 0)
            expected_j = torch.eye(3, dtype=x.dtype)+torch.diag(s)@layer.proj.weight+torch.diag(receiving)@gate_j
            actual_j = torch.autograd.functional.jacobian(layer, x)
            torch.testing.assert_close(actual_j, expected_j, atol=3e-16, rtol=2e-14)
            jacobians.append(actual_j)
            gate_cotangent = r*receiving*slope
            first_cotangent = (last.weight.T@gate_cotangent)*inner_slope
            expected = {'proj.weight': (r*s)[:, None]*x[None, :],
                        'forcing': r*(s if layer.forcing_mode == 'conditional' else layer.shift+layer.scale/2),
                        't_func.2.weight': gate_cotangent[:, None]*hidden[None, :],
                        't_func.2.bias': gate_cotangent,
                        't_func.0.weight': first_cotangent[:, None]*x[None, :],
                        't_func.0.bias': first_cotangent}
            parameters = dict(layer.named_parameters())
            gradients = torch.autograd.grad((layer(x)*r).sum(), (x, *parameters.values()))
            torch.testing.assert_close(gradients[0], expected_j.T@r, atol=3e-16, rtol=2e-14)
            self.assertEqual(set(parameters), set(expected))
            for name, value in zip(parameters, gradients[1:]):
                torch.testing.assert_close(value, expected[name], atol=3e-16, rtol=2e-14)
        torch.testing.assert_close(jacobians[1]-jacobians[0], torch.diag(affine.forcing)@gate_j, atol=3e-16, rtol=2e-14)

    def test_zero_bias_alone_does_not_pair_new_parameter_columns(self):
        affine = self.active(); static = copy.deepcopy(affine); static.forcing_mode = 'static'
        with torch.no_grad():affine.forcing.zero_(); static.forcing.zero_()
        x = torch.tensor([[.8, -.4, .6]], dtype=torch.float64)
        torch.testing.assert_close(affine(x), static(x), atol=0, rtol=0)
        ga = torch.autograd.grad(affine(x).sum(), affine.forcing)[0]
        gb = torch.autograd.grad(static(x).sum(), static.forcing)[0]
        self.assertGreater(float((ga-gb).norm()), 1e-3)
        torch.testing.assert_close(ga-gb, (affine.step_size(x)-(affine.shift+affine.scale/2)).sum(0), atol=0, rtol=0)

    def test_active_statistics_mean_variance_and_no_state_mutation(self):
        x = torch.linspace(-1.1, 1.4, 18, dtype=torch.float64).reshape(2, 3, 3)
        rms = lambda value: float(value.detach().square().mean().sqrt())
        for mode in ('static', 'conditional'):
            layer = self.active(mode); before = copy.deepcopy(layer.state_dict())
            stats = layer.activation_stats(x); json.dumps(stats, allow_nan=False)
            for name in before:torch.testing.assert_close(before[name], layer.state_dict()[name], atol=0, rtol=0)
            s = layer.step_size(x); c = layer.shift+layer.scale/2
            static = c*layer.forcing
            correction = (s-c)*layer.forcing if mode == 'conditional' else torch.zeros_like(x)
            actual = s*layer.forcing if mode == 'conditional' else static
            mean = correction.reshape(-1, 3).mean(0); denominator = rms(x)
            homogeneous = s*F.linear(x, layer.proj.weight)
            expected = {'forcing_parameter_norm': float(layer.forcing.detach().norm()),
                        'forcing_rms_over_input': rms(actual)/denominator,
                        'static_bias_rms_over_input': rms(static)/denominator,
                        'conditional_bias_rms_over_input': rms(correction)/denominator,
                        'conditional_bias_mean_rms_over_input': rms(mean)/denominator,
                        'conditional_bias_centered_rms_over_input': rms(correction-mean)/denominator,
                        'homogeneous_term1_rms_over_input': rms(homogeneous)/denominator,
                        'term1_rms_over_input': rms(homogeneous+actual)/denominator,
                        'output_rms_over_input': rms(x+homogeneous+actual)/denominator}
            self.assertEqual(stats['forcing_mode'], mode)
            for name, value in expected.items():self.assertAlmostEqual(stats[name], value, places=14, msg=name)
            self.assertAlmostEqual(rms(correction)**2, rms(mean)**2+rms(correction-mean)**2, places=14)
            if mode == 'conditional':
                self.assertGreater(stats['conditional_bias_centered_rms_over_input'], 1e-4)
                self.assertNotAlmostEqual(stats['term1_rms_over_input'], stats['homogeneous_term1_rms_over_input'])
                frozen = copy.deepcopy(layer); frozen.forcing_mode = 'static'
                differences = [torch.autograd.functional.jacobian(layer, point)
                               -torch.autograd.functional.jacobian(frozen, point)
                               for point in x.reshape(-1, 3)]
                sensitivity = float(torch.linalg.matrix_norm(torch.stack(differences), ord='fro').mean())
                self.assertAlmostEqual(stats['conditional_bias_local_jacobian_frobenius_mean'], sensitivity, places=14)
                self.assertGreater(sensitivity, 1e-4)
            else:
                self.assertEqual(stats['conditional_bias_rms_over_input'], 0.)
                self.assertEqual(stats['conditional_bias_local_jacobian_frobenius_mean'], 0.)

    def test_single_layer_translation_conjugacy_with_shifted_controller(self):
        # This is a local existence identity, not a full LayerNorm/FiLM/SiLU
        # backbone equivalence or an oracle-derived training initializer.
        affine = self.active()
        homogeneous = VectorODELayer(3, 4, reference_gate_hidden=4, orders=1).double()
        homogeneous.load_state_dict({k: v for k, v in affine.state_dict().items() if k != 'forcing'})
        translation = torch.linalg.solve(affine.proj.weight.detach(), affine.forcing.detach())
        with torch.no_grad():
            homogeneous.t_func[0].bias.sub_(homogeneous.t_func[0].weight@translation)
        x = torch.tensor([[.4, -.7, .2], [-.3, .1, .8]], dtype=torch.float64)
        translated = lambda z: homogeneous(z+translation)-translation
        torch.testing.assert_close(affine(x), translated(x), atol=6e-16, rtol=3e-15)
        torch.testing.assert_close(torch.autograd.functional.jacobian(affine, x),
                                   torch.autograd.functional.jacobian(translated, x), atol=3e-16, rtol=3e-14)

    def test_unsupported_combinations_are_rejected(self):
        valid = dict(gate_hidden=32, orders=1, ode_solve_init='constant_paired', ode_forcing_mode='conditional')
        for change in (dict(ode_forcing_mode='affine'), dict(orders=2), dict(ode_solve_init='legacy'),
                       dict(ode_feature_mode='shared_projection'), dict(ode_operator_mode='additive'),
                       dict(ode_middle_activation='identity'), dict(ode_clock_mode='tied'),
                       dict(ode_gate_transform='affine'), dict(ode_gate_reparam=2.),
                       dict(ode_rate_placement='before_projection'), dict(ode_solve_return='increment'),
                       dict(ode_solve_pattern='increment_increment'), dict(tau=1.), dict(scale=1.), dict(shift=0.),
                       dict(spectral_cap=.5)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                VelocityNet(2, 'ode_vector', **dict(valid, **change))
        for kind in ('swiglu', 'glu', 'ode_original', 'ode_state_clock'):
            with self.subTest(kind=kind), self.assertRaises(ValueError):VelocityNet(2, kind, **valid)
        for change in (dict(orders=2), dict(gate_source='fixed'), dict(spectral_cap=1.),
                       dict(clock_curvature=0.), dict(gate_transform='affine'), dict(rate_placement='before_projection'),
                       dict(solve_return='increment'), dict(operator_mode='additive')):
            options = dict(orders=1, forcing_mode='conditional'); options.update(change)
            with self.subTest(layer=change), self.assertRaises(ValueError):VectorODELayer(3, 4, **options)


if __name__ == '__main__':
    unittest.main()
