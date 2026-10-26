import copy
import io
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from scripts.run_architecture_study import macs
from scripts.check_gate_geometry import gate_context,jacobian
from synthetic.models import VectorODEFFN, VectorODELayer, VelocityNet


def actual_linear_macs(call):
    """Instrument executed dense products, including functional cached products."""
    linear = F.linear
    count = 0
    def measured(x, weight, bias=None):
        nonlocal count
        count += (x.numel() // x.shape[-1]) * weight.numel()
        return linear(x, weight, bias)
    with patch.object(F, 'linear', measured), torch.no_grad():
        call()
    return count


def parameter_kernel(model, x):
    outputs = model(x).flatten()
    parameters = tuple(model.parameters())
    rows = []
    for value in outputs:
        gradient = torch.autograd.grad(value, parameters, retain_graph=True)
        rows.append(torch.cat([g.flatten() for g in gradient]))
    response = torch.stack(rows)
    return response @ response.T


class SharedFeatureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(1)

    def pair(self, width=4):
        layers = []
        for mode in ('shared_projection', 'paired_untied'):
            torch.manual_seed(0)
            layers.append(VectorODELayer(width, width, orders=1,
                reference_gate_hidden=width, feature_mode=mode).double())
        return layers

    def test_formula_input_jacobian_and_shared_P_gradient(self):
        shared, untied = self.pair()
        x = torch.randn(4, dtype=torch.float64)
        q = torch.randn_like(x)
        p = shared.proj.weight
        z = p @ x
        u = z + shared.t_func[0].bias
        logit = shared.t_func[2](F.silu(u))
        probability = torch.sigmoid(logit / 10.)
        duration = .2 + .8 * probability
        phi_prime = torch.sigmoid(u) * (1 + u * (1 - torch.sigmoid(u)))
        gate_z = torch.diag(.08 * probability * (1 - probability)) @ shared.t_func[2].weight @ torch.diag(phi_prime)
        expected_j = torch.eye(4, dtype=x.dtype) + torch.diag(duration) @ p + torch.diag(z) @ gate_z @ p
        torch.testing.assert_close(shared(x), x + duration * z, atol=0, rtol=0)
        actual_j = torch.autograd.functional.jacobian(shared, x)
        torch.testing.assert_close(actual_j, expected_j, atol=2e-15, rtol=2e-13)
        torch.testing.assert_close(shared(x), untied(x), atol=2e-15, rtol=2e-13)
        g_shared = torch.autograd.grad((shared(x) * q).sum(), shared.proj.weight)[0]
        g_p, g_r = torch.autograd.grad((untied(x) * q).sum(), (untied.proj.weight, untied.t_func[0].weight))
        value = torch.outer(duration * q, x)
        feature = torch.outer(gate_z.T @ (z * q), x)
        torch.testing.assert_close(g_shared, value + feature, atol=2e-15, rtol=2e-13)
        torch.testing.assert_close(g_p, value, atol=2e-15, rtol=2e-13)
        torch.testing.assert_close(g_r, feature, atol=2e-15, rtol=2e-13)
        torch.testing.assert_close(g_shared, g_p + g_r, atol=2e-15, rtol=2e-13)
        self.assertGreater(float(feature.detach().norm()), 1e-7)
        stats = shared.activation_stats(x[None])
        self.assertTrue(stats['projection_shared_with_controller'])
        self.assertEqual(stats['feature_mode'], 'shared_projection')
        self.assertAlmostEqual(stats['clock_raw_logit_sensitivity_mean'], float((.08 * probability * (1-probability)).mean().detach()), places=14)

    def test_zero_gate_initial_complete_ffn_kernel_equals_untied(self):
        models = []
        rng = []
        for mode in ('shared_projection', 'paired_untied'):
            torch.manual_seed(0)
            ffn = VectorODEFFN(4, 4, reference_gate_hidden=4,
                              orders=1, feature_mode=mode).double()
            rng.append(torch.get_rng_state())
            for layer in (ffn.ode1, ffn.ode2):
                with torch.no_grad():
                    layer.t_func[-1].weight.zero_(); layer.t_func[-1].bias.zero_()
            models.append(ffn)
        torch.testing.assert_close(rng[0], rng[1], atol=0, rtol=0)
        x = torch.randn(3, 4, dtype=torch.float64)
        torch.testing.assert_close(models[0](x), models[1](x), atol=0, rtol=0)
        torch.testing.assert_close(parameter_kernel(models[0], x), parameter_kernel(models[1], x), atol=2e-14, rtol=2e-13)
        for layer in (models[1].ode1, models[1].ode2):
            gradient = torch.autograd.grad(models[1](x).sum(), layer.t_func[0].weight)[0]
            self.assertEqual(torch.count_nonzero(gradient), 0)

    def test_nonzero_gate_kernel_change_can_have_either_sign(self):
        differences = []
        for sign in (-1., 1.):
            shared, untied = self.pair(1)
            with torch.no_grad():
                for layer in (shared, untied):
                    layer.proj.weight.fill_(.5)
                    layer.t_func[0].weight.fill_(.5)
                    layer.t_func[-1].weight.fill_(sign * 2.)
                    layer.t_func[-1].bias.zero_()
            x = torch.ones(1, 1, dtype=torch.float64)
            torch.testing.assert_close(shared(x), untied(x), atol=0, rtol=0)
            a,b = torch.autograd.grad(untied(x).sum(), (untied.proj.weight, untied.t_func[0].weight))
            delta = parameter_kernel(shared, x) - parameter_kernel(untied, x)
            torch.testing.assert_close(delta.squeeze(), 2 * (a*b).sum(), atol=2e-16, rtol=2e-13)
            differences.append(float(delta))
        self.assertLess(differences[0], 0)
        self.assertGreater(differences[1], 0)

    def test_alias_serialization_conversion_and_parameter_deduplication(self):
        shared, _ = self.pair()
        for model in (shared, copy.deepcopy(shared), shared.float().double()):
            self.assertIs(model.t_func[0].weight, model.proj.weight)
            self.assertEqual(sum(p.numel() for p in model.parameters()), 2*4*4+2*4)
        buffer = io.BytesIO()
        torch.save(shared.state_dict(), buffer); buffer.seek(0)
        state = torch.load(buffer, weights_only=True)
        for assign in (False, True):
            target, _ = self.pair()
            target.load_state_dict(state, assign=assign)
            self.assertIs(target.t_func[0].weight, target.proj.weight)
            x = torch.randn(5, 4, dtype=torch.float64)
            torch.testing.assert_close(target(x), shared(x), atol=0, rtol=0)
        # The same alias guarantee holds when loading through a parent model.
        model = VelocityNet(2, 'ode_vector', width=4, depth=2, gate_hidden=4,
                            orders=1, ode_feature_mode='shared_projection')
        model.load_state_dict(copy.deepcopy(model.state_dict()), assign=True)
        for _,layer in model.ode_layers():
            self.assertIs(layer.t_func[0].weight, layer.proj.weight)

    def test_real_dense_macs_and_full_width_budget(self):
        expected = {'independent': (32, 8288, 8192),
                    'shared_projection': (64, 8320, 8192),
                    'paired_untied': (64, 12416, 12288)}
        for mode,(hidden,parameters,dense_macs) in expected.items():
            layer = VectorODELayer(64, hidden, orders=1, feature_mode=mode)
            self.assertEqual(sum(p.numel() for p in layer.parameters()), parameters)
            self.assertEqual(actual_linear_macs(lambda: layer(torch.zeros(1,64))), dense_macs)
            net = VelocityNet(16, 'ode_vector', orders=1, gate_hidden=hidden, ode_feature_mode=mode)
            measured = actual_linear_macs(lambda: net(torch.zeros(1,16),torch.zeros(1,1)))
            self.assertEqual(macs(net,16), measured)

    def test_rejects_combinations_outside_single_feature_change(self):
        for change in (dict(kind='ode_original'), dict(orders=2), dict(gate_hidden=32),
                       dict(ode_clock_mode='tied'), dict(ode_clock_mode='reuse'),
                       dict(ode_middle_activation='identity'), dict(ode_operator_mode='additive'),
                       dict(ode_rate_placement='before_projection'), dict(ode_solve_return='increment'),
                       dict(ode_solve_pattern='map_increment'), dict(ode_gate_reparam=8.),
                       dict(ode_gate_transform='affine'), dict(tau=1.), dict(scale=1.6), dict(shift=0.)):
            options = dict(dim=2, kind='ode_vector', orders=1, gate_hidden=64,
                           ode_feature_mode='shared_projection')
            options.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                VelocityNet(**options)
        with self.assertRaises(ValueError):
            VectorODELayer(4,4,orders=1,feature_mode='shared_projection',spectral_cap=1.)

    def test_existing_gate_capture_and_freezing_interventions_remain_valid(self):
        for mode in ('shared_projection','paired_untied'):
            model=VelocityNet(2,'ode_vector',width=4,depth=2,gate_hidden=4,
                gate_reference_hidden=4,orders=1,ode_feature_mode=mode).double().eval()
            with torch.no_grad():model.output.weight.normal_(std=.3)
            x,t=torch.randn(5,2,dtype=torch.float64),torch.rand(5,1,dtype=torch.float64)
            native,full=jacobian(model,x,t);values={}
            with gate_context(model,values,capture=True):
                captured,derivative=jacobian(model,x,t)
            self.assertEqual(set(values),{name for name,_ in model.ode_layers()})
            torch.testing.assert_close(captured,native,atol=0,rtol=0)
            # Capture recomputes the shared product, changing gradient summation
            # order. The declared float64 Jacobian tolerance covers roundoff.
            torch.testing.assert_close(derivative,full,atol=2e-13,rtol=2e-13)
            with gate_context(model,values):anchored,frozen=jacobian(model,x,t)
            torch.testing.assert_close(anchored,native,atol=0,rtol=0)
            self.assertGreater(float((full-frozen).norm()),1e-9)
            self.assertTrue(all('step_size' not in layer.__dict__ for _,layer in model.ode_layers()))


if __name__ == '__main__':
    unittest.main()
