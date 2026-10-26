"""Research invariants of the one pruned, factorized K1 allocation proposal."""
import copy
import json
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from scripts.run_architecture_study import macs
from synthetic.augmented import AugmentedODEFFN, PrunedLowRankLayer
from synthetic.diagnostics import collect
from synthetic.models import OriginalODELayer, VectorODEFFN, VectorODELayer, VelocityNet


class AugmentedStateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(1)

    def ffn(self, expanded=True, width=5, rank=2, gate=2):
        reference = VectorODEFFN(width, 4, 4, orders=1).double()
        return AugmentedODEFFN(reference, width=width, state_width=width*(2 if expanded else 1), rank=rank, gate_hidden=gate)

    def models(self, dim=2, dtype=torch.float64):
        result, states = {}, []
        for label, state_width, rank, gate in (('parent', 64, 64, 32), ('factor32', 64, 32, 32),
                                             ('narrow21', 64, 21, 21), ('aug21', 128, 21, 21)):
            torch.manual_seed(0)
            if label == 'parent':
                model = VelocityNet(dim, 'ode_vector', orders=1, gate_hidden=32, ode_solve_init='constant_paired')
            else:
                model = VelocityNet(dim, 'ode_lowrank', orders=1, gate_hidden=32, lowrank_state_width=state_width,
                                    lowrank_rank=rank, lowrank_gate_hidden=gate)
            result[label] = model.to(dtype); states.append(torch.get_rng_state().clone())
        for value in states[1:]:torch.testing.assert_close(value, states[0], atol=0, rtol=0)
        return result

    def activate(self, layer):
        with torch.no_grad():
            for value in (layer.left, layer.right, layer.t_func[-1].weight, layer.t_func[-1].bias):
                value.copy_(torch.linspace(-.7, .9, value.numel(), dtype=value.dtype).reshape_as(value))

    def analytic(self, layer, x):
        first, _, last = layer.t_func
        u = first(x); sigmoid = torch.sigmoid(u)
        inner_slope = sigmoid+u*sigmoid*(1-sigmoid)
        probability = torch.sigmoid(last(F.silu(u))/layer.tau)
        s = layer.shift+layer.scale*probability
        slope = layer.scale/layer.tau*probability*(1-probability)
        js = torch.diag(slope)@last.weight@torch.diag(inner_slope)@first.weight
        carrier = layer.left@(layer.right@x)
        skip = torch.eye(layer.output_width, layer.input_width, dtype=x.dtype)
        return skip+torch.diag(s)@layer.left@layer.right+torch.diag(carrier)@js, (u, inner_slope, slope, s, carrier)

    def test_rectangular_input_factor_controller_and_full_FFN_derivatives(self):
        model = self.ffn()
        for layer in (model.ode1, model.ode2):
            self.activate(layer)
            x = torch.linspace(-.8, .9, layer.input_width, dtype=torch.float64, requires_grad=True)
            receiver = torch.linspace(.3, -.6, layer.output_width, dtype=x.dtype)
            expected_j, (u, inner_slope, slope, s, carrier) = self.analytic(layer, x)
            actual_j = torch.autograd.functional.jacobian(layer, x)
            torch.testing.assert_close(actual_j, expected_j, atol=8e-16, rtol=3e-14)
            skip = torch.eye(layer.output_width, layer.input_width, dtype=x.dtype)
            self.assertLessEqual(int(torch.linalg.matrix_rank(actual_j-skip, atol=1e-12)), layer.rank+layer.gate_hidden)
            gate_cotangent = receiver*carrier*slope
            first_cotangent = (layer.t_func[-1].weight.T@gate_cotangent)*inner_slope
            expected = {'left': (s*receiver)[:, None]*(layer.right@x)[None, :],
                        'right': (layer.left.T@(s*receiver))[:, None]*x[None, :],
                        't_func.2.weight': gate_cotangent[:, None]*F.silu(u)[None, :],
                        't_func.2.bias': gate_cotangent,
                        't_func.0.weight': first_cotangent[:, None]*x[None, :],
                        't_func.0.bias': first_cotangent}
            parameters = dict(layer.named_parameters())
            gradients = torch.autograd.grad((layer(x)*receiver).sum(), (x, *parameters.values()))
            torch.testing.assert_close(gradients[0], expected_j.T@receiver, atol=8e-16, rtol=3e-14)
            self.assertEqual(set(parameters), set(expected))
            for name, gradient in zip(parameters, gradients[1:]):
                torch.testing.assert_close(gradient, expected[name], atol=8e-16, rtol=3e-14)
        x = torch.linspace(-.8, .9, 5, dtype=torch.float64)
        z = model.ode1(x); probability = torch.sigmoid(z)
        middle = torch.diag(probability+z*probability*(1-probability))
        expected = self.analytic(model.ode2, F.silu(z))[0]@middle@self.analytic(model.ode1, x)[0]
        torch.testing.assert_close(torch.autograd.functional.jacobian(model, x), expected, atol=2e-15, rtol=3e-14)

    def test_formal_common_RNG_backbone_counts_and_actual_dense_MACs(self):
        original = F.linear
        for dim in (2, 16):
            models = self.models(dim, torch.float32); parent = models['parent']
            shared = {k: v for k, v in parent.state_dict().items() if '.ffn.' not in k}
            expected = {'parent': (83664, 81984), 'factor32': (83664, 81984),
                        'narrow21': (66702, 65088), 'aug21': (83022, 81216)}
            for label, model in models.items():
                for name, value in shared.items():torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
                for name, value in parent.named_buffers():torch.testing.assert_close(dict(model.named_buffers())[name], value, atol=0, rtol=0)
                count = [0]
                def counted(x, weight, bias=None):
                    count[0] += (x.numel()//x.shape[-1])*weight.numel()
                    return original(x, weight, bias)
                with patch.object(F, 'linear', counted), torch.no_grad():model(torch.zeros(1, dim), torch.zeros(1, 1))
                parameters, work = expected[label]
                if dim == 2:parameters -= 1806; work -= 1792
                self.assertEqual(model.parameter_count(), parameters)
                self.assertEqual(count[0], work); self.assertEqual(macs(model, dim), work)
                if label != 'parent':
                    self.assertEqual(len(list(model.ode_layers())), 0)
                    self.assertEqual(len(list(model.lowrank_layers())), 6)
                    self.assertEqual(sum(p.numel() for p in model.blocks[0].ffn.parameters()),
                                     {'factor32': 16576, 'narrow21': 10922, 'aug21': 16362}[label])

    def test_common_SVD_core_truncation_balancing_and_no_extra_RNG(self):
        reference = VectorODELayer(5, 4, 4, orders=1).double()
        with torch.no_grad():reference.proj.weight.copy_(torch.diag(torch.tensor([5., 4., 3., 2., 1.], dtype=torch.float64)))
        rng = torch.get_rng_state().clone()
        lift = PrunedLowRankLayer(reference, width=5, state_width=10, rank=2, gate_hidden=2, role='lift')
        project = PrunedLowRankLayer(reference, width=5, state_width=10, rank=2, gate_hidden=2, role='project')
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
        truncated = torch.diag(torch.tensor([5., 4., 0., 0., 0.], dtype=torch.float64))
        torch.testing.assert_close(lift.left[:5]@lift.right, truncated, atol=1e-15, rtol=1e-15)
        residual = reference.proj.weight-lift.left[:5]@lift.right
        self.assertAlmostEqual(float(torch.linalg.matrix_norm(residual.detach(), ord=2)), 3.)
        self.assertAlmostEqual(float(residual.detach().square().sum()), 14.)
        torch.testing.assert_close(lift.left[5:], lift.left[:5], atol=0, rtol=0)
        torch.testing.assert_close(lift.left[:5].T@lift.left[:5], lift.right@lift.right.T, atol=1e-15, rtol=1e-15)
        torch.testing.assert_close(lift.left.T@lift.left, 2*lift.right@lift.right.T, atol=2e-15, rtol=1e-15)
        self.assertEqual(int(torch.count_nonzero(project.right[:, 5:])), 0)
        self.assertEqual(int(torch.count_nonzero(project.t_func[0].weight[:, 5:])), 0)
        for layer in (lift, project):
            torch.testing.assert_close(layer.t_func[0].weight[:, :5], reference.t_func[0].weight[:2], atol=0, rtol=0)
            self.assertEqual(int(torch.count_nonzero(layer.t_func[-1].weight)), 0)
            self.assertEqual(int(torch.count_nonzero(layer.t_func[-1].bias)), 0)

    def test_narrow_augmented_functions_and_masked_versus_live_parameter_responses(self):
        def common(name, value):
            if any(name.endswith(suffix) for suffix in ('.ode1.left', '.ode1.t_func.2.weight', '.ode1.t_func.2.bias')):return value[:64]
            if any(name.endswith(suffix) for suffix in ('.ode2.right', '.ode2.t_func.0.weight')):return value[..., :64]
            return value
        for dtype, atol, rtol in ((torch.float32, 3e-6, 5e-5), (torch.float64, 3e-13, 3e-12)):
            models = self.models(dtype=dtype); narrow, expanded = models['narrow21'], models['aug21']
            x = torch.tensor([[-.7, .4]], dtype=dtype); t = torch.tensor([[.6]], dtype=dtype)
            parameter_maps = [dict(m.named_parameters()) for m in (narrow, expanded)]
            self.assertEqual(set(parameter_maps[0]), set(parameter_maps[1]))
            for name, parameter in parameter_maps[0].items():torch.testing.assert_close(common(name, parameter_maps[1][name]), parameter, atol=0, rtol=0)
            # At the native zero output head, all backbone columns vanish and
            # shared output-head columns match: this initial full NTK is paired.
            predictions = [m(x, t) for m in (narrow, expanded)]
            for index in range(2):
                gradients = [dict(zip(p, torch.autograd.grad(y[0, index], tuple(p.values()), retain_graph=True))) for p, y in zip(parameter_maps, predictions)]
                for name, gradient in gradients[0].items():
                    torch.testing.assert_close(common(name, gradients[1][name]), gradient, atol=atol, rtol=rtol)
                    if not name.startswith('output.'):
                        self.assertEqual(int(torch.count_nonzero(gradient)), 0)
                        self.assertEqual(int(torch.count_nonzero(gradients[1][name])), 0)
            for model in (narrow, expanded):
                with torch.no_grad():model.output.weight.copy_(torch.linspace(-.2, .3, 128, dtype=dtype).reshape(2, 64))
            predictions = [m(x, t) for m in (narrow, expanded)]
            torch.testing.assert_close(predictions[0], predictions[1], atol=atol, rtol=rtol)
            torch.testing.assert_close(torch.autograd.functional.jacobian(lambda value: narrow(value, t), x),
                                       torch.autograd.functional.jacobian(lambda value: expanded(value, t), x), atol=atol, rtol=rtol)
            gradients = [dict(zip(p, torch.autograd.grad(y[0, 0], tuple(p.values())))) for p, y in zip(parameter_maps, predictions)]
            for name, gradient in gradients[0].items():torch.testing.assert_close(common(name, gradients[1][name]), gradient, atol=atol, rtol=rtol)
            live = sum(float(value[:, 64:].square().sum()) for name, value in gradients[1].items() if name.endswith('.ode2.right'))
            self.assertGreater(live, 1e-6)
            for name, value in gradients[1].items():
                if name.endswith('.ode1.left'):self.assertEqual(int(torch.count_nonzero(value[64:])), 0)
            if dtype == torch.float64:
                difference = sum(float(v.square().sum()) for v in gradients[1].values())-sum(float(v.square().sum()) for v in gradients[0].values())
                self.assertAlmostEqual(difference, live, places=11)

    def test_nonzero_extra_producer_prevents_a_dead_new_branch(self):
        model = self.ffn(); x = torch.linspace(-1., .9, 20, dtype=torch.float64).reshape(4, 5)
        head = torch.tensor([.3, -.4, .7, .2, -.1], dtype=x.dtype)
        live = torch.autograd.grad((model(x)*head).sum(), model.ode2.right)[0][:, 5:]
        self.assertGreater(float(live.norm()), 1e-5)
        dead = copy.deepcopy(model)
        with torch.no_grad():dead.ode1.left[5:].zero_()
        producer, receiver = torch.autograd.grad((dead(x)*head).sum(), (dead.ode1.left, dead.ode2.right))
        self.assertEqual(int(torch.count_nonzero(producer[5:])), 0)
        self.assertEqual(int(torch.count_nonzero(receiver[:, 5:])), 0)

    def test_pruned_lifted_operators_lowrank_powers_and_spectral_blind_blocks(self):
        model = self.ffn()
        for layer in (model.ode1, model.ode2):
            duration = torch.linspace(.3, .9, layer.output_width, dtype=torch.float64)
            full = layer.full_operator()
            if layer.role == 'lift':
                self.assertEqual(int(torch.count_nonzero(full[:, 5:])), 0)
                left, right, rates = layer.left, F.pad(layer.right, (0, 5)), duration
            else:
                self.assertEqual(int(torch.count_nonzero(full[5:])), 0)
                left, right, rates = F.pad(layer.left, (0, 0, 0, 5)), layer.right, F.pad(duration, (0, 5), value=1.)
            generator = rates[:, None]*full; compressed = layer.compressed_operator(duration)
            torch.testing.assert_close(compressed, right@(rates[:, None]*left), atol=1e-16, rtol=2e-14)
            batch = layer.compressed_operator(torch.stack((duration, duration*.8)))
            torch.testing.assert_close(batch, torch.stack((compressed, .8*compressed)), atol=2e-16, rtol=2e-14)
            for order in (1, 2, 3):
                expected = (rates[:, None]*left)@torch.linalg.matrix_power(compressed, order-1)@right
                torch.testing.assert_close(torch.linalg.matrix_power(generator, order), expected, atol=2e-16, rtol=3e-14)
            for scale in (-.3, .2):
                self.assertAlmostEqual(float(torch.linalg.det(torch.eye(10, dtype=full.dtype)+scale*generator).detach()),
                                       float(torch.linalg.det(torch.eye(2, dtype=full.dtype)+scale*compressed).detach()), places=13)
            changed = copy.deepcopy(layer)
            with torch.no_grad():
                if layer.role == 'lift':changed.left[5:].mul_(3.)
                else:changed.right[:, 5:].fill_(.4)
            torch.testing.assert_close(changed.compressed_operator(duration), compressed, atol=0, rtol=0)
            self.assertGreater(float((changed.full_operator()-full).detach().norm()), 1e-4)

    def test_active_statistics_collector_and_invalid_settings(self):
        model = self.ffn(); x = torch.linspace(-.7, .9, 20, dtype=torch.float64).reshape(4, 5)
        rms = lambda z: float(z.detach().square().mean().sqrt())
        for layer in (model.ode1, model.ode2):
            self.activate(layer); state = copy.deepcopy(layer.state_dict())
            inputs = x if layer.role == 'lift' else torch.cat((x, x*.3), dim=-1)
            stats = layer.activation_stats(inputs); json.dumps(stats, allow_nan=False)
            carrier = F.linear(F.linear(inputs, layer.right), layer.left)
            self.assertAlmostEqual(stats['term1_rms_over_input'], rms(layer.step_size(inputs)*carrier)/rms(inputs), places=14)
            self.assertAlmostEqual(stats['output_rms_over_input'], rms(layer(inputs))/rms(inputs), places=14)
            for name, value in state.items():torch.testing.assert_close(layer.state_dict()[name], value, atol=0, rtol=0)
        whole = VelocityNet(2, 'ode_lowrank', width=5, depth=2, gate_hidden=4, gate_reference_hidden=4,
                            orders=1, lowrank_state_width=10, lowrank_rank=2, lowrank_gate_hidden=2)
        entries = collect(whole, torch.randn(4, 2), torch.full((4, 1), .5)); json.dumps(entries, allow_nan=False)
        self.assertEqual(len(entries), 4)
        self.assertTrue(all(row['operator_type'] == 'pruned_lowrank_K1' for row in entries))
        valid = dict(orders=1, gate_hidden=32, lowrank_state_width=128, lowrank_rank=21, lowrank_gate_hidden=21)
        for change in (dict(orders=2), dict(lowrank_rank=0), dict(lowrank_rank=65), dict(lowrank_state_width=96),
                       dict(lowrank_gate_hidden=33), dict(lowrank_rank=2.5), dict(lowrank_gate_hidden=None),
                       dict(ode_solve_init='constant_paired'), dict(ode_gate_transform='affine'), dict(ode_forcing_mode='conditional'),
                       dict(ode_feature_mode='shared_projection'), dict(ode_operator_mode='additive'), dict(spectral_cap=.5), dict(tau=1.)):
            with self.subTest(change=change), self.assertRaises(ValueError):VelocityNet(2, 'ode_lowrank', **dict(valid, **change))
        with self.assertRaises(ValueError):VelocityNet(2, 'ode_vector', **valid)
        for change in (dict(gate_transform='affine'), dict(tau=1.), dict(scale=1.), dict(shift=0.)):
            reference = VectorODELayer(5, 4, 4, orders=1, **change)
            with self.subTest(reference=change), self.assertRaises(ValueError):
                PrunedLowRankLayer(reference, width=5, state_width=10, rank=2, gate_hidden=2, role='lift')
        reference = VectorODELayer(5, 4, 4, orders=1)
        with self.assertRaises(ValueError):PrunedLowRankLayer(reference, width=4, state_width=8, rank=2, gate_hidden=2, role='lift')
        with self.assertRaises(ValueError):PrunedLowRankLayer(OriginalODELayer(5, 4, orders=1), width=5, state_width=10, rank=2, gate_hidden=2, role='lift')


if __name__ == '__main__':
    unittest.main()
