import unittest

import numpy as np
import torch

from scripts.diagnose_projection_geometry import capture, moments, moment_stats, responses
from synthetic.models import VelocityNet


class ProjectionGeometryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def test_fixed_time_first_input_rank_bound_and_time_pooling(self):
        for dim in (2, 16):
            for kind in ('ode_vector', 'swiglu'):
                with self.subTest(dim=dim, kind=kind):
                    model = VelocityNet(dim, kind, width=32, depth=2, orders=1,
                        gate_hidden=12, gate_reference_hidden=32).double()
                    # Exercise arbitrary learned affine scales/biases, not just init ones.
                    with torch.no_grad():
                        model.blocks[0].norm.weight.normal_(1., .3)
                        model.blocks[0].norm.bias.normal_(0., .2)
                        model.blocks[0].condition.weight.normal_(0., .2)
                        model.blocks[0].condition.bias.normal_(0., .2)
                    x = torch.randn(1024, dim, dtype=torch.float64)
                    name = 'blocks.0.ffn.' + ('ode1.P' if kind == 'ode_vector' else 'w12')
                    fixed = moments(model, x, torch.full((len(x), 1), .3, dtype=torch.float64))
                    rank = moment_stats(fixed[name]['matrix'])['rank']
                    self.assertLessEqual(rank, dim + 2)
                    pooled = moments(model, x, torch.rand(len(x), 1, dtype=torch.float64))
                    if dim == 2:
                        self.assertGreater(moment_stats(pooled[name]['matrix'])['rank'], dim + 2)

    def test_complete_network_projection_response_matches_perturbations(self):
        for kind in ('ode_vector', 'swiglu'):
            with self.subTest(kind=kind):
                model = VelocityNet(3, kind, width=12, depth=2, orders=1,
                    gate_hidden=6, gate_reference_hidden=12, ffn_hidden=18).double()
                with torch.no_grad():
                    model.output.weight.normal_(0., .2)
                x = torch.randn(11, 3, dtype=torch.float64)
                t = torch.rand(11, 1, dtype=torch.float64)
                target = torch.randn_like(x)
                measured, _, error = responses(model, x, t, {'pair': target})
                self.assertLess(error, 1e-12)
                _, nodes = capture(model, x, t)
                for name, node in nodes.items():
                    weight = node['parameter']
                    direction = torch.randn_like(weight)
                    h = measured[name]['input']
                    response = measured[name]['response']
                    predicted = np.einsum('bdi,bi->bd', response, h @ direction.numpy().T)
                    original = weight.detach().clone()
                    epsilon = 2e-5
                    with torch.no_grad():
                        weight.copy_(original + epsilon * direction)
                        positive = model(x, t).numpy()
                        weight.copy_(original - epsilon * direction)
                        negative = model(x, t).numpy()
                        weight.copy_(original)
                    np.testing.assert_allclose((positive-negative)/(2*epsilon), predicted,
                                                atol=2e-8, rtol=2e-7, err_msg=name)

    def test_zero_head_has_zero_backbone_response(self):
        model = VelocityNet(2, 'ode_vector', width=12, depth=1, orders=1,
                            gate_hidden=6, gate_reference_hidden=12).double()
        x, t = torch.randn(7, 2, dtype=torch.float64), torch.rand(7, 1, dtype=torch.float64)
        measured, _, error = responses(model, x, t, {'pair': torch.randn_like(x)})
        self.assertEqual(error, 0.)
        for node in measured.values():
            self.assertEqual(np.count_nonzero(node['response']), 0)
            self.assertEqual(np.count_nonzero(node['gradients']['pair']), 0)


if __name__ == '__main__':
    unittest.main()
