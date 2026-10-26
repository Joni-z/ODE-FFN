import unittest

import numpy as np
import torch

from synthetic.covariance import CovarianceField


class CovarianceFieldTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def test_field_is_potential_gradient_and_jacobian_is_covariance(self):
        for mode in ('tied', 'linearized'):
            field = CovarianceField(5, heads=2, atoms=4, temperature=.7, mode=mode).double()
            field.gamma.data.fill_(-.4)
            x = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
            gradient = torch.autograd.grad(field.potential(x).sum(), x, create_graph=True)[0]
            torch.testing.assert_close(gradient, field(x), atol=1e-14, rtol=1e-13)
            jacobian = torch.func.vmap(torch.func.jacrev(field))(x)
            exact = field.field_jacobian(x)
            torch.testing.assert_close(jacobian, exact, atol=1e-14, rtol=1e-13)
            torch.testing.assert_close(exact, exact.transpose(-2, -1), atol=1e-14, rtol=1e-13)
            eigen = torch.linalg.eigvalsh(exact)
            upper = field.gamma + torch.linalg.matrix_norm(field.weight, ord=2).square() / (2 * field.temperature)
            self.assertGreaterEqual(float(eigen.min().detach()), float(field.gamma.detach()) - 1e-13)
            self.assertLessEqual(float(eigen.max().detach()), float(upper.detach()) + 1e-13)

    def test_untied_starts_with_same_function_rng_and_breaks_structure(self):
        torch.manual_seed(0)
        tied = CovarianceField(5, heads=2, atoms=4).double()
        rng = torch.get_rng_state()
        torch.manual_seed(0)
        untied = CovarianceField(5, heads=2, atoms=4, mode='untied').double()
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
        x = torch.randn(7, 5, dtype=torch.float64)
        torch.testing.assert_close(untied(x), tied(x), atol=0, rtol=0)
        with torch.no_grad():
            untied.decoder[0, 0] += .8
        actual = torch.func.vmap(torch.func.jacrev(untied))(x)
        torch.testing.assert_close(actual, untied.field_jacobian(x), atol=1e-14, rtol=1e-13)
        self.assertGreater(float((actual - actual.transpose(-2, -1)).detach().abs().max()), .001)

    def test_linearized_control_has_exact_origin_tangent(self):
        tied = CovarianceField(5, heads=2, atoms=4).double()
        control = CovarianceField(5, heads=2, atoms=4, mode='linearized').double()
        control.load_state_dict(tied.state_dict())
        x = torch.zeros(5, dtype=torch.float64)
        torch.testing.assert_close(control(x), tied(x), atol=0, rtol=0)
        torch.testing.assert_close(torch.func.jacrev(control)(x), torch.func.jacrev(tied)(x), atol=1e-14, rtol=1e-13)
        z = torch.randn(5, dtype=torch.float64)
        torch.testing.assert_close(control.field_jacobian(z), control.field_jacobian(x), atol=0, rtol=0)
        self.assertGreater(float((tied.field_jacobian(8 * z) - tied.field_jacobian(x)).detach().abs().max()), .001)

    def test_positive_frozen_gate_preserves_real_generator_spectrum(self):
        field = CovarianceField(5, heads=2, atoms=4).double()
        field.gamma.data.fill_(-.4)
        hessian = field.field_jacobian(torch.randn(5, dtype=torch.float64))
        gate = torch.rand(5, dtype=torch.float64) + .2
        actual = torch.linalg.eigvals(gate[:, None] * hessian)
        symmetric = gate.sqrt()[:, None] * hessian * gate.sqrt()[None, :]
        self.assertLess(float(actual.imag.detach().abs().max()), 1e-13)
        torch.testing.assert_close(actual.real.sort().values, torch.linalg.eigvalsh(symmetric), atol=1e-13, rtol=1e-12)

    def test_independent_gaussian_empirical_target_has_symmetric_jacobian(self):
        # Independent finite target atoms, fresh isotropic Gaussian source.
        # Unequal prior weights/atom norms prevent a symmetry-only special case.
        atoms = torch.tensor([[2., -.7], [-1., .3], [.2, 1.4]], dtype=torch.float64)
        prior = torch.tensor([.2, .5, .3], dtype=torch.float64)
        x = torch.tensor([.35, -.15], dtype=torch.float64)
        for time in (.1, .5, .9):
            a = 1 - time
            def velocity(z):
                log_weight = prior.log() - (z[None] - time * atoms).square().sum(dim=1) / (2 * a * a)
                p = torch.softmax(log_weight, dim=0)
                return (p @ atoms - z) / a
            probabilities = torch.softmax(prior.log() - (x[None] - time * atoms).square().sum(dim=1) / (2 * a * a), dim=0)
            mean = probabilities @ atoms
            delta = atoms - mean
            covariance = (delta.t() * probabilities) @ delta
            predicted = -torch.eye(2, dtype=x.dtype) / a + time * covariance / a ** 3
            actual = torch.func.jacrev(velocity)(x)
            torch.testing.assert_close(actual, predicted, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(actual.numpy(), actual.numpy().T, atol=1e-12)

    def test_bregman_energy_identity_and_descent_failure_boundary(self):
        field = CovarianceField(5, heads=2, atoms=4).double()
        field.gamma.data.fill_(-.7)
        x = torch.randn(17, 5, dtype=torch.float64)
        # An explicitly input-dependent positive diagonal gate.
        duration = .2 + .8 * torch.sigmoid(3 * x)
        drift = field(x)
        increment = duration * drift
        updated = x + increment
        convex = lambda z: field.potential(z) - .5 * field.gamma * z.square().sum(dim=-1)
        bregman = convex(updated) - convex(x) - ((drift - field.gamma * x) * increment).sum(dim=-1)
        change = field.potential(x) - field.potential(updated)
        exact = -(drift * increment).sum(dim=-1) - .5 * field.gamma * increment.square().sum(dim=-1) - bregman
        torch.testing.assert_close(change, exact, atol=1e-13, rtol=1e-12)
        self.assertGreaterEqual(float(bregman.min().detach()), -1e-13)
        margin = 1 - (-field.gamma) * duration.max(dim=-1).values / 2
        bound = -margin * (duration * drift.square()).sum(dim=-1)
        self.assertTrue(torch.all(change <= bound + 1e-12))
        self.assertTrue(torch.all(change < 0))
        with torch.no_grad():
            field.weight.zero_()
            field.gamma.fill_(-4)
        updated = x + .8 * field(x)
        change = field.potential(x) - field.potential(updated)
        self.assertTrue(torch.all(change > 0))


if __name__ == '__main__':
    unittest.main()
