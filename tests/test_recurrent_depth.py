"""Function, gradient, integration and cost controls for shared hidden evolution."""
import copy
import math
import unittest

import torch

from synthetic.recurrent_depth import build, training_loss, compute_budget


def nonzero_head(model):
    with torch.no_grad():
        values = torch.linspace(-.2, .2, model.raw.output.weight.numel(),
                                dtype=model.raw.output.weight.dtype)
        model.raw.output.weight.copy_(values.reshape_as(model.raw.output.weight))


class RecurrentDepthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_native_forward_matches_original_backbone(self):
        for family in ("ode_k2", "swiglu128", "swiglu192"):
            model = build(2, family, "native").double()
            nonzero_head(model)
            x = torch.linspace(-1, 1, 14, dtype=torch.float64).reshape(7, 2)
            t = torch.linspace(.1, .9, 7, dtype=torch.float64)[:, None]
            self.assertTrue(torch.equal(model(x, t), model.raw(x, t)))

    def test_cloned_shared_functions_and_complete_input_derivatives_match(self):
        for family in ("ode_k2", "swiglu192"):
            cloned, shared = [build(2, family, mode).double() for mode in ("cloned", "shared")]
            for model in (cloned, shared):
                nonzero_head(model)
            x = torch.tensor([[.2, -.6], [1., .4]], dtype=torch.float64, requires_grad=True)
            t = torch.tensor([[.25], [.75]], dtype=torch.float64)
            a, b = cloned(x, t), shared(x, t)
            self.assertTrue(torch.equal(a, b))
            ja = torch.autograd.functional.jacobian(lambda z: cloned(z, t), x)
            jb = torch.autograd.functional.jacobian(lambda z: shared(z, t), x)
            torch.testing.assert_close(ja, jb, rtol=1e-13, atol=1e-13)

    def test_tied_gradient_equals_sum_of_independent_block_gradients(self):
        for family in ("ode_k2", "swiglu192"):
            for supervision in ("final", "deep"):
                cloned, shared = [build(2, family, mode).double() for mode in ("cloned", "shared")]
                for model in (cloned, shared):
                    nonzero_head(model)
                x = torch.tensor([[.2, -.6], [1., .4]], dtype=torch.float64)
                t = torch.tensor([[.25], [.75]], dtype=torch.float64)
                target = torch.tensor([[.8, -.5], [-.2, .7]], dtype=torch.float64)
                losses = [training_loss(model, x, t, target, supervision) for model in (cloned, shared)]
                torch.testing.assert_close(*losses, rtol=1e-13, atol=1e-13)
                for loss in losses:
                    loss.backward()
                reference = [dict(block.named_parameters()) for block in cloned.raw.blocks]
                for name, parameter in shared.raw.blocks[0].named_parameters():
                    expected = sum(block[name].grad for block in reference)
                    torch.testing.assert_close(parameter.grad, expected, rtol=2e-11, atol=2e-13)

    def test_deep_supervision_loss_uses_normalized_declared_weights(self):
        model = build(2, "ode_k2", "shared").double()
        nonzero_head(model)
        x = torch.tensor([[.2, -.6], [1., .4]], dtype=torch.float64)
        t = torch.full((2, 1), .5, dtype=torch.float64)
        target = -x
        predictions = model(x, t, return_all=True)
        expected = sum(weight*(value-target).square().mean()
                       for weight, value in zip((.25, .25, .5), predictions))
        torch.testing.assert_close(training_loss(model, x, t, target, "deep"), expected)
        torch.testing.assert_close(predictions[-1], model(x, t))

    def test_actual_executed_cost_does_not_shrink_with_parameter_sharing(self):
        for family in ("ode_k2", "swiglu192"):
            native = compute_budget(build(16, family, "native"))
            shared_model = build(16, family, "shared")
            shared = compute_budget(shared_model)
            deep = compute_budget(shared_model, supervision="deep")
            self.assertEqual(native["dense_forward_macs"], shared["dense_forward_macs"])
            self.assertLess(shared["parameters"], .6*native["parameters"])
            self.assertEqual(deep["dense_forward_macs"]-shared["dense_forward_macs"], 2*64*16)
            six = compute_budget(shared_model, updates=6)
            self.assertGreater(six["dense_forward_macs"], shared["dense_forward_macs"])
        ode = compute_budget(build(16, "ode_k2", "shared"))
        swiglu = compute_budget(build(16, "swiglu192", "shared"))
        self.assertLess(abs(ode["dense_forward_macs"]/swiglu["dense_forward_macs"]-1), .02)

    def test_fixed_horizon_grid_refinement_is_first_order_for_the_same_field(self):
        model = build(2, "ode_k2", "shared").double()
        nonzero_head(model)
        x = torch.tensor([[.2, -.6], [1., .4]], dtype=torch.float64)
        t = torch.full((2, 1), .5, dtype=torch.float64)
        horizon = .05
        with torch.no_grad():
            reference = model(x, t, updates=512, step_size=horizon/512)
            errors = [float((model(x, t, updates=count, step_size=horizon/count)-reference).norm())
                      for count in (8, 16, 32)]
        self.assertGreater(errors[0]/errors[1], 1.8)
        self.assertGreater(errors[1]/errors[2], 1.8)
        self.assertLess(errors[0]/errors[1], 2.2)

    def test_actual_shared_parameter_aliases_survive_checkpoint_reload(self):
        model = build(2, "ode_k2", "shared")
        state = copy.deepcopy(model.state_dict())
        restored = build(2, "ode_k2", "shared")
        restored.load_state_dict(state)
        self.assertEqual(len(restored.raw.blocks), 1)
        self.assertEqual(len(list(restored.ode_layers())), 2)
        x, t = torch.zeros(3, 2), torch.full((3, 1), .5)
        self.assertTrue(torch.equal(model(x, t), restored(x, t)))

    def test_invalid_independent_rollout_is_rejected(self):
        model = build(2, "swiglu192", "native")
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 2), torch.zeros(1, 1), updates=6)


if __name__ == "__main__":
    unittest.main()
