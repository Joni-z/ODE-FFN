"""Check the actual corrector algebra, gradients, costs and old-control identity."""
import copy
import unittest

import torch
from torch import nn

from synthetic.anchored_corrector import build
from synthetic.recurrent_depth import build as old_build, compute_budget


def nonzero_head(model):
    with torch.no_grad():
        model.raw.output.weight.copy_(torch.linspace(-.2, .2, model.raw.output.weight.numel(),
            dtype=model.raw.output.weight.dtype).reshape_as(model.raw.output.weight))


class CorrectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_accumulating_control_is_bitwise_identical_after_optimizer_updates(self):
        for family in ("ode_k2", "swiglu192", "swiglu128"):
            new, old = build(16, family, "accumulating"), old_build(16, family, "shared")
            optimizers = [torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.) for m in (new, old)]
            rng = torch.Generator().manual_seed(0)
            for _ in range(5):
                x, target = [torch.randn(8, 16, generator=rng) for _ in range(2)]
                t = torch.rand(8, 1, generator=rng)
                losses = []
                for model, optimizer in zip((new, old), optimizers):
                    optimizer.zero_grad(set_to_none=True)
                    loss = (model(x, t)-target).square().mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    losses.append(loss)
                self.assertTrue(torch.equal(*losses))
                for key, value in new.state_dict().items():
                    self.assertTrue(torch.equal(value, old.state_dict()[key]), key)

    def test_one_update_and_complete_gradients_match_for_both_rules(self):
        for family in ("ode_k2", "swiglu192", "swiglu128"):
            models = [build(2, family, mode).double() for mode in ("accumulating", "anchored")]
            for model in models:
                nonzero_head(model)
            x = torch.tensor([[.2, -.5], [.6, .8]], dtype=torch.float64, requires_grad=True)
            t = torch.tensor([[.25], [.75]], dtype=torch.float64)
            values = [model(x, t, updates=1) for model in models]
            self.assertTrue(torch.equal(*values))
            jacobians = [torch.autograd.functional.jacobian(lambda z: model(z, t, updates=1), x)
                         for model in models]
            torch.testing.assert_close(*jacobians, rtol=0, atol=0)
            for value in values:
                value.square().mean().backward()
            for a, b in zip(models[0].parameters(), models[1].parameters()):
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    def test_linear_actual_forward_matches_geometric_polynomial_and_resolvent(self):
        x = torch.tensor([[.2, -.5]], dtype=torch.float64)
        t = torch.tensor([[.5]], dtype=torch.float64)
        model = build(2, "swiglu128", "anchored").double()
        block = model.raw.blocks[0]
        block.norm = nn.Identity()
        block.ffn = nn.Linear(64, 64, bias=False, dtype=torch.float64)
        with torch.no_grad():
            block.condition.weight.zero_()
            block.condition.bias.zero_()
            values = torch.linspace(-.3, .3, 64, dtype=torch.float64)
            block.ffn.weight.copy_(torch.diag(values)/model.default_step)
        a = model.default_step*block.ffn.weight.detach()
        h0 = model.raw.input(x)
        states = model.hidden_states(x, t, updates=12)
        identity = torch.eye(64, dtype=torch.float64)
        polynomial, power = identity.clone(), identity.clone()
        for count in range(1, 13):
            power = power@a
            polynomial = polynomial+power
            torch.testing.assert_close(states[count], h0@polynomial.T, rtol=2e-14, atol=2e-14)
        exact = torch.linalg.solve(identity-a, h0.T).T
        q = float(torch.linalg.matrix_norm(a, ord=float("inf")))
        tail_bound = float(h0.detach().abs().max())*q**13/(1-q)
        error = float((states[12]-exact).detach().abs().max())
        self.assertLessEqual(error, tail_bound+1e-14)
        tail = torch.linalg.solve(identity-a, (h0@torch.linalg.matrix_power(a, 13).T).T).T
        torch.testing.assert_close(exact-states[12], tail, rtol=1e-8, atol=1e-14)
        model.update_rule = "accumulating"
        expected = h0@torch.linalg.matrix_power(identity+a, 3).T
        torch.testing.assert_close(model.hidden_states(x, t)[-1], expected, rtol=2e-14, atol=2e-14)

    def test_anchor_gradients_include_input_path_and_pass_finite_difference(self):
        model = build(2, "ode_k2", "anchored").double()
        nonzero_head(model)
        x = torch.tensor([[.2, -.5]], dtype=torch.float64, requires_grad=True)
        t = torch.tensor([[.4]], dtype=torch.float64)
        self.assertTrue(torch.autograd.gradcheck(lambda state: model(state, t), (x,), fast_mode=True))

    def test_rule_does_not_change_parameters_or_executed_matrix_calls(self):
        for family in ("ode_k2", "swiglu192", "swiglu128"):
            a, b = [build(16, family, mode) for mode in ("accumulating", "anchored")]
            self.assertEqual(compute_budget(a), compute_budget(b))
            for key, value in a.state_dict().items():
                self.assertTrue(torch.equal(value, b.state_dict()[key]))
            visits = []
            hooks = [layer.register_forward_hook(lambda *args: visits.append(1)) for _, layer in b.ode_layers()]
            try:
                b(torch.zeros(1, 16), torch.zeros(1, 1))
            finally:
                for hook in hooks:
                    hook.remove()
            self.assertEqual(len(visits), 6 if family=="ode_k2" else 0)

    def test_checkpoint_and_repeated_iteration_validation(self):
        model = build(2, "ode_k2", "anchored")
        nonzero_head(model)
        restored = build(2, "ode_k2", "anchored")
        restored.load_state_dict(copy.deepcopy(model.state_dict()))
        x, t = torch.zeros(2, 2), torch.full((2, 1), .5)
        self.assertTrue(torch.equal(model(x, t, updates=6), restored(x, t, updates=6)))
        for options in (dict(updates=0), dict(updates=True), dict(step_size=.1)):
            with self.assertRaises(ValueError):
                model(x, t, **options)


if __name__ == "__main__":
    unittest.main()
