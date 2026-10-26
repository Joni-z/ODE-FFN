"""Loss, gradient, initialization and actual compute controls for the bridge."""
import torch
from torch import nn
import unittest

from scripts.run_prediction_bridge import PredictionBridge, target_velocity, build, budget
from synthetic.models import OriginalODELayer


class FreePrediction(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = nn.Parameter(value.clone())

    def forward(self, x, t):
        return self.value


def check_image_loss_and_parameter_gradient_equal_weighted_clean_loss(time):
    target = torch.tensor([[1., -2.], [.5, 3.]], dtype=torch.float64)
    source = torch.tensor([[.3, .8], [-.2, -.7]], dtype=torch.float64)
    t = torch.full((2, 1), time, dtype=torch.float64)
    state = t*target+(1-t)*source
    raw = FreePrediction(torch.tensor([[.2, -.1], [2., 1.]], dtype=torch.float64))
    model = PredictionBridge(raw, "x")
    actual = (model(state, t)-target_velocity(target, state, t)).square().mean()
    expected = ((raw.value-target)/(1-t).clamp_min(.05)).square().mean()
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    actual_gradient = torch.autograd.grad(actual, raw.value, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, raw.value)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)


def check_common_pair_target_includes_clipped_scale():
    target = torch.tensor([[1., 2.]], dtype=torch.float64)
    source = torch.tensor([[-1., 3.]], dtype=torch.float64)
    for time in [.5, .99, 1.]:
        t = torch.full((1, 1), time, dtype=torch.float64)
        state = t*target+(1-t)*source
        expected = (target-source)*(1-t)/(1-t).clamp_min(.05)
        torch.testing.assert_close(target_velocity(target, state, t), expected, atol=1e-12, rtol=1e-12)


def check_prediction_spaces_share_exact_raw_states_and_buffers(label):
    image, velocity = build(16, label, "x"), build(16, label, "v")
    for name, value in image.state_dict().items():
        assert torch.equal(value, velocity.state_dict()[name]), name
    for name, value in image.named_buffers():
        assert torch.equal(value, dict(velocity.named_buffers())[name]), name


def check_all_non_ffn_parameters_and_buffers_are_paired():
    reference = build(16, "ode_k2", "v")
    for label in ["ode_k1", "swiglu128", "swiglu192"]:
        other = build(16, label, "v")
        for getter in ["named_parameters", "named_buffers"]:
            values = dict(getattr(other, getter)())
            for name, value in getattr(reference, getter)():
                if ".ffn." not in name:
                    assert torch.equal(value, values[name]), (label, name)


def check_compute_counts_include_second_projection_call_and_both_baselines():
    k1, k2 = [budget(build(16, label, "v")) for label in ["ode_k1", "ode_k2"]]
    assert k1["parameters"] == k2["parameters"]
    assert k2["dense_forward_macs"]-k1["dense_forward_macs"] == 6*64*64
    small, matched = [budget(build(16, label, "v")) for label in ["swiglu128", "swiglu192"]]
    assert abs(k1["parameters"]/small["parameters"]-1) < .02
    assert abs(k2["dense_forward_macs"]/matched["dense_forward_macs"]-1) < .02


def check_original_operator_is_the_professor_frozen_scalar_recurrence(order):
    torch.manual_seed(0)
    layer = OriginalODELayer(8, 8, orders=order).double()
    x = torch.linspace(-1., 1., 24, dtype=torch.float64).reshape(3, 8)
    duration = layer.step_size(x)
    assert duration.shape == (3, 1)
    expected, term = x, x
    for index in range(1, order+1):
        term = (term@layer.proj.weight.T)*duration/index
        expected = expected+term
    torch.testing.assert_close(layer(x), expected, atol=1e-14, rtol=1e-14)


def check_image_conversion_jacobian_includes_direct_input_term():
    raw = FreePrediction(torch.tensor([[.2, .3]], dtype=torch.float64))
    image = PredictionBridge(raw, "x")
    state = torch.tensor([[1., 2.]], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([[.5]], dtype=torch.float64)
    jac = torch.autograd.functional.jacobian(lambda x: image(x, t), state)[0, :, 0, :]
    torch.testing.assert_close(jac, -2*torch.eye(2, dtype=torch.float64))


def check_three_updates_are_deterministic_with_identical_data_rng():
    torch.set_num_threads(1)
    outcomes = []
    for _ in range(2):
        model = build(2, "ode_k2", "x")
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.)
        generator = torch.Generator().manual_seed(100000)
        for step in range(3):
            target = torch.randn((16, 2), generator=generator)
            source = torch.randn((16, 2), generator=generator)
            t = torch.sigmoid(-.8+.8*torch.randn((16, 1), generator=generator))
            state = t*target+(1-t)*source
            optimizer.zero_grad(set_to_none=True)
            loss = (model(state, t)-target_velocity(target, state, t)).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        outcomes.append((model.state_dict(), generator.get_state()))
    for name, value in outcomes[0][0].items():
        assert torch.equal(value, outcomes[1][0][name]), name
    assert torch.equal(outcomes[0][1], outcomes[1][1])


class PredictionBridgeTests(unittest.TestCase):
    def test_loss_and_gradient(self):
        for time in [.2, .5, .98, 1.]:
            with self.subTest(time=time):
                check_image_loss_and_parameter_gradient_equal_weighted_clean_loss(time)

    def test_clipped_target(self):
        check_common_pair_target_includes_clipped_scale()

    def test_prediction_pairing(self):
        for label in ["ode_k1", "ode_k2", "swiglu128", "swiglu192"]:
            with self.subTest(label=label):
                check_prediction_spaces_share_exact_raw_states_and_buffers(label)

    def test_common_backbone(self):
        check_all_non_ffn_parameters_and_buffers_are_paired()

    def test_budget(self):
        check_compute_counts_include_second_projection_call_and_both_baselines()

    def test_professor_operator(self):
        for order in [1, 2]:
            with self.subTest(order=order):
                check_original_operator_is_the_professor_frozen_scalar_recurrence(order)

    def test_conversion_jacobian(self):
        check_image_conversion_jacobian_includes_direct_input_term()

    def test_deterministic_updates(self):
        check_three_updates_are_deterministic_with_identical_data_rng()


if __name__ == "__main__":
    unittest.main()
