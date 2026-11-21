"""Check that reducing the state preserves both output and all solve gradients."""
from reduced_trial import build_model, torch

torch.set_num_threads(1)
torch.manual_seed(0)
model = build_model(16, "reduced_midpoint4")
layer = next(iter(model.ode_layers()))[1].double()
x = torch.randn(7, 64, dtype=torch.float64, requires_grad=True)
encode, decode = layer.factors()


def drift(y):
    hidden = y @ encode.T
    return (hidden * layer.step_size(hidden)) @ decode.T


explicit = x
for _ in range(4):
    explicit = explicit + 0.25 * drift(explicit + 0.125 * drift(explicit))
reduced = layer(x)
torch.testing.assert_close(explicit, reduced, rtol=1e-12, atol=1e-12)
inputs = x, layer.proj.weight
for full, compact in zip(torch.autograd.grad(explicit.square().sum(), inputs),
                        torch.autograd.grad(reduced.square().sum(), inputs)):
    torch.testing.assert_close(full, compact, rtol=1e-11, atol=1e-11)
for family in ("state_midpoint4", "frozen_midpoint4", "state_midpoint2", "reduced_midpoint4"):
    torch.manual_seed(0)
    model = build_model(16, family)
    assert model.parameter_count() == 58512
    assert model(torch.zeros(2, 16), torch.ones(2, 1)).shape == (2, 16)
    print(family, model.parameter_count(), flush=True)
