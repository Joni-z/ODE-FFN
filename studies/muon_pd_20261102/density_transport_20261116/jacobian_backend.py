"""Exact batched VJPs; the frozen observer and native sampler stay unchanged."""
from contextlib import contextmanager
import density_ops

torch = density_ops.torch


def batched_jacobian(model, x, t):
    """Vectorize the same coordinate VJPs over sample-separable model outputs."""
    with torch.enable_grad():
        state = x.detach().requires_grad_(True)
        value = model(state, t.detach())
        dim = x.shape[1]
        basis = torch.eye(dim, dtype=x.dtype, device=x.device)[:, None, :].expand(dim, len(x), dim)
        result = torch.autograd.grad(value, state, grad_outputs=basis, is_grads_batched=True)[0]
    return value.detach(), result.permute(1, 0, 2).detach()


@contextmanager
def derivative_backend(name):
    """Process-local injection; workers are processes, never concurrent threads."""
    assert name in ['coordinate', 'batched']
    if name == 'coordinate':
        yield
        return
    original = density_ops.exact_jacobian
    density_ops.exact_jacobian = batched_jacobian
    try:
        yield
    finally:
        density_ops.exact_jacobian = original
