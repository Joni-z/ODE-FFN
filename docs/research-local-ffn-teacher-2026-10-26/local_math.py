"""Independent closed local maps and input Jacobians, for verification only."""
import torch
from torch.nn import functional as F


def silu_prime(x: torch.Tensor) -> torch.Tensor:
    sigmoid = torch.sigmoid(x)
    return sigmoid + x * sigmoid * (1 - sigmoid)


def solve(layer: torch.nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    p = layer.proj.weight
    pre = F.linear(x, layer.t_func[0].weight, layer.t_func[0].bias)
    logit = F.linear(F.silu(pre), layer.t_func[-1].weight, layer.t_func[-1].bias)
    sigmoid = torch.sigmoid(logit / layer.tau)
    a = sigmoid * layer.scale + layer.shift
    grad_a = (silu_prime(pre) * layer.t_func[-1].weight) @ layer.t_func[0].weight
    grad_a = grad_a * (layer.scale / layer.tau) * sigmoid * (1 - sigmoid)
    px = x @ p.T
    p2x = px @ p.T
    value = x + a * px + .5 * a.square() * p2x
    eye = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
    q = eye + a[:, :, None] * p + .5 * a.square()[:, :, None] * (p @ p)
    jacobian = q + (px + a * p2x)[:, :, None] * grad_a[:, None, :]
    return value, jacobian


def value_jacobian(module: torch.nn.Module, x: torch.Tensor, family: str) -> tuple[torch.Tensor, torch.Tensor]:
    if family == 'ode_k2':
        first, j1 = solve(module.ode1, x)
        value, j2 = solve(module.ode2, F.silu(first))
        return value, (j2 * silu_prime(first)[:, None, :]) @ j1
    gate, value = F.linear(x, module.w12.weight, module.w12.bias).chunk(2, -1)
    gate_weight, value_weight = module.w12.weight.chunk(2, 0)
    hidden_j = (value * silu_prime(gate))[:, :, None] * gate_weight
    hidden_j = hidden_j + F.silu(gate)[:, :, None] * value_weight
    result = F.linear(F.silu(gate) * value, module.w3.weight, module.w3.bias)
    return result, torch.einsum('oi,bid->bod', module.w3.weight, hidden_j)
