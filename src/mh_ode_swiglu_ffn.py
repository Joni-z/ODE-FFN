from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from time_condition import build_time_mlp, resolve_time_scalar


def _linear_param_count(in_features: int, out_features: int, bias: bool) -> int:
    return in_features * out_features + (out_features if bias else 0)


def _time_mlp_param_count(in_features: int, hidden_features: int, bias: bool) -> int:
    return _linear_param_count(in_features, hidden_features, bias) + _linear_param_count(hidden_features, 1, bias)


def _round_down_to_multiple(value: int, divisor: int) -> int:
    return max(divisor, (value // divisor) * divisor)


def _baseline_swiglu_param_budget(dim: int, hidden_dim: int, bias: bool) -> tuple[int, int]:
    hidden_dim_eff = max(1, int(hidden_dim * 2 / 3))
    budget = _linear_param_count(dim, 2 * hidden_dim_eff, bias) + _linear_param_count(hidden_dim_eff, dim, bias)
    return hidden_dim_eff, budget


def _mh_ode_core_param_count(dim: int, hidden_dim_eff: int, num_heads: int, bias: bool) -> int:
    head_dim = hidden_dim_eff // num_heads
    dynamics_params = 2 * num_heads * head_dim * head_dim
    return (
        2 * _linear_param_count(dim, hidden_dim_eff, bias)
        + _linear_param_count(hidden_dim_eff, dim, bias)
        + dynamics_params
    )


def quota_aligned_hidden_dim(
    dim: int,
    hidden_dim: int,
    num_heads: int,
    bias: bool,
    time_hidden_dim: int,
    t_embed_dim: Optional[int],
) -> int:
    baseline_hidden_dim, budget = _baseline_swiglu_param_budget(dim, hidden_dim, bias)
    budget -= _time_mlp_param_count(dim, time_hidden_dim, bias)
    if t_embed_dim is not None:
        budget -= _time_mlp_param_count(t_embed_dim, time_hidden_dim, bias)

    max_hidden_dim = _round_down_to_multiple(baseline_hidden_dim, num_heads)
    for hidden_dim_eff in range(max_hidden_dim, num_heads - 1, -num_heads):
        if _mh_ode_core_param_count(dim, hidden_dim_eff, num_heads, bias) <= budget:
            return hidden_dim_eff
    return num_heads


class MultiHeadODELinear(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_heads: int = 8,
        bias: bool = True,
        orders: int = 1,
    ) -> None:
        super().__init__()
        if d_out % num_heads != 0:
            raise ValueError(f"d_out={d_out} must be divisible by num_heads={num_heads}")

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.orders = orders

        self.proj_in = nn.Linear(d_in, d_out, bias=bias)
        self.A = nn.Parameter(torch.randn(num_heads, self.head_dim, self.head_dim) * 0.02)

    def forward(self, x: Tensor, t_scalar: Tensor) -> Tensor:
        z = self.proj_in(x)
        batch_size, seq_len, _ = z.shape
        z = z.view(batch_size, seq_len, self.num_heads, self.head_dim)

        out = z
        term = z
        for order in range(1, self.orders + 1):
            term = torch.einsum("hij,bnhj->bnhi", self.A, term)
            term = term * (t_scalar / float(order))
            out = out + term

        return out.reshape(batch_size, seq_len, self.d_out)


class MultiHeadODESwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        drop: float = 0.0,
        bias: bool = True,
        tau: float = 10.0,
        scale: float = 0.8,
        shift: float = 0.2,
        orders: int = 1,
        t_embed_dim: Optional[int] = None,
        time_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")

        time_hidden_dim = time_hidden_dim or min(dim, 128)
        hidden_dim_eff = quota_aligned_hidden_dim(
            dim=dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            bias=bias,
            time_hidden_dim=time_hidden_dim,
            t_embed_dim=t_embed_dim,
        )

        self.tau = tau
        self.scale = scale
        self.shift = shift
        self.hidden_dim_eff = hidden_dim_eff

        self.t_func = build_time_mlp(dim, time_hidden_dim, bias=bias)
        self.t_from_embed = None
        if t_embed_dim is not None:
            self.t_from_embed = build_time_mlp(t_embed_dim, time_hidden_dim, bias=bias)

        self.ode_w1 = MultiHeadODELinear(dim, hidden_dim_eff, num_heads=num_heads, bias=bias, orders=orders)
        self.ode_w2 = MultiHeadODELinear(dim, hidden_dim_eff, num_heads=num_heads, bias=bias, orders=orders)
        self.w3 = nn.Linear(hidden_dim_eff, dim, bias=bias)
        self.drop = nn.Dropout(drop)

    def _time_scalar(self, x: Tensor, t: Optional[Tensor]) -> Tensor:
        t_scalar = resolve_time_scalar(x, t, self.t_func, self.t_from_embed)
        return torch.sigmoid(t_scalar / self.tau) * self.scale + self.shift

    def forward(self, x: Tensor, t: Optional[Tensor] = None) -> Tensor:
        t_scalar = self._time_scalar(x, t)
        gate = self.ode_w1(x, t_scalar)
        value = self.ode_w2(x, t_scalar)
        hidden = F.silu(gate) * value
        hidden = self.drop(hidden)
        return self.w3(hidden)


MHODESwiGLUFFN = MultiHeadODESwiGLUFFN
