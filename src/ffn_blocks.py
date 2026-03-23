from typing import Callable, Optional

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


def _search_hidden_dim(
    max_hidden_dim: int,
    multiple: int,
    budget: int,
    core_param_count: Callable[[int], int],
) -> int:
    for hidden_dim_eff in range(max_hidden_dim, multiple - 1, -multiple):
        if core_param_count(hidden_dim_eff) <= budget:
            return hidden_dim_eff
    return multiple


def _cond_param_count(cond_dim: Optional[int], out_dim: int, bias: bool) -> int:
    if cond_dim is None:
        return 0
    return _linear_param_count(cond_dim, out_dim, bias)


def _swiglu_hidden_arg_from_eff(hidden_dim_eff: int) -> int:
    return max(1, int(round(hidden_dim_eff * 3 / 2)))


class BaseFFN(nn.Module):
    def _condition_term(self, cond: Optional[Tensor], proj: Optional[nn.Module]) -> Optional[Tensor]:
        if cond is None or proj is None or not torch.is_tensor(cond):
            return None
        if cond.dim() < 2 or cond.shape[-1] != proj.in_features:
            return None
        cond_term = proj(cond)
        if cond_term.dim() == 2:
            cond_term = cond_term.unsqueeze(1)
        return cond_term

    def _aux(
        self,
        x: Tensor,
        out: Tensor,
        *,
        step: Optional[Tensor] = None,
        gate: Optional[Tensor] = None,
        alpha: Optional[Tensor] = None,
        beta: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        aux = {
            "input_norm": x.detach().norm(dim=-1).mean(),
            "output_norm": out.detach().norm(dim=-1).mean(),
        }
        if step is not None:
            aux["step_mean"] = step.detach().mean()
            aux["step_std"] = step.detach().std(unbiased=False)
        if gate is not None:
            aux["gate_mean"] = gate.detach().mean()
            aux["gate_std"] = gate.detach().std(unbiased=False)
        if alpha is not None:
            aux["alpha_mean"] = alpha.detach().mean()
            aux["alpha_std"] = alpha.detach().std(unbiased=False)
        if beta is not None:
            aux["beta_mean"] = beta.detach().mean()
            aux["beta_std"] = beta.detach().std(unbiased=False)
        return aux

    def _maybe_return(self, out: Tensor, aux: dict[str, Tensor], return_aux: bool):
        if return_aux:
            return out, aux
        return out


class SwiGLUFFN(BaseFFN):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0, bias: bool = True, **_: object) -> None:
        super().__init__()
        hidden_dim_eff = max(1, int(hidden_dim * 2 / 3))
        self.w12 = nn.Linear(dim, 2 * hidden_dim_eff, bias=bias)
        self.w3 = nn.Linear(hidden_dim_eff, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x: Tensor, cond: Optional[Tensor] = None, return_aux: bool = False):
        del cond
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        gate = F.silu(x1)
        hidden = gate * x2
        out = self.w3(self.ffn_dropout(hidden))
        return self._maybe_return(out, self._aux(x, out, gate=gate), return_aux)


class MLP(BaseFFN):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0, bias: bool = True, **_: object) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor, cond: Optional[Tensor] = None, return_aux: bool = False):
        del cond
        hidden = self.drop(self.act(self.fc1(x)))
        out = self.fc2(hidden)
        return self._maybe_return(out, self._aux(x, out), return_aux)


class ODELayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        bias: bool = True,
        tau: float = 10.0,
        scale: float = 0.8,
        shift: float = 0.2,
        orders: int = 2,
        t_embed_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        self.tau = tau
        self.scale = scale
        self.shift = shift
        self.orders = orders
        self.t_func = build_time_mlp(in_features, hidden_features, bias=bias)
        self.t_from_embed = None
        if t_embed_dim is not None:
            self.t_from_embed = build_time_mlp(t_embed_dim, hidden_features, bias=bias)

        self.proj = nn.Linear(in_features, in_features, bias=False)

    @torch.compile
    def forward(self, x: Tensor, t: Optional[Tensor] = None) -> Tensor:
        t_scalar = resolve_time_scalar(x, t, self.t_func, self.t_from_embed)
        t_scalar = torch.sigmoid(t_scalar / self.tau) * self.scale + self.shift

        approx = [x]
        for i in range(self.orders):
            term = (self.proj(approx[-1]) * t_scalar) / (i + 1)
            approx.append(term)
        return sum(approx)


class ODEFFN(BaseFFN):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop: float = 0.0,
        bias: bool = True,
        tau: float = 10.0,
        scale: float = 0.8,
        shift: float = 0.2,
        orders: int = 2,
        ode_hidden_features: Optional[int] = None,
        t_embed_dim: Optional[int] = None,
        **_: object,
    ) -> None:
        super().__init__()
        del hidden_dim
        ode_hidden_features = ode_hidden_features or dim
        self.ode1 = ODELayer(
            in_features=dim,
            hidden_features=ode_hidden_features,
            bias=bias,
            tau=tau,
            scale=scale,
            shift=shift,
            orders=orders,
            t_embed_dim=t_embed_dim,
        )
        self.act = nn.SiLU()
        self.ode2 = ODELayer(
            in_features=dim,
            hidden_features=ode_hidden_features,
            bias=bias,
            tau=tau,
            scale=scale,
            shift=shift,
            orders=orders,
            t_embed_dim=t_embed_dim,
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor, cond: Optional[Tensor] = None, return_aux: bool = False):
        x1 = self.ode1(x, cond)
        x2 = self.act(x1)
        out = self.drop(self.ode2(x2, cond))
        return self._maybe_return(out, self._aux(x, out), return_aux)


class ODESwiGLUFFN(BaseFFN):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop: float = 0.0,
        bias: bool = True,
        tau: float = 10.0,
        scale: float = 0.8,
        shift: float = 0.2,
        orders: int = 1,
        ode_hidden_features: Optional[int] = None,
        t_embed_dim: Optional[int] = None,
        delta_normalize: bool = True,
        gate_init_logit: float = 0.0,
        **_: object,
    ) -> None:
        super().__init__()
        ode_hidden_features = ode_hidden_features or dim
        self.ode = ODELayer(
            in_features=dim,
            hidden_features=ode_hidden_features,
            bias=bias,
            tau=tau,
            scale=scale,
            shift=shift,
            orders=orders,
            t_embed_dim=t_embed_dim,
        )
        self.ode_gate_logit = nn.Parameter(torch.tensor(gate_init_logit))
        self.delta_normalize = delta_normalize

        hidden_dim_eff = max(1, int(hidden_dim * 2 / 3))
        self.w12 = nn.Linear(dim, 2 * hidden_dim_eff, bias=bias)
        self.w3 = nn.Linear(hidden_dim_eff, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def _swiglu(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        gate = F.silu(x1)
        hidden = self.ffn_dropout(gate * x2)
        return self.w3(hidden), gate

    def forward(self, x: Tensor, cond: Optional[Tensor] = None, return_aux: bool = False):
        base, swiglu_gate = self._swiglu(x)
        delta = self.ode(x, cond) - x

        if self.delta_normalize:
            x_rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
            d_rms = torch.sqrt(torch.mean(delta * delta, dim=-1, keepdim=True) + 1e-6)
            delta = delta * (x_rms / d_rms.clamp_min(1e-6))

        ode_gate = torch.sigmoid(self.ode_gate_logit)
        out = base + ode_gate * delta
        aux = self._aux(x, out, gate=swiglu_gate)
        aux["ode_gate"] = ode_gate.detach()
        return self._maybe_return(out, aux, return_aux)


class ODEOnlyFFN(BaseFFN):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop: float = 0.0,
        bias: bool = True,
        tau: float = 10.0,
        scale: float = 0.8,
        shift: float = 0.2,
        orders: int = 1,
        ode_hidden_features: Optional[int] = None,
        t_embed_dim: Optional[int] = None,
        delta_normalize: bool = True,
        gate_init_logit: float = 0.0,
        **_: object,
    ) -> None:
        super().__init__()
        del hidden_dim
        ode_hidden_features = ode_hidden_features or dim
        self.ode = ODELayer(
            in_features=dim,
            hidden_features=ode_hidden_features,
            bias=bias,
            tau=tau,
            scale=scale,
            shift=shift,
            orders=orders,
            t_embed_dim=t_embed_dim,
        )
        self.ode_gate_logit = nn.Parameter(torch.tensor(gate_init_logit))
        self.delta_normalize = delta_normalize
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x: Tensor, cond: Optional[Tensor] = None, return_aux: bool = False):
        delta = self.ode(x, cond) - x
        if self.delta_normalize:
            x_rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
            d_rms = torch.sqrt(torch.mean(delta * delta, dim=-1, keepdim=True) + 1e-6)
            delta = delta * (x_rms / d_rms.clamp_min(1e-6))
        ode_gate = torch.sigmoid(self.ode_gate_logit)
        out = self.ffn_dropout(ode_gate * delta)
        aux = self._aux(x, out)
        aux["ode_gate"] = ode_gate.detach()
        return self._maybe_return(out, aux, return_aux)


def _reshape_headwise_step(t_scalar: Tensor, batch_size: int, seq_len: int) -> Tensor:
    if t_scalar.dim() == 1:
        return t_scalar.view(batch_size, 1, 1, 1)
    if t_scalar.dim() == 2:
        if t_scalar.shape[0] != batch_size:
            raise ValueError(f"Expected step batch {batch_size}, got shape {tuple(t_scalar.shape)}")
        if t_scalar.shape[1] == seq_len:
            return t_scalar.view(batch_size, seq_len, 1, 1)
        return t_scalar.view(batch_size, 1, t_scalar.shape[1], 1)
    if t_scalar.dim() == 3:
        if t_scalar.shape[0] != batch_size:
            raise ValueError(f"Expected step batch {batch_size}, got shape {tuple(t_scalar.shape)}")
        if t_scalar.shape[1] not in (1, seq_len):
            raise ValueError(f"Expected step sequence length 1 or {seq_len}, got shape {tuple(t_scalar.shape)}")
        return t_scalar.unsqueeze(-1)
    return t_scalar.view(batch_size, 1, 1, 1)


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
        t_scalar = _reshape_headwise_step(t_scalar, batch_size, seq_len)

        out = z
        term = z
        for order in range(1, self.orders + 1):
            term = torch.einsum("hij,bnhj->bnhi", self.A, term)
            term = term * (t_scalar / float(order))
            out = out + term

        return out.reshape(batch_size, seq_len, self.d_out)


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

    def _core(hidden_dim_eff: int) -> int:
        head_dim = hidden_dim_eff // num_heads
        dynamics_params = 2 * num_heads * head_dim * head_dim
        return (
            2 * _linear_param_count(dim, hidden_dim_eff, bias)
            + _linear_param_count(hidden_dim_eff, dim, bias)
            + dynamics_params
        )

    max_hidden_dim = _round_down_to_multiple(baseline_hidden_dim, num_heads)
    return _search_hidden_dim(max_hidden_dim, num_heads, budget, _core)


class MultiHeadODESwiGLUFFN(BaseFFN):
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
        **_: object,
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

        self.t_func = build_time_mlp(dim, time_hidden_dim, bias=bias)
        self.t_from_embed = None
        if t_embed_dim is not None:
            self.t_from_embed = build_time_mlp(t_embed_dim, time_hidden_dim, bias=bias)

        self.ode_w1 = MultiHeadODELinear(dim, hidden_dim_eff, num_heads=num_heads, bias=bias, orders=orders)
        self.ode_w2 = MultiHeadODELinear(dim, hidden_dim_eff, num_heads=num_heads, bias=bias, orders=orders)
        self.w3 = nn.Linear(hidden_dim_eff, dim, bias=bias)
        self.drop = nn.Dropout(drop)

    def _time_scalar(self, x: Tensor, cond: Optional[Tensor]) -> Tensor:
        t_scalar = resolve_time_scalar(x, cond, self.t_func, self.t_from_embed)
        return torch.sigmoid(t_scalar / self.tau) * self.scale + self.shift

    def forward(self, x: Tensor, cond: Optional[Tensor] = None, return_aux: bool = False):
        step = self._time_scalar(x, cond)
        gate = self.ode_w1(x, step)
        value = self.ode_w2(x, step)
        hidden = self.drop(F.silu(gate) * value)
        out = self.w3(hidden)
        return self._maybe_return(out, self._aux(x, out, step=step, gate=gate), return_aux)


def _headwise_value_param_count(
    dim: int,
    hidden_dim_eff: int,
    num_heads: int,
    bias: bool,
    cond_dim: Optional[int],
) -> int:
    head_dim = hidden_dim_eff // num_heads
    return (
        _linear_param_count(dim, hidden_dim_eff, bias)
        + _linear_param_count(dim, hidden_dim_eff, bias)
        + _linear_param_count(hidden_dim_eff, dim, bias)
        + num_heads * head_dim * head_dim
        + _linear_param_count(dim, num_heads, True)
        + _cond_param_count(cond_dim, num_heads, True)
    )


class HeadwiseODEValueGLU(BaseFFN):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        drop: float = 0.0,
        bias: bool = True,
        tau: float = 4.0,
        scale: float = 0.8,
        shift: float = 0.1,
        orders: int = 2,
        t_embed_dim: Optional[int] = None,
        **_: object,
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")

        baseline_hidden_dim, budget = _baseline_swiglu_param_budget(dim, hidden_dim, bias)
        max_hidden_dim = _round_down_to_multiple(baseline_hidden_dim, num_heads)
        hidden_dim_eff = _search_hidden_dim(
            max_hidden_dim,
            num_heads,
            budget,
            lambda value: _headwise_value_param_count(dim, value, num_heads, bias, t_embed_dim),
        )

        self.tau = tau
        self.scale = scale
        self.shift = shift
        self.step_from_x = nn.Linear(dim, num_heads, bias=True)
        self.step_from_cond = nn.Linear(t_embed_dim, num_heads, bias=True) if t_embed_dim is not None else None
        self.gate_proj = nn.Linear(dim, hidden_dim_eff, bias=bias)
        self.value_proj = MultiHeadODELinear(dim, hidden_dim_eff, num_heads=num_heads, bias=bias, orders=orders)
        self.out_proj = nn.Linear(hidden_dim_eff, dim, bias=bias)
        self.drop = nn.Dropout(drop)

    def _step(self, x: Tensor, cond: Optional[Tensor]) -> Tensor:
        step_logits = self.step_from_x(x)
        cond_term = self._condition_term(cond, self.step_from_cond)
        if cond_term is not None:
            step_logits = step_logits + cond_term
        return torch.sigmoid(step_logits / self.tau) * self.scale + self.shift

    def forward(self, x: Tensor, cond: Optional[Tensor] = None, return_aux: bool = False):
        step = self._step(x, cond)
        gate = F.silu(self.gate_proj(x))
        value = self.value_proj(x, step)
        out = self.out_proj(self.drop(gate * value))
        return self._maybe_return(out, self._aux(x, out, step=step, gate=gate), return_aux)


def _lowrank_param_count(
    dim: int,
    hidden_dim_eff: int,
    rank: int,
    bias: bool,
    cond_dim: Optional[int],
) -> int:
    return (
        _linear_param_count(dim, hidden_dim_eff, bias)
        + _linear_param_count(dim, hidden_dim_eff, True)
        + _linear_param_count(dim, hidden_dim_eff, True)
        + _linear_param_count(hidden_dim_eff, dim, bias)
        + _linear_param_count(dim, rank, True)
        + _linear_param_count(dim, 1, True)
        + _cond_param_count(cond_dim, 1, True)
        + 2 * hidden_dim_eff * rank
    )


class LowRankStateODEFFN(BaseFFN):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        rank: int = 32,
        drop: float = 0.0,
        bias: bool = True,
        tau: float = 4.0,
        scale: float = 0.8,
        shift: float = 0.1,
        orders: int = 1,
        t_embed_dim: Optional[int] = None,
        **_: object,
    ) -> None:
        super().__init__()
        baseline_hidden_dim, budget = _baseline_swiglu_param_budget(dim, hidden_dim, bias)
        rank = max(1, min(rank, baseline_hidden_dim))
        hidden_dim_eff = _search_hidden_dim(
            baseline_hidden_dim,
            1,
            budget,
            lambda value: _lowrank_param_count(dim, value, min(rank, value), bias, t_embed_dim),
        )
        rank = min(rank, hidden_dim_eff)

        self.orders = orders
        self.tau = tau
        self.scale = scale
        self.shift = shift

        self.in_proj = nn.Linear(dim, hidden_dim_eff, bias=bias)
        self.diag_head = nn.Linear(dim, hidden_dim_eff, bias=True)
        self.rank_head = nn.Linear(dim, rank, bias=True)
        self.step_from_x = nn.Linear(dim, 1, bias=True)
        self.step_from_cond = nn.Linear(t_embed_dim, 1, bias=True) if t_embed_dim is not None else None
        self.gate_proj = nn.Linear(dim, hidden_dim_eff, bias=True)
        self.out_proj = nn.Linear(hidden_dim_eff, dim, bias=bias)
        self.U = nn.Parameter(torch.randn(hidden_dim_eff, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(hidden_dim_eff, rank) * 0.02)
        self.drop = nn.Dropout(drop)

    def apply_A(self, state: Tensor, diag: Tensor, rank_state: Tensor) -> Tensor:
        low_rank = torch.einsum("bnm,mr->bnr", state, self.V)
        low_rank = low_rank * rank_state
        low_rank = torch.einsum("bnr,mr->bnm", low_rank, self.U)
        return diag * state + low_rank

    def _step(self, x: Tensor, cond: Optional[Tensor]) -> Tensor:
        step_logits = self.step_from_x(x)
        cond_term = self._condition_term(cond, self.step_from_cond)
        if cond_term is not None:
            step_logits = step_logits + cond_term
        return torch.sigmoid(step_logits / self.tau) * self.scale + self.shift

    def forward(self, x: Tensor, cond: Optional[Tensor] = None, return_aux: bool = False):
        diag = self.diag_head(x)
        rank_state = self.rank_head(x)
        step = self._step(x, cond)

        out = self.in_proj(x)
        term = out
        for order in range(self.orders):
            term = self.apply_A(term, diag, rank_state)
            term = term * (step / float(order + 1))
            out = out + term

        gate = torch.sigmoid(self.gate_proj(x))
        out = self.out_proj(self.drop(gate * out))
        return self._maybe_return(out, self._aux(x, out, step=step, gate=gate), return_aux)


class TiedFlowFFN(BaseFFN):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        steps: int = 2,
        drop: float = 0.0,
        bias: bool = True,
        tau: float = 4.0,
        scale: float = 0.6,
        shift: float = 0.1,
        step_init: float = 0.0,
        t_embed_dim: Optional[int] = None,
        **_: object,
    ) -> None:
        super().__init__()
        if steps <= 0:
            raise ValueError("steps must be positive")

        hidden_dim_eff = max(1, int(hidden_dim * 2 / 3))
        self.steps = steps
        self.tau = tau
        self.scale = scale
        self.shift = shift
        self.step_logits = nn.Parameter(torch.full((steps,), step_init))
        self.step_from_cond = nn.Linear(t_embed_dim, steps, bias=True) if t_embed_dim is not None else None

        self.w1 = nn.Linear(dim, hidden_dim_eff, bias=bias)
        self.w2 = nn.Linear(dim, hidden_dim_eff, bias=bias)
        self.wo = nn.Linear(hidden_dim_eff, dim, bias=bias)
        self.drop = nn.Dropout(drop)

    def _field(self, z: Tensor) -> tuple[Tensor, Tensor]:
        gate = F.silu(self.w1(z))
        value = self.w2(z)
        hidden = self.drop(gate * value)
        return self.wo(hidden), gate

    def _step(self, x: Tensor, cond: Optional[Tensor]) -> Tensor:
        del x
        step_logits = self.step_logits.view(1, 1, self.steps)
        cond_term = self._condition_term(cond, self.step_from_cond)
        if cond_term is not None:
            step_logits = step_logits + cond_term
        return torch.sigmoid(step_logits / self.tau) * self.scale + self.shift

    def forward(self, x: Tensor, cond: Optional[Tensor] = None, return_aux: bool = False):
        steps = self._step(x, cond)
        z = x
        last_gate = None
        for idx in range(self.steps):
            dz, last_gate = self._field(z)
            z = z + steps[..., idx : idx + 1] * dz
        return self._maybe_return(z, self._aux(x, z, step=steps, gate=last_gate), return_aux)


def _nav_branch_param_count(dim: int, num_heads: int, bias: bool, cond_dim: Optional[int]) -> int:
    head_dim = dim // num_heads
    return (
        _linear_param_count(dim, dim, bias)
        + _linear_param_count(dim, dim, bias)
        + _linear_param_count(dim, dim, bias)
        + num_heads * head_dim * head_dim
        + _linear_param_count(dim, num_heads, True)
        + _cond_param_count(cond_dim, num_heads, True)
    )


class HeadwiseODEResidualBranch(BaseFFN):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        drop: float,
        bias: bool,
        tau: float,
        scale: float,
        shift: float,
        orders: int,
        cond_dim: Optional[int],
    ) -> None:
        super().__init__()
        self.tau = tau
        self.scale = scale
        self.shift = shift
        self.step_from_x = nn.Linear(dim, num_heads, bias=True)
        self.step_from_cond = nn.Linear(cond_dim, num_heads, bias=True) if cond_dim is not None else None
        self.gate_proj = nn.Linear(dim, dim, bias=bias)
        self.value_proj = MultiHeadODELinear(dim, dim, num_heads=num_heads, bias=bias, orders=orders)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor, cond: Optional[Tensor] = None, return_aux: bool = False):
        step_logits = self.step_from_x(x)
        cond_term = self._condition_term(cond, self.step_from_cond)
        if cond_term is not None:
            step_logits = step_logits + cond_term
        step = torch.sigmoid(step_logits / self.tau) * self.scale + self.shift
        gate = F.silu(self.gate_proj(x))
        value = self.value_proj(x, step)
        out = self.out_proj(self.drop(gate * value))
        return self._maybe_return(out, self._aux(x, out, step=step, gate=gate), return_aux)


class NavRefineFFN(BaseFFN):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        drop: float = 0.0,
        bias: bool = True,
        tau: float = 4.0,
        scale: float = 0.8,
        shift: float = 0.1,
        orders: int = 2,
        t_embed_dim: Optional[int] = None,
        mix_hidden_dim: Optional[int] = None,
        **_: object,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        baseline_hidden_dim, budget = _baseline_swiglu_param_budget(dim, hidden_dim, bias)
        mix_hidden_dim = mix_hidden_dim or max(32, dim // 8)
        mix_params = (
            _linear_param_count(t_embed_dim, mix_hidden_dim, True)
            + _linear_param_count(mix_hidden_dim, 2, True)
            if t_embed_dim is not None
            else 2
        )
        nav_params = _nav_branch_param_count(dim, num_heads, bias, t_embed_dim)
        refine_budget = max(num_heads, budget - nav_params - mix_params)
        refine_hidden_dim_eff = _search_hidden_dim(
            baseline_hidden_dim,
            1,
            refine_budget,
            lambda value: _linear_param_count(dim, 2 * value, bias) + _linear_param_count(value, dim, bias),
        )

        self.nav = HeadwiseODEResidualBranch(
            dim=dim,
            num_heads=num_heads,
            drop=drop,
            bias=bias,
            tau=tau,
            scale=scale,
            shift=shift,
            orders=orders,
            cond_dim=t_embed_dim,
        )
        self.ref = SwiGLUFFN(dim=dim, hidden_dim=_swiglu_hidden_arg_from_eff(refine_hidden_dim_eff), drop=drop, bias=bias)
        self.mix = None
        if t_embed_dim is not None:
            self.mix = nn.Sequential(
                nn.Linear(t_embed_dim, mix_hidden_dim, bias=True),
                nn.SiLU(),
                nn.Linear(mix_hidden_dim, 2, bias=True),
            )
        self.mix_bias = nn.Parameter(torch.zeros(2))

    def _mix_weights(self, x: Tensor, cond: Optional[Tensor]) -> tuple[Tensor, Tensor]:
        del x
        logits = self.mix_bias.view(1, 1, 2)
        if self.mix is not None and torch.is_tensor(cond) and cond.dim() >= 2 and cond.shape[-1] == self.mix[0].in_features:
            logits = self.mix(cond)
            if logits.dim() == 2:
                logits = logits.unsqueeze(1)
        alpha = torch.sigmoid(logits[..., 0:1])
        beta = torch.sigmoid(logits[..., 1:2])
        return alpha, beta

    def forward(self, x: Tensor, cond: Optional[Tensor] = None, return_aux: bool = False):
        nav, nav_aux = self.nav(x, cond, return_aux=True)
        ref = self.ref(x)
        alpha, beta = self._mix_weights(x, cond)
        out = alpha * nav + beta * ref
        aux = self._aux(x, out, alpha=alpha, beta=beta)
        aux["nav_step_mean"] = nav_aux["step_mean"]
        aux["nav_step_std"] = nav_aux["step_std"]
        return self._maybe_return(out, aux, return_aux)


def _mean_pool_tokens(x: Tensor) -> Tensor:
    return x.mean(dim=1, keepdim=True)


class TimeSplitFFN(BaseFFN):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop: float = 0.0,
        bias: bool = True,
        t_embed_dim: Optional[int] = None,
        nav_ratio: float = 0.5,
        gate_hidden_dim: Optional[int] = None,
        **_: object,
    ) -> None:
        super().__init__()
        baseline_hidden_dim, _ = _baseline_swiglu_param_budget(dim, hidden_dim, bias)
        nav_hidden_dim = max(1, min(baseline_hidden_dim - 1, int(round(baseline_hidden_dim * nav_ratio))))
        ref_hidden_dim = max(1, baseline_hidden_dim - nav_hidden_dim)
        gate_hidden_dim = gate_hidden_dim or min(dim, 128)

        self.nav_in = nn.Linear(dim, nav_hidden_dim, bias=bias)
        self.nav_ctx = nn.Linear(dim, nav_hidden_dim, bias=bias)
        self.nav_out = nn.Linear(nav_hidden_dim, dim, bias=bias)

        self.ref_w12 = nn.Linear(dim, 2 * ref_hidden_dim, bias=bias)
        self.ref_w3 = nn.Linear(ref_hidden_dim, dim, bias=bias)

        self.gate_from_x = nn.Sequential(
            nn.Linear(dim, gate_hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(gate_hidden_dim, 1, bias=True),
        )
        self.gate_from_cond = nn.Linear(t_embed_dim, 1, bias=True) if t_embed_dim is not None else None
        self.gate_bias = nn.Parameter(torch.tensor(0.0))
        self.drop = nn.Dropout(drop)

    def _refine(self, x: Tensor) -> tuple[Tensor, Tensor]:
        ref12 = self.ref_w12(x)
        ref_gate, ref_value = ref12.chunk(2, dim=-1)
        ref_gate = F.silu(ref_gate)
        ref_hidden = self.drop(ref_gate * ref_value)
        return self.ref_w3(ref_hidden), ref_gate

    def _mix_gate(self, x: Tensor, cond: Optional[Tensor]) -> Tensor:
        pooled = _mean_pool_tokens(x)
        gate_logits = self.gate_from_x(pooled) + self.gate_bias.view(1, 1, 1)
        cond_term = self._condition_term(cond, self.gate_from_cond)
        if cond_term is not None:
            gate_logits = gate_logits + cond_term
        return torch.sigmoid(gate_logits)

    def forward(self, x: Tensor, cond: Optional[Tensor] = None, return_aux: bool = False):
        pooled = _mean_pool_tokens(x)
        nav_hidden = self.nav_in(x) + self.nav_ctx(pooled)
        nav = self.nav_out(self.drop(F.silu(nav_hidden)))
        ref, ref_gate = self._refine(x)
        gate = self._mix_gate(x, cond)
        out = gate * nav + (1.0 - gate) * ref
        aux = self._aux(x, out, gate=gate)
        aux["ref_gate_mean"] = ref_gate.detach().mean()
        aux["ref_gate_std"] = ref_gate.detach().std(unbiased=False)
        return self._maybe_return(out, aux, return_aux)


class CleanTargetFFN(BaseFFN):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop: float = 0.0,
        bias: bool = True,
        t_embed_dim: Optional[int] = None,
        tau: float = 4.0,
        min_denom: float = 0.1,
        alpha_init: float = -2.0,
        max_update_rms_scale: float = 1.0,
        apply_update_norm: bool = True,
        time_hidden_dim: Optional[int] = None,
        **_: object,
    ) -> None:
        super().__init__()
        hidden_dim_eff = max(1, int(hidden_dim * 2 / 3))
        time_hidden_dim = time_hidden_dim or min(dim, 128)

        self.tau = tau
        self.min_denom = min_denom
        self.max_update_rms_scale = max_update_rms_scale
        self.apply_update_norm = apply_update_norm

        self.t_func = build_time_mlp(dim, time_hidden_dim, bias=bias)
        self.t_from_embed = build_time_mlp(t_embed_dim, time_hidden_dim, bias=bias) if t_embed_dim is not None else None

        self.pred_w12 = nn.Linear(dim, 2 * hidden_dim_eff, bias=bias)
        self.pred_from_cond = nn.Linear(t_embed_dim, 2 * hidden_dim_eff, bias=True) if t_embed_dim is not None else None
        self.pred_w3 = nn.Linear(hidden_dim_eff, dim, bias=bias)

        self.alpha_from_x = nn.Linear(dim, 1, bias=True)
        self.alpha_from_cond = nn.Linear(t_embed_dim, 1, bias=True) if t_embed_dim is not None else None
        self.alpha_bias = nn.Parameter(torch.tensor(alpha_init))
        self.drop = nn.Dropout(drop)

    def _time_fraction(self, x: Tensor, cond: Optional[Tensor]) -> Tensor:
        time_logits = resolve_time_scalar(x, cond, self.t_func, self.t_from_embed)
        return torch.sigmoid(time_logits / self.tau)

    def _alpha(self, x: Tensor, cond: Optional[Tensor]) -> Tensor:
        pooled = _mean_pool_tokens(x)
        alpha_logits = self.alpha_from_x(pooled) + self.alpha_bias.view(1, 1, 1)
        cond_term = self._condition_term(cond, self.alpha_from_cond)
        if cond_term is not None:
            alpha_logits = alpha_logits + cond_term
        return torch.sigmoid(alpha_logits)

    def forward(self, x: Tensor, cond: Optional[Tensor] = None, return_aux: bool = False):
        pred12 = self.pred_w12(x)
        cond_term = self._condition_term(cond, self.pred_from_cond)
        if cond_term is not None:
            pred12 = pred12 + cond_term
        pred_gate, pred_value = pred12.chunk(2, dim=-1)
        pred_gate = F.silu(pred_gate)
        x0_hat = self.pred_w3(self.drop(pred_gate * pred_value))

        time_fraction = self._time_fraction(x, cond)
        denom = (1.0 - time_fraction).clamp_min(self.min_denom)
        update = (x0_hat - x) / denom
        if self.apply_update_norm:
            x_rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
            update_rms = torch.sqrt(torch.mean(update * update, dim=-1, keepdim=True) + 1e-6)
            limit = self.max_update_rms_scale * x_rms
            scale = torch.clamp(limit / update_rms, max=1.0)
            update = update * scale

        alpha = self._alpha(x, cond)
        out = alpha * update
        aux = self._aux(x, out, step=time_fraction, alpha=alpha, gate=pred_gate)
        aux["denom_mean"] = denom.detach().mean()
        aux["denom_min"] = denom.detach().amin()
        return self._maybe_return(out, aux, return_aux)


class TimeMoEFFN(BaseFFN):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop: float = 0.0,
        bias: bool = True,
        t_embed_dim: Optional[int] = None,
        gate_hidden_dim: Optional[int] = None,
        **_: object,
    ) -> None:
        super().__init__()
        baseline_hidden_dim, _ = _baseline_swiglu_param_budget(dim, hidden_dim, bias)
        expert_hidden_dim = max(1, baseline_hidden_dim // 3)
        gate_hidden_dim = gate_hidden_dim or min(dim, 128)

        self.coarse_in = nn.Linear(dim, expert_hidden_dim, bias=bias)
        self.coarse_ctx = nn.Linear(dim, expert_hidden_dim, bias=bias)
        self.coarse_out = nn.Linear(expert_hidden_dim, dim, bias=bias)

        self.mid_w12 = nn.Linear(dim, 2 * expert_hidden_dim, bias=bias)
        self.mid_w3 = nn.Linear(expert_hidden_dim, dim, bias=bias)

        self.ref_w12 = nn.Linear(dim, 2 * expert_hidden_dim, bias=bias)
        self.ref_w3 = nn.Linear(expert_hidden_dim, dim, bias=bias)

        self.router_from_x = nn.Sequential(
            nn.Linear(dim, gate_hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(gate_hidden_dim, 3, bias=True),
        )
        self.router_from_cond = nn.Linear(t_embed_dim, 3, bias=True) if t_embed_dim is not None else None
        self.router_bias = nn.Parameter(torch.zeros(3))
        self.drop = nn.Dropout(drop)

    def _swiglu(self, x: Tensor, w12: nn.Linear, w3: nn.Linear) -> Tensor:
        x12 = w12(x)
        gate, value = x12.chunk(2, dim=-1)
        hidden = self.drop(F.silu(gate) * value)
        return w3(hidden)

    def _router(self, x: Tensor, cond: Optional[Tensor]) -> Tensor:
        pooled = _mean_pool_tokens(x)
        logits = self.router_from_x(pooled) + self.router_bias.view(1, 1, 3)
        cond_term = self._condition_term(cond, self.router_from_cond)
        if cond_term is not None:
            logits = logits + cond_term
        return torch.softmax(logits, dim=-1)

    def forward(self, x: Tensor, cond: Optional[Tensor] = None, return_aux: bool = False):
        pooled = _mean_pool_tokens(x)
        coarse = self.coarse_out(self.drop(F.silu(self.coarse_in(x) + self.coarse_ctx(pooled))))
        balanced = self._swiglu(x, self.mid_w12, self.mid_w3)
        refined = self._swiglu(x - pooled, self.ref_w12, self.ref_w3)

        weights = self._router(x, cond)
        out = (
            weights[..., 0:1] * coarse
            + weights[..., 1:2] * balanced
            + weights[..., 2:3] * refined
        )
        aux = self._aux(x, out)
        aux["expert0_mean"] = weights[..., 0].detach().mean()
        aux["expert1_mean"] = weights[..., 1].detach().mean()
        aux["expert2_mean"] = weights[..., 2].detach().mean()
        return self._maybe_return(out, aux, return_aux)


MHODESwiGLUFFN = MultiHeadODESwiGLUFFN


FFN_REGISTRY = {
    "swiglu": SwiGLUFFN,
    "mlp": MLP,
    "ode": ODEOnlyFFN,
    "ode_swiglu": ODESwiGLUFFN,
    "mh_ode_swiglu": MultiHeadODESwiGLUFFN,
    "headwise_ode_value_glu": HeadwiseODEValueGLU,
    "lowrank_state_ode": LowRankStateODEFFN,
    "tied_flow": TiedFlowFFN,
    "nav_refine": NavRefineFFN,
    "time_split": TimeSplitFFN,
    "clean_target": CleanTargetFFN,
    "time_moe": TimeMoEFFN,
}


FFN_ALIASES = {
    "mhode_swiglu": "mh_ode_swiglu",
    "multihead_ode_swiglu": "mh_ode_swiglu",
    "headwise_ode_glu": "headwise_ode_value_glu",
    "headwise_value_ode_glu": "headwise_ode_value_glu",
    "lowrank_ode": "lowrank_state_ode",
    "lowrank_state_ode_ffn": "lowrank_state_ode",
    "tied_flow_ffn": "tied_flow",
    "nav_refine_ffn": "nav_refine",
    "time_split_dual_path": "time_split",
    "time_split_ffn": "time_split",
    "clean_target_ffn": "clean_target",
    "time_routed_moe": "time_moe",
    "time_moe_ffn": "time_moe",
}
