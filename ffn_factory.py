# --------------------------------------------------------
# FFN factory: build FFN by type for ablation (swiglu / ode / mlp / ode_swiglu).
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .ode_ffn import ODEFFN, ODELayer  # 确保 ode_ffn.py 里导出了 ODELayer


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class ODESwiGLUFFN(nn.Module):
    """
    ODE + SwiGLU hybrid FFN:
      x_ode = x + alpha * (ODELayer(x) - x)   (alpha init=0 for safe ablation)
      out   = SwiGLU(x_ode)
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop: float = 0.0,
        bias: bool = True,
        # ODE hyperparams
        tau: float = 10.0,
        scale: float = 0.8,
        shift: float = 0.2,
        orders: int = 2,
        # width for t_func inside ODELayer (not the same as SwiGLU hidden_dim)
        ode_hidden_features: Optional[int] = None,
    ) -> None:
        super().__init__()

        # 1) ODE preconditioner on token feature (dim)
        ode_hidden_features = ode_hidden_features or dim
        self.ode = ODELayer(
            in_features=dim,
            hidden_features=ode_hidden_features,
            bias=bias,
            tau=tau,
            scale=scale,
            shift=shift,
            orders=orders,
        )

        # learnable interpolation; start from pure identity => exactly baseline SwiGLU
        self.ode_alpha = nn.Parameter(torch.zeros(()))  # scalar

        # 2) Standard SwiGLU body (keep identical for fair comparison)
        hidden_dim_eff = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim_eff, bias=bias)
        self.w3 = nn.Linear(hidden_dim_eff, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        # safe mixing: at init ode_alpha=0 => x_ode = x
        x_ode = x + self.ode_alpha * (self.ode(x) - x)

        x12 = self.w12(x_ode)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        hidden = self.ffn_dropout(hidden)
        return self.w3(hidden)


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return self.fc2(x)


def build_ffn(
    ffn_type: str,
    in_features: int,
    hidden_features: int,
    drop: float = 0.0,
    bias: bool = True,
    **ode_kwargs,
) -> nn.Module:
    """
    ffn_type: "swiglu" | "ode" | "mlp" | "ode_swiglu"
    ode_kwargs: (tau, scale, shift, orders, ode_hidden_features, ...)
    """
    ffn_type = ffn_type.lower().strip()

    if ffn_type == "swiglu":
        return SwiGLUFFN(in_features, hidden_features, drop=drop, bias=bias)

    if ffn_type == "ode":
        return ODEFFN(
            in_features=in_features,
            hidden_features=hidden_features,
            bias=bias,
            **ode_kwargs,
        )

    if ffn_type == "ode_swiglu":
        return ODESwiGLUFFN(
            dim=in_features,
            hidden_dim=hidden_features,
            drop=drop,
            bias=bias,
            **ode_kwargs,
        )

    if ffn_type == "mlp":
        return MLP(
            in_features=in_features,
            hidden_features=hidden_features,
            drop=drop,
            bias=bias,
        )

    raise ValueError(f"Unknown ffn_type: {ffn_type}. Use one of: swiglu, ode, ode_swiglu, mlp")
