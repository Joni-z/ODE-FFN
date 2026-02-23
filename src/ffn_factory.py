import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ode_ffn import ODEFFN, ODELayer


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x, t: Optional[torch.Tensor] = None):
        # keep signature compatible with t-passthrough
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class ODESwiGLUFFN(nn.Module):
    """
    ODE + SwiGLU hybrid FFN (FM-friendly):
      base = SwiGLU(x)
      ode  = (ODELayer(x,t) - x)  (ODE bias)
      out  = base + gate(t) * ode

    Key points:
      - gate is bounded via sigmoid(logit)
      - supports passing flow-time t (scalar or embedding)
      - does NOT perturb SwiGLU input distribution (more stable than preconditioning)
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
        orders: int = 1,  # IMPORTANT: start with 1; you can ablate 2/3 later
        ode_hidden_features: Optional[int] = None,
        # NEW: if you pass time embedding into forward, set t_embed_dim
        t_embed_dim: Optional[int] = None,
        # NEW: stabilize delta magnitude
        delta_normalize: bool = True,
        # NEW: gate init; use 0 so gate=0.5 and ODE branch has visible contribution (was -5 ~ 0.0067)
        gate_init_logit: float = 0.0,
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

        # bounded gate in (0,1)
        self.ode_gate_logit = nn.Parameter(torch.tensor(gate_init_logit))
        self.delta_normalize = delta_normalize

        # SwiGLU (unchanged)
        hidden_dim_eff = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim_eff, bias=bias)
        self.w3 = nn.Linear(hidden_dim_eff, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def _swiglu(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        hidden = self.ffn_dropout(hidden)
        return self.w3(hidden)

    def forward(self, x, t: Optional[torch.Tensor] = None):
        base = self._swiglu(x)

        # ODE bias branch
        ode_out = self.ode(x, t) if t is not None else self.ode(x)
        delta = ode_out - x

        if self.delta_normalize:
            # match delta RMS to x RMS (token-wise), prevents delta from dominating
            x_rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
            d_rms = torch.sqrt(torch.mean(delta * delta, dim=-1, keepdim=True) + 1e-6)
            delta = delta * (x_rms / d_rms)

        gate = torch.sigmoid(self.ode_gate_logit)  # scalar in (0,1)
        return base + gate * delta


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x, t: Optional[torch.Tensor] = None):
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
    ode_kwargs: (tau, scale, shift, orders, ode_hidden_features, t_embed_dim, delta_normalize, gate_init_logit, ...)
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