
import torch
from typing import Optional
from torch import Tensor, nn

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
        self.t_func = nn.Sequential(
            nn.Linear(in_features, hidden_features, bias=bias),
            nn.SiLU(),
            nn.Linear(hidden_features, 1, bias=bias),
        )
        # if given external t embedding, map it to scalar step
        self.t_from_embed = None
        if t_embed_dim is not None:
            self.t_from_embed = nn.Sequential(
                nn.Linear(t_embed_dim, hidden_features, bias=bias),
                nn.SiLU(),
                nn.Linear(hidden_features, 1, bias=bias),
            )

        self.proj = nn.Linear(in_features, in_features, bias=False)

    @torch.compile
    def forward(self, x: Tensor, t: Optional[Tensor] = None) -> Tensor:
        """
        x: (B, N, D)
        t:
          - None: use old t_func(x)  (feature-dependent)
          - scalar time: shape (B,) or (B,1) or (B,N,1)
          - time embedding: shape (B, t_embed_dim) or (B,N,t_embed_dim) if t_from_embed is set
        """
        if t is None:
            t_scalar = self.t_func(x)  # (B,N,1)
        else:
            # Case 1: have learned mapper for embedding
            if self.t_from_embed is not None and t.dim() >= 2 and t.shape[-1] != 1:
                # (B, tE) or (B,N,tE) -> (B,1) or (B,N,1)
                t_scalar = self.t_from_embed(t)
            else:
                # Case 2: already scalar-like
                if t.dim() == 1:
                    t_scalar = t[:, None, None]          # (B,1,1)
                elif t.dim() == 2 and t.shape[-1] == 1:
                    t_scalar = t[:, None, :]             # (B,1,1)
                else:
                    # (B,N,1)
                    t_scalar = t

        t_scalar = torch.sigmoid(t_scalar / self.tau) * self.scale + self.shift  # (..,1)

        approx = [x]
        for i in range(self.orders):
            term = (self.proj(approx[-1]) * t_scalar) / (i + 1)
            approx.append(term)
        return sum(approx)

class ODEFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        bias: bool = True,
        tau: float = 10.0,
        scale: float = 0.8,
        shift: float = 0.2,
        orders: int = 2,
    ) -> None:
        super().__init__()
        # First ODE-based block
        self.ode1 = ODELayer(
            in_features=in_features,
            hidden_features=hidden_features,
            bias=bias,
            tau=tau,
            scale=scale,
            shift=shift,
            orders=orders,
        )
        # Nonlinearity between solves
        self.act = nn.SiLU()
        # Second ODE-based block
        self.ode2 = ODELayer(
            in_features=in_features,
            hidden_features=hidden_features,
            bias=bias,
            tau=tau,
            scale=scale,
            shift=shift,
            orders=orders,
        )

    @torch.compile
    def forward(self, x: Tensor) -> Tensor:
        # first Taylor-ODE solve
        x1 = self.ode1(x)
        # nonlinear mixing
        x2 = self.act(x1)
        # second Taylor-ODE solve
        return self.ode2(x2)

if __name__ == "__main__":
    # Example usage
    model = ODEFFN(in_features=16, hidden_features=64)
    x = torch.randn(2, 4, 16)
    output = model(x)
    print(output.shape)