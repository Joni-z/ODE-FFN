
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
        self.proj = nn.Linear(in_features, in_features, bias=False)

    @torch.compile
    def forward(self, x: Tensor) -> Tensor:
        t = self.t_func(x)
        t = torch.sigmoid(t / self.tau) * self.scale + self.shift
        approx = [x]
        for i in range(self.orders):
            term = (self.proj(approx[-1]) * t) / (i + 1)
            approx.append(term)
        return sum(approx)
        # return sum(approx[1:])

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