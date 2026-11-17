"""Analytic and measured field forcings through the original discrete sampler."""
import torch


class JointSensitivityField(torch.nn.Module):
    """The native field plus two independent first variations at lambda zero."""

    def __init__(self, native, crossed, analytic_response, steps, observer=None):
        super().__init__()
        self.native, self.crossed = native, crossed
        self.analytic_response = analytic_response
        self.steps, self.calls, self.observer = steps, 0, observer
        self.primal_max_error = 0.
        self.primal_qualified = True

    @torch.no_grad()
    def forward(self, state, time):
        x, ua, um = (v.contiguous() for v in state.chunk(3, dim=1))
        direct = self.native(x, time)
        pa, ja = torch.func.jvp(lambda point: self.native(point, time), (x,), (ua,))
        pm, jm = torch.func.jvp(lambda point: self.native(point, time), (x,), (um,))
        for primal in (pa, pm):
            error = (primal-direct).abs()
            self.primal_max_error = max(self.primal_max_error, float(error.max()))
            self.primal_qualified &= bool(torch.all(error <= 1e-12*(1+direct.abs())))
        analytic = self.analytic_response(x, time)
        measured = self.crossed(x, time)-direct
        if self.observer is not None:
            self.observer(self.calls, x, time, analytic)
        self.calls += 1
        assert self.calls <= 2*self.steps
        return torch.cat((direct, ja+analytic, jm+measured), dim=1)


class ForcedField(torch.nn.Module):
    """Independent finite-difference field, without the tangent equations."""

    def __init__(self, native, forcing, coefficient):
        super().__init__()
        self.native, self.forcing, self.coefficient = native, forcing, coefficient

    @torch.no_grad()
    def forward(self, x, time):
        return self.native(x, time)+self.coefficient*self.forcing(x, time)
