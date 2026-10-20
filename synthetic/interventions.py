"""Read-only changes to ODELayer evaluation; training parameters are never edited."""
from contextlib import contextmanager
from types import MethodType

import torch
from torch.nn import functional as F


def action(h, p, s, order, chunk=512):
    """Evaluate a Taylor polynomial or matrix exponential in the supplied dtype."""
    if chunk < 1:
        raise ValueError("chunk must be positive")
    if order == "exact":
        pieces = []
        for start in range(0, len(h), chunk):
            x = h[start:start+chunk]
            step = s[start:start+chunk].reshape(-1, 1, 1)
            matrices = torch.matrix_exp(step * p)
            pieces.append(torch.bmm(matrices, x.unsqueeze(-1)).squeeze(-1))
        return torch.cat(pieces)
    if type(order) is not int or order < 1:
        raise ValueError("order must be a positive integer or 'exact'")
    term = h
    total = h
    for k in range(1, order+1):
        term = F.linear(term, p) * s / k
        total = total + term
    return total


@contextmanager
def evaluation_order(model, order, chunk=512):
    """Freeze weights, not activations: gates still respond to upstream changes.

    All interventions use float64 for P's action, then cast back to the original
    hidden dtype. Gate computation and the rest of the network remain unchanged.
    Native evaluation outside this context is retained as a precision control.
    """
    saved = []
    try:
        for _, layer in model.ode_layers():
            saved.append((layer, layer.__dict__.get("forward")))
            p = layer.operator().detach().double()

            def forward(self, x, context=None, operator=p):
                s = self.step_size(x, context).double()
                return action(x.double(), operator, s, order, chunk).to(x.dtype)

            layer.forward = MethodType(forward, layer)
        yield model
    finally:
        for layer, old in saved:
            if old is None:
                del layer.forward
            else:
                layer.forward = old
