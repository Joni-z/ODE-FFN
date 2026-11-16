"""Compile one original coupled RK4step, preserving the complete eager model.

API: https://docs.pytorch.org/docs/2.8/generated/torch.compile.html
The canonical hidden_defect_transport implementation is not edited.
"""
from unittest.mock import patch
import torch
import torch.nn.functional as F
import hidden_defect_transport as original


def coupled_step(state, matrix, dt, native, crossed):
    width = matrix.shape[0]
    def drift(time, value, **unused):
        point, tangent = (part.contiguous() for part in value.split(width, dim=1))
        primal = F.linear(original.response(point), matrix)
        derivative = F.linear((1+1.2*point.square()).pow(-1.5)*tangent, matrix)
        derivative = derivative+original.forcing(point, matrix, native, crossed)
        return torch.cat((primal, derivative), dim=1)
    # This drift is autonomous: RK4 time arguments have no numerical effect.
    return state+original.rk4_step_func(drift, 0., dt, dt, state)


def make_compiled_step():
    return torch.compile(coupled_step, fullgraph=True, dynamic=False)


@torch.no_grad()
def response_with_step(model, x, time, steps, native, crossed, step):
    def solve(y, z, matrix, count, n, c):
        state = torch.cat((y, z), dim=1)
        for _ in range(count):
            state = step(state, matrix, 1/count, n, c)
        return tuple(part.contiguous() for part in state.split(y.shape[1], dim=1))
    with patch.object(original, 'variation_solve', new=solve):
        return original.model_response(model, x, time, steps, native, crossed)
