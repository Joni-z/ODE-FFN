"""Classical step copied from torchdiffeq; see rk4-torchdiffeq-LICENSE.

Source: rtqichen/torchdiffeq@657943acefa826ef04c025ebeb1ff5e9d60dc268,
torchdiffeq/_impl/rk_common.py::rk4_step_func (Git blob f0050dbfd30f90234b0921a10acc18890c0a351d).
The local enum supplies standalone flags; our autonomous callback ignores them.
The numerical step below is unchanged from upstream.
"""
from enum import Enum


class Perturb(Enum):
    NONE = 0
    PREV = 1
    NEXT = 2


_one_sixth = 1 / 6


def rk4_step_func(func, t0, dt, t1, y0, f0=None, perturb=False):
    k1 = f0
    if k1 is None:
        k1 = func(t0, y0, perturb=Perturb.NEXT if perturb else Perturb.NONE)
    half_dt = dt * 0.5
    k2 = func(t0 + half_dt, y0 + half_dt * k1)
    k3 = func(t0 + half_dt, y0 + half_dt * k2)
    k4 = func(t1, y0 + dt * k3, perturb=Perturb.PREV if perturb else Perturb.NONE)
    return (k1 + 2 * (k2 + k3) + k4) * dt * _one_sixth
