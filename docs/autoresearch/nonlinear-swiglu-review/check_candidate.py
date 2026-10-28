"""Bounded algebra/cost fixtures, with no fitted model or dataset access."""
from __future__ import annotations

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.integrate import quad
from scipy.special import exp1
import torch
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def parts(p):
    w12, wo = p[:2]
    b12, bo = p[2:] if len(p) == 4 else (None, None)
    return w12, wo, b12, bo


def field(p, h):
    w12, wo, b12, bo = parts(p)
    a, b = F.linear(h, w12, b12).chunk(2, dim=-1)
    return F.linear(F.silu(a) * b, wo, bo)


def slope(a):
    s = torch.sigmoid(a)
    return s + a * s * (1-s)


def curvature(a):
    s = torch.sigmoid(a)
    return s * (1-s) * (2 + a * (1-2*s))


def tangent(p, h, u):
    w12, wo, b12, _ = parts(p)
    a, b = F.linear(h, w12, b12).chunk(2, dim=-1)
    au, bu = F.linear(u, w12).chunk(2, dim=-1)
    return F.linear(slope(a) * au * b + F.silu(a) * bu, wo)


def hessian(p, h, u, v):
    w12, wo, b12, _ = parts(p)
    a, b = F.linear(h, w12, b12).chunk(2, dim=-1)
    au, bu = F.linear(u, w12).chunk(2, dim=-1)
    av, bv = F.linear(v, w12).chunk(2, dim=-1)
    return F.linear(curvature(a)*au*av*b + slope(a)*(au*bv+av*bu), wo)


def analytic_t2(p, h):
    # Reuse the native intermediates; do not execute the first projection twice.
    w12, wo, b12, bo = parts(p)
    a, b = F.linear(h, w12, b12).chunk(2, dim=-1)
    activation = F.silu(a)
    q = F.linear(activation*b, wo, bo)
    aq, bq = F.linear(q, w12).chunk(2, dim=-1)
    correction = F.linear(slope(a)*aq*b + activation*bq, wo)
    return q + .5*correction


def ad_t2(p, h):
    q = field(p, h)
    return q + .5*torch.func.jvp(lambda z: field(p, z), (h,), (q,))[1]


def frozen_gate(p, h):
    w12, wo, b12, bo = parts(p)
    a, b = F.linear(h, w12, b12).chunk(2, dim=-1)
    activation = F.silu(a)
    q = F.linear(activation*b, wo, bo)
    value_weight = w12.chunk(2, dim=0)[1]
    return q + .5*F.linear(activation * F.linear(q, value_weight), wo)


def midpoint(p, h):
    return field(p, h + .5*field(p, h))


def normalized_error(a, b):
    return float((a-b).abs().max().detach()) / (1+float(b.abs().max().detach()))


def parameters(native, bias):
    result = [native.w12.weight, native.w3.weight]
    if bias:
        result += [native.w12.bias, native.w3.bias]
    return tuple(p.detach().clone().requires_grad_() for p in result)


class DenseCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.records = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        pair = None
        if name == 'aten.mm.default':
            pair = args[:2]
        elif name == 'aten.addmm.default':
            pair = args[1:3]
        elif name == 'aten.bmm.default':
            a, b = args[:2]
            self.records.append(dict(op=name, a=list(a.shape), b=list(b.shape),
                                     mac=a.shape[0]*a.shape[1]*a.shape[2]*b.shape[2]))
        if pair is not None:
            a, b = pair
            self.records.append(dict(op=name, a=list(a.shape), b=list(b.shape),
                                     mac=a.shape[-2]*a.shape[-1]*b.shape[-1]))
        return func(*args, **(kwargs or {}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    assert not output.exists(), 'Preserve any previous run; never overwrite it.'
    output.mkdir(parents=True)
    source = output/'source'
    source.mkdir()
    document = Path(__file__).resolve().parent
    for name in ('check_candidate.py', 'protocol.md'):
        shutil.copyfile(document/name, source/name)
    paths = [source/'check_candidate.py', source/'protocol.md', repo/'synthetic/models.py',
             repo/'docs/autoresearch/cycle01-complete-field/analysis.md',
             repo/'docs/research-architecture-search/nonlinear-drift-readout.md',
             repo/'docs/research-architecture-search/coupled-generator-prior-review.md']
    bindings = {str(p): sha(p) for p in paths}
    write(output/'bindings.json', bindings)
    write(output/'execution.json', dict(command=sys.argv, pid=os.getpid(), host=os.uname().nodename,
          source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True).strip(),
          executable=sys.executable, torch=torch.__version__, numpy=np.__version__,
          new_training=False, model_seed=0))
    started = time.monotonic()
    sys.path.insert(0, str(repo))
    from synthetic.models import SwiGLUFFN
    torch.set_num_threads(1)
    rng = torch.Generator().manual_seed(0)
    records = []
    for d, nominal in ((1,2), (4,8), (64,128)):
        torch.manual_seed(0)
        native = SwiGLUFFN(d, nominal).double()
        r = native.w3.in_features
        for bias in (False, True):
            p = parameters(native, bias)
            for scale in (.1, 1., 4.):
                h = torch.randn(d, generator=rng, dtype=torch.float64)*scale
                u, v = (torch.randn(d, generator=rng, dtype=torch.float64) for _ in range(2))
                f = lambda z: field(p, z)
                q = f(h)
                j_u = torch.func.jvp(f, (h,), (u,))[1]
                hu_v = torch.func.jvp(lambda z: torch.func.jvp(f, (z,), (v,))[1], (h,), (u,))[1]
                t2 = analytic_t2(p, h)
                reference = ad_t2(p, h)
                complete = tangent(p,h,u) + .5*(hessian(p,h,u,q) + tangent(p,h,tangent(p,h,u)))
                full_ad = torch.func.jvp(lambda z: ad_t2(p,z), (h,), (u,))[1]
                grad = torch.autograd.grad(t2.square().mean(), p)
                ref_grad = torch.autograd.grad(reference.square().mean(), p)
                errors = dict(tangent=normalized_error(tangent(p,h,u),j_u),
                    hessian=normalized_error(hessian(p,h,u,v),hu_v),
                    t2=normalized_error(t2,reference), complete_t2_tangent=normalized_error(complete,full_ad),
                    full_parameter_gradients=max(normalized_error(a,b) for a,b in zip(grad,ref_grad)))
                native_error = None
                if bias:
                    native_error = normalized_error(field(p,h),native(h))
                    assert native_error == 0
                directions = tuple(torch.randn(a.shape,generator=rng,dtype=a.dtype)/math.sqrt(a.numel()) for a in p)
                directional = sum(float((a*b).sum()) for a,b in zip(grad,directions))
                fd = []
                for step in (1e-5,5e-6):
                    plus = tuple(a+step*b for a,b in zip(p,directions))
                    minus = tuple(a-step*b for a,b in zip(p,directions))
                    numeric = float((analytic_t2(plus,h).square().mean()-analytic_t2(minus,h).square().mean()).detach())/(2*step)
                    fd.append(dict(step=step, analytic=directional, finite_difference=numeric,
                                   normalized_error=abs(numeric-directional)/(1+abs(directional))))
                assert max(errors.values()) <= 1e-10, errors
                assert max(row['normalized_error'] for row in fd) <= 1e-5, fd
                records.append(dict(dim=d, nominal_hidden=nominal, actual_hidden=r,bias=bias,scale=scale,
                    errors=errors,native_error=native_error,finite_differences=fd,
                    frozen_gate_omission_RMS=float(torch.mean((t2-frozen_gate(p,h))**2).sqrt().detach())))
    write(output/'derivative-checks.json', records)
    costs = []
    for name, fn, nominal in (('native',field,128),('taylor2',analytic_t2,128),
                             ('frozen_gate',frozen_gate,128),('midpoint',midpoint,128),
                             ('wide_actual170',field,255)):
        torch.manual_seed(0)
        model = SwiGLUFFN(64,nominal).double()
        p = parameters(model,True)
        x = torch.randn(8,64,generator=rng,dtype=torch.float64,requires_grad=True)
        counter = DenseCounter()
        with counter:
            y = fn(p,x)
            split = len(counter.records)
            y.sum().backward()
        forward = sum(row['mac'] for row in counter.records[:split])//8
        backward = sum(row['mac'] for row in counter.records[split:])//8
        r = model.w3.in_features
        multiple = 5 if name=='frozen_gate' else 3 if name in ('native','wide_actual170') else 6
        assert forward == multiple*64*r and backward == 2*forward
        assert all(a.grad is not None and torch.isfinite(a.grad).all() for a in p)
        costs.append(dict(arm=name,dim=64,actual_hidden=r,parameters=sum(a.numel() for a in p),
                          forward_MAC_per_row=forward,backward_MAC_per_row=backward,
                          total_training_MAC_per_row=forward+backward,
                          dense_calls=counter.records,forward_call_count=split,
                          all_input_and_parameter_gradients_active=True))
    write(output/'dense-work.json', costs)
    escape = []
    for y0 in (1.,2.,4.):
        original, e_original = quad(lambda y:(1+math.exp(-y))/y**2,y0,np.inf,epsabs=1e-12,epsrel=1e-12)
        inverse, e_inverse = quad(lambda z:1+math.exp(-1/z) if z>0 else 1.,0,1/y0,epsabs=1e-12,epsrel=1e-12)
        closed = float(1/y0+math.exp(-y0)/y0-exp1(y0))
        assert max(abs(original-closed),abs(inverse-closed))<=1e-10
        assert 1/y0<closed<(1+math.exp(-y0))/y0
        escape.append(dict(y0=y0,quadrature_original=original,quadrature_inverse=inverse,
                           exponential_integral_identity=closed,quad_error_original=e_original,
                           quad_error_inverse=e_inverse,lower_bound=1/y0,upper_bound=(1+math.exp(-y0))/y0))
    assert escape[1]['upper_bound']<1
    write(output/'finite-escape.json', escape)
    for path,digest in bindings.items():
        assert sha(path)==digest, path
    elapsed = time.monotonic()-started
    assert elapsed<=120
    write(output/'verification.json',dict(status='PASS',derivative_cases=len(records),
          finite_difference_checks=sum(len(row['finite_differences']) for row in records),
          cost_arms=len(costs),escape_cases=len(escape),wall_seconds=elapsed,new_training=False,
          useful_model_demonstrated=False,global_flow_guarantee_refuted=True,training_nominated=False))
    print(json.dumps(json.loads((output/'verification.json').read_text())),flush=True)


if __name__ == '__main__':
    main()
