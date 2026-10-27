"""Independent matrix-exponential and derivative checks for the exact baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F


def verify(component: type[nn.Module]) -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = component(nn.Linear(4, 10), nn.Linear(5, 4)).double()
    x = torch.cos(torch.arange(12, dtype=torch.float64)).reshape(3, 4).requires_grad_(True)
    gate, value = model.w12(x).chunk(2, -1)
    g = F.silu(gate)
    diagonal = torch.diag_embed(g)
    zero = torch.zeros_like(diagonal)
    p = torch.cat([torch.cat([zero, diagonal], -1), torch.cat([zero, zero], -1)], -2)
    identity = torch.eye(10, dtype=torch.float64).expand(3, 10, 10)
    exponential = torch.matrix_exp(p)
    torch.testing.assert_close(p @ p, torch.zeros_like(p), rtol=0, atol=0)
    torch.testing.assert_close(exponential, identity + p, rtol=1e-12, atol=1e-12)
    initial = torch.cat([torch.zeros_like(value), value], -1)
    endpoint = torch.einsum('bij,bj->bi', exponential, initial)
    literal = model.w3(endpoint[:, :5])
    efficient = model(x)
    torch.testing.assert_close(literal, efficient, rtol=1e-12, atol=1e-12)
    weights = torch.sin(torch.arange(12, dtype=torch.float64)).reshape(3, 4)
    derivatives = []
    parameters = (x, *model.parameters())
    for result in (literal, efficient):
        loss = (result * weights).sum() + .03 * result.square().sum()
        derivatives.append(torch.autograd.grad(loss, parameters, retain_graph=True))
    differences = []
    for a, b in zip(*derivatives):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-11)
        differences.append(float((a-b).abs().max()))
    u = torch.sin(torch.arange(15, dtype=torch.float64)).reshape(3,5)
    first = component.flow(g, u, value, .3)
    second = component.flow(g, *first, .7)
    whole = component.flow(g, u, value, 1.)
    inverse = component.flow(g, *whole, -1.)
    torch.testing.assert_close(second[0], whole[0], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(inverse[0], u, rtol=1e-12, atol=1e-12)
    expected_singular = torch.sort(torch.cat([g.abs(), torch.zeros_like(g)], -1), descending=True).values
    torch.testing.assert_close(torch.linalg.svdvals(p), expected_singular, rtol=1e-12, atol=1e-12)
    strength = g.abs().amax(-1)
    expected_norm = .5 * (torch.sqrt(4+strength.square()) + strength)
    torch.testing.assert_close(torch.linalg.matrix_norm(exponential,ord=2),expected_norm,rtol=1e-12,atol=1e-12)
    witness = torch.tensor([[0.,100.],[0.,0.]],dtype=torch.float64)
    assert torch.equal(torch.linalg.eigvals(witness),torch.zeros(2,dtype=torch.complex128))
    witness_norm = float(torch.linalg.matrix_norm(torch.matrix_exp(witness),ord=2))
    assert witness_norm > 100.
    # All biases are retained; the zero-input case is not assumed to output zero.
    with torch.no_grad():
        zz = torch.zeros(2,4,dtype=torch.float64)
        a,b=model.w12(zz).chunk(2,-1)
        torch.testing.assert_close(model(zz),model.w3(F.silu(a)*b),rtol=0,atol=0)
        for layer in (model.w12,model.w3):layer.bias.zero_()
        assert torch.equal(model(zz),torch.zeros_like(zz))
    return dict(status='PASS',literal_exponential_value_max=float((literal-efficient).abs().max().detach()),
        input_and_parameter_gradient_max=max(differences),gradient_tensors=len(differences),
        witness_zero_eigenvalues=True,witness_flow_norm=witness_norm,
        semigroup_and_inverse=True,exact_singular_values=True,exact_flow_norm=True,bias_cases=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--component-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();sys.path.insert(0,str(args.component_dir));from gated_shear import GatedShearFFN
    result=verify(GatedShearFFN);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
