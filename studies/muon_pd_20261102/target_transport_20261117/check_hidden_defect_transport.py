"""Deterministic full-model checks; never load data or learned checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from hidden_defect_transport import defects,model_response,forced_model
from rk4_defect import local_defects
from state_trial import build_model


def tensor_hashes(model: torch.nn.Module) -> dict[str,str]:
    items = list(model.named_parameters())+list(model.named_buffers())
    return {name:hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name,value in items}


def close(actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> float:
    error = (actual-expected).abs()
    assert bool(torch.all(error<=atol+rtol*expected.abs()))
    return float(error.max())


def fixture(dim: int) -> dict:
    torch.manual_seed(0)
    model = build_model(dim,'state_midpoint4').double().eval()
    with torch.no_grad():
        values = .02*torch.sin(torch.arange(model.output.weight.numel(),dtype=torch.float64))
        model.output.weight.copy_(values.reshape_as(model.output.weight))
    x = .7*torch.cos(torch.arange(8*dim,dtype=torch.float64).reshape(8,dim)+.1)
    time = torch.linspace(.1,.9,8,dtype=torch.float64)[:,None]
    hashes = tensor_hashes(model)
    solves = [layer.solve.__func__ for _,layer in model.ode_layers()]
    rng = torch.get_rng_state().clone()
    with torch.no_grad():native_before = model(x,time)

    # Independent polynomial-jet construction on fixed states and an actual P.
    layer = next(model.ode_layers())[1]
    y = torch.cos(torch.arange(8*64,dtype=torch.float64).reshape(8,64))
    matrix = layer.operator().detach()
    reference = local_defects(y.numpy(),matrix.numpy())
    actual = defects(y,matrix)
    jet_errors = {name:float(np.max(np.abs(actual[name].detach().numpy()-reference[name])))
                  for name in ('midpoint','rk4')}
    assert all(error<=1e-12 for error in jet_errors.values())

    refinements = {}
    for steps in (16,32,64):
        refinements[steps] = model_response(model,x,time,steps,'midpoint','rk4')
    primal,tangent = refinements[16]
    assert float(torch.linalg.vector_norm(tangent))>1e-10
    epsilon = torch.tensor(0.,dtype=torch.float64)
    with torch.no_grad():
        jvp_primal,jvp_tangent = torch.func.jvp(
            lambda e:forced_model(model,x,time,16,e,'midpoint','rk4'),
            (epsilon,),(torch.ones_like(epsilon),))
    jvp_errors = dict(primal=close(primal,jvp_primal,1e-11,1e-11),
                      tangent=close(tangent,jvp_tangent,1e-11,1e-11))
    finite = []
    with torch.no_grad():
        for epsilon_value in (1e-4,1e-5):
            e = torch.tensor(epsilon_value,dtype=torch.float64)
            plus = forced_model(model,x,time,16,e,'midpoint','rk4')
            minus = forced_model(model,x,time,16,-e,'midpoint','rk4')
            fd = (plus-minus)/(2*e)
            error = close(tangent,fd,2e-7,2e-4)
            finite.append(dict(epsilon=epsilon_value,max_error=error))
    zero_primal,zero_tangent = model_response(model,x,time,16,'midpoint','midpoint')
    assert not torch.count_nonzero(zero_tangent)
    reverse_primal,reverse_tangent = model_response(model,x,time,16,'rk4','midpoint')
    zero_primal_error = close(zero_primal,primal,1e-12,0.)
    reverse_errors = dict(primal=close(reverse_primal,primal,1e-12,0.),
                          tangent=close(reverse_tangent,-tangent,1e-12,0.))
    convergence = []
    for old,new in ((16,32),(32,64)):
        p0,z0 = refinements[old];p1,z1 = refinements[new]
        convergence.append(dict(old=old,new=new,primal=float((p1-p0).abs().max()),
                                 tangent=float((z1-z0).abs().max())))
    for key,limit in (('primal',1e-7),('tangent',1e-8)):
        before,after = (row[key] for row in convergence)
        assert after<=limit and (after<before or max(before,after)<=1e-12)
    assert hashes == tensor_hashes(model)
    assert solves == [layer.solve.__func__ for _,layer in model.ode_layers()]
    assert torch.equal(rng,torch.get_rng_state())
    with torch.no_grad():assert torch.equal(native_before,model(x,time))
    return dict(dim=dim,width=64,depth=3,local_flows=6,points=8,seed=0,
        fixture_head='0.02*sin(arange(numel)); original bias retained',
        polynomial_jet_errors=jet_errors,whole_model_jvp_errors=jvp_errors,
        finite_differences=finite,nonzero_tangent_norm=float(torch.linalg.vector_norm(tangent)),
        zero_forcing_exact=True,zero_primal_error=zero_primal_error,
        sign_reversal_errors=reverse_errors,refinements=convergence,
        tensors_methods_rng_unchanged=True,native_replay_exact=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(1)
    rows = []
    for dim in (2,16):
        rows.append(fixture(dim))
        print('HIDDEN_DEFECT_FIXTURE_PASS',dim,flush=True)
    result = dict(passed=True,scope='untrained deterministic implementation controls only',
        learned_checkpoints_used=0,data_banks_used=0,training_runs=0,rows=rows,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        observer_sha256=hashlib.sha256((Path(__file__).parent/'hidden_defect_transport.py').read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('HIDDEN_DEFECT_FULL_MODEL_CONTROLS_PASS',flush=True)


if __name__ == '__main__':main()
