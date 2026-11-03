"""Count executed dense matrix MACs; excludes nonlinear and scalar arithmetic."""
import argparse
import json
from pathlib import Path
from audit import base, build_model, torch
from ordinary import FreeISRU


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    base.verify_sources()
    rows=[]
    for dim in [2,16]:
        for family in ['pd_isru_k1','free_isru','swiglu64','swiglu192']:
            torch.manual_seed(0)
            model=build_model(dim,family).eval()
            calls=[];handles=[]
            def hook(module,args,output):
                assert args[0].ndim==2 and args[0].shape[0]==1
                calls.append(module.weight.numel())
            for module in model.modules():
                if isinstance(module,torch.nn.Linear):handles.append(module.register_forward_hook(hook))
            with torch.no_grad():model(torch.full((1,dim),.25),torch.full((1,1),.4))
            for h in handles:h.remove()
            # Native solves call F.linear on P directly, and the copied ordinary
            # control calls it on its two Parameter matrices; Linear hooks cannot
            # observe these. Each used exactly once under the registered K1.
            direct=sum(layer.proj.weight.numel()*layer.orders for _,layer in model.ode_layers())
            direct+=sum(module.matrix1.numel()+module.matrix2.numel() for module in model.modules() if isinstance(module,FreeISRU))
            rows.append(dict(dim=dim,family=family,parameters=sum(p.numel() for p in model.parameters()),
                linear_hook_macs=sum(calls),direct_parameter_macs=direct,dense_macs=sum(calls)+direct))
    reference={r['family']:r for r in rows if r['dim']==16}
    assert reference['pd_isru_k1']['dense_macs']==reference['free_isru']['dense_macs']==57408
    assert reference['swiglu64']['dense_macs']==57024
    result=dict(passed=True,device='cpu',scope='Executed dense matrix MACs per single vector-field forward; nonlinear/scalar work excluded.',
        rows=rows,source_sha256=base.digest(__file__),model_source_sha256=base.digest(Path(__file__).resolve().parents[1]/'cycle13/trial13.py'))
    base.dump(a.output,result)
    print(json.dumps(result))


if __name__=='__main__':main()
