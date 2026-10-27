"""Reproduce the audit's nested-LayerNorm derivative discrepancy in float64."""
from pathlib import Path
import argparse
import ast
import json
import sys

import torch
from torch import nn
from torch.nn import functional as F

parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();sys.path.insert(0,str(args.root/'source'))
import run_cycle as study
settings,build,cost,restore,legacy=study.setup(args.root)
source=(Path(__file__).parent/'audit_and_analyze.py').read_text()
nodes=ast.parse(source).body
selected=[n for n in nodes if (isinstance(n,ast.FunctionDef) and n.name=='manual_norm') or isinstance(n,ast.ClassDef)]
exec(compile(ast.Module(body=selected,type_ignores=[]),'elementary_norm_independent','exec'))
Elementary=Independent
native_source=source.replace('manual_norm(b.norm,state)','b.norm(state)').replace('manual_norm(b.norm,h)','b.norm(h)')
selected=[n for n in ast.parse(native_source).body if isinstance(n,ast.ClassDef)]
exec(compile(ast.Module(body=selected,type_ignores=[]),'fused_norm_independent','exec'))
Fused=Independent
model=study.restored(build,restore,settings['cohort'][0],5000).double()
x=torch.tensor([[.3,.7],[-.2,.4]],dtype=torch.float64);t=torch.full((2,1),.3,dtype=torch.float64)
rows=[]
for method in ('taylor','frozen_gate'):
    analytic=study.Intervention(model,method,'all')
    elementary=Elementary(model,method,'all');fused=Fused(model,method,'all')
    value,jac=legacy.jacobian(analytic,x,t)
    ev,ej=legacy.jacobian(elementary,x,t);fv,fj=legacy.jacobian(fused,x,t)
    finite=[]
    epsilon=1e-6
    with torch.no_grad():
        for j in range(2):
            plus=x.clone();minus=x.clone();plus[:,j]+=epsilon;minus[:,j]-=epsilon
            finite.append((analytic(plus,t)-analytic(minus,t))/(2*epsilon))
    finite=torch.stack(finite,dim=-1)
    assert float((ej-jac).abs().max())<1e-10
    assert float((finite-jac).abs().max())<1e-7
    rows.append(dict(method=method,elementary_value_max=float((ev-value).abs().max()),
        fused_value_max=float((fv-value).abs().max()),elementary_jacobian_max=float((ej-jac).abs().max()),
        fused_jacobian_max=float((fj-jac).abs().max()),central_difference_max=float((finite-jac).abs().max())))
result=dict(torch_version=torch.__version__,dtype='float64',x=x.tolist(),t=t.tolist(),
    finite_difference_epsilon=epsilon,rows=rows,scope='Installed local primitive path and fixed native checkpoint only.')
study.write(args.output,result);print(json.dumps(result,indent=2))
