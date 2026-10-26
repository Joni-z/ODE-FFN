"""Check the explicitly scoped origin jet identity against native derivatives."""
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT, DOCS = map(Path, sys.argv[1:3])
sys.path.insert(0, str(ROOT/'source'))
import local_study as study
from teacher_support import teacher

torch.set_num_threads(1)
subjects = []
for j in range(3):
    subjects.append((f'ODE-teacher-{j}', teacher('ode_k2')[0].raw.blocks[j].ffn))
for folder in sorted((ROOT/'output/runs').glob('*--ode_k2--*')):
    result = json.loads((folder/'result.json').read_text())
    module = study.cold('ode_k2', result['position'])
    module.load_state_dict(torch.load(folder/'step_0005000.pt', map_location='cpu', weights_only=False)['model'])
    subjects.append((folder.name, module))
eye = torch.eye(64, dtype=torch.float64)
index = torch.arange(1,65,dtype=torch.float64)
unit = lambda x:x/x.norm()
pairs = [(eye[0],eye[1]),(eye[0],eye[0]),(unit(torch.sin(index)),unit(torch.cos(index))),
         (unit(index),unit(index.flip(0)))]
rows = []
for name, subject in subjects:
    module = copy.deepcopy(subject).double()
    zero = torch.zeros(64,dtype=torch.float64)
    data = []
    for layer in (module.ode1,module.ode2):
        duration = lambda h:layer.step_size(h[None]).squeeze()
        a = duration(zero).detach()
        g = torch.func.grad(duration)(zero).detach()
        p = layer.proj.weight.detach()
        data.append((eye+a*p+.5*a*a*(p@p),p+a*(p@p),g))
    a,c1,g1 = data[0]
    b,c2,g2 = data[1]
    field = lambda h:module(h[None])[0]
    assert torch.equal(field(zero).detach(),zero)
    jac = torch.func.jacrev(field)(zero).detach()
    j_error = float((jac-.5*b@a).abs().max())
    assert j_error < 1e-10
    for k,(u,v) in enumerate(pairs):
        derivative_v = lambda x:torch.func.jvp(field,(x,),(v,))[1]
        actual = torch.func.jvp(derivative_v,(zero,),(u,))[1].detach()
        au,av = a@u,a@v
        explicit = .5*b@(au*av)
        explicit += .5*b@(c1@u*torch.dot(g1,v)+c1@v*torch.dot(g1,u))
        explicit += .25*(c2@au*torch.dot(g2,av)+c2@av*torch.dot(g2,au))
        difference = float((actual-explicit).abs().max())
        assert difference < 1e-10
        rows.append(dict(subject=name,pair=k,jacobian_max_abs=j_error,hessian_directional_max_abs=difference))
receipt = dict(status='PASS',subjects=len(subjects),directional_checks=len(rows),
               maximum_jacobian_difference=max(r['jacobian_max_abs'] for r in rows),
               maximum_hessian_difference=max(r['hessian_directional_max_abs'] for r in rows),
               scope='Origin only. No claim about proximity of actual normalized inputs or approximation lower bounds.',
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows)
(DOCS/'origin-identity-check.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({key:value for key,value in receipt.items() if key!='rows'},indent=2))
