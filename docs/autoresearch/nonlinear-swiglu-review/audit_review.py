"""Verify and curate the completed, data-free candidate eligibility review."""
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import torch

repo = Path(__file__).resolve().parents[3]
doc = Path(__file__).resolve().parent
root = Path('/Users/mr.z/ode-workspace-prep/autoresearch/oct28-nonlinear-swiglu-review')
load = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
for name, digest in load(root/'bindings.json').items():
    assert sha(Path(name)) == digest, name
assert sha(doc/'check_candidate.py') == sha(root/'source/check_candidate.py')
assert sha(doc/'protocol.md') == sha(root/'source/protocol.md')
run = load(root/'verification.json')
assert run['status'] == 'PASS' and not run['new_training'] and not run['training_nominated']
rows = load(root/'derivative-checks.json')
assert len(rows) == 18
assert {(r['dim'],r['actual_hidden']) for r in rows} == {(1,1),(4,5),(64,85)}
assert all(set(v['step'] for v in r['finite_differences']) == {1e-5,5e-6} for r in rows)
assert max(v for r in rows for v in r['errors'].values()) <= 1e-10
assert max(v['normalized_error'] for r in rows for v in r['finite_differences']) <= 1e-5
costs = load(root/'dense-work.json')
for row in costs:
    d, r = row['dim'], row['actual_hidden']
    assert row['parameters'] == 3*d*r+2*r+d
    split = row['forward_call_count']
    assert sum(v['mac'] for v in row['dense_calls'][:split]) == 8*row['forward_MAC_per_row']
    assert sum(v['mac'] for v in row['dense_calls'][split:]) == 8*row['backward_MAC_per_row']
    assert row['total_training_MAC_per_row'] == 3*row['forward_MAC_per_row']
    for call in row['dense_calls']:
        a,b = call['a'],call['b']
        assert len(a) == len(b) == 2 and a[1] == b[0]
        assert call['mac'] == a[0]*a[1]*b[1]
expected = {'native':16320,'taylor2':32640,'midpoint':32640,
            'wide_actual170':32640,'frozen_gate':27200}
assert {r['arm']:r['forward_MAC_per_row'] for r in costs} == expected
sys.path.insert(0,str(repo))
from synthetic.models import SwiGLUFFN
model = SwiGLUFFN(1,2).double()
with torch.no_grad():
    model.w12.weight.fill_(1)
    model.w3.weight.fill_(1)
    model.w12.bias.zero_()
    model.w3.bias.zero_()
y = torch.tensor([[.1],[1.],[2.],[4.]],dtype=torch.float64)
torch.testing.assert_close(model(y),y*y*torch.sigmoid(y),rtol=1e-15,atol=1e-15)
escape = load(root/'finite-escape.json')
assert [r['y0'] for r in escape] == [1.,2.,4.]
for row in escape:
    a,b,c = (row[k] for k in ('quadrature_original','quadrature_inverse','exponential_integral_identity'))
    assert max(abs(a-b),abs(a-c)) <= 1e-10
    assert row['lower_bound'] < c < row['upper_bound']
assert escape[1]['upper_bound'] < 1
protected = ['src','synthetic','configs','scripts','reports/report1']
assert not subprocess.check_output(['git','-C',str(repo),'diff','--name-only','98730e0','--',*protected],text=True)
for path in doc.glob('*.py'):
    ast.parse(path.read_text())
result = doc/'results'
result.mkdir(exist_ok=True)
links = 0
for path in doc.glob('*.md'):
    for name in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',path.read_text()):
        if '://' in name:
            continue
        assert (path.parent/name.split('#')[0]).exists(),(path,name)
        links += 1
for name in ('bindings.json','execution.json','derivative-checks.json','dense-work.json',
             'finite-escape.json','verification.json'):
    shutil.copyfile(root/name,result/name)
assert all(sha(result/p.name)==sha(p) for p in root.glob('*.json') if (result/p.name).exists())
receipt = dict(status='PASS',source_commit=load(root/'execution.json')['source_commit'],
               registered_source_unchanged=True,native_scalar_counterexample_verified=True,
               inspected_matrix_calls=True,derivative_cases=18,finite_difference_checks=36,
               unchanged_scopes=protected,local_links=links,new_training=False,
               training_nominated=False,quality_improvement_demonstrated=False,
               numerical_process_exit_code=0)
(result/'artifact-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
closure = root/'closure'
closure.mkdir(exist_ok=False)
for file in doc.iterdir():
    if file.is_file():
        shutil.copyfile(file,closure/file.name)
shutil.copyfile(result/'artifact-verification.json',root/'artifact-verification.json')
shutil.copyfile('/Users/mr.z/ode-workspace-prep/nonlinear-swiglu-review.log',root/'run.log')
files = sorted(p for p in root.rglob('*') if p.is_file() and p.name!='sealed-runtime.json')
seal = {str(p.relative_to(root)):sha(p) for p in files}
(root/'sealed-runtime.json').write_text(json.dumps(seal,indent=2)+'\n')
assert all(sha(root/name)==digest for name,digest in load(root/'sealed-runtime.json').items())
print(json.dumps(dict(status='PASS',files=len(files),seal_sha256=sha(root/'sealed-runtime.json'),
                     new_training=False,training_nominated=False)))
