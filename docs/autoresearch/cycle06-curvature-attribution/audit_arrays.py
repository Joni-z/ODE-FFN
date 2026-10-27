"""Independent NumPy verification of the saved attribution and panel arrays."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
root=args.root.resolve();out=root/'output';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert load(out/'status.json')['state']=='COMPLETE'
base=Path(load(root/'settings.json')['cycle05']);config=load(base/'settings.json')
for name,digest in load(root/'bindings.json').items():assert sha(Path(name))==digest,name
for name,digest in load(base/'sealed-runtime.json').items():assert sha(base/name)==digest,name
read=lambda name:list(csv.DictReader((out/name).open()))
fields=read('field-attribution.csv');panels=read('time-panels.csv');cases=read('cases.csv');tangents=read('tangent-identities.csv')
assert (len(fields),len(panels),len(cases),len(tangents))==(5,25,15,20)
largest=0.;checked=0
for entry in config['laws']:
    law=entry['law'];banks=Path(entry['banks']);row=next(r for r in fields if r['law']==law)
    with np.load(out/f'{law}-actual.npz') as data:q=data['prediction'].astype(np.float64)
    with np.load(out/f'{law}-midpoint.npz') as data:m=data['prediction'].astype(np.float64)
    with np.load(banks/'field.npz') as data:y=data['oracle'].astype(np.float64);time=data['t']
    assert len(q)==int(row['rows']);checked+=len(q)
    residual=m-y;movement=q-m
    alignment=2*np.einsum('nd,nd->',residual,movement)/residual.size
    energy=np.einsum('nd,nd->',movement,movement)/movement.size
    aq=np.einsum('nd,nd->',q-y,q-y)/q.size;am=np.einsum('nd,nd->',residual,residual)/residual.size
    for key,value in [('actual_MSE',aq),('midpoint_counterfactual_MSE',am),('alignment_cross_term',alignment),('displacement_energy',energy),('actual_minus_counterfactual_MSE',aq-am)]:
        np.testing.assert_allclose(value,float(row[key]),rtol=1e-10,atol=1e-12)
    largest=max(largest,abs(aq-am-alignment-energy))
    changes=((q-y)**2).mean(1)-((m-y)**2).mean(1);selected=np.lexsort((np.arange(len(changes)),-np.abs(changes)))[:3]
    for rank,index in enumerate(selected):
        case=next(r for r in cases if r['law']==law and int(r['rank_by_absolute_error_change'])==rank+1)
        assert int(case['row'])==index;assert float(case['time'])==float(time[index,0])
        np.testing.assert_allclose(float(case['actual_minus_counterfactual_MSE']),changes[index],rtol=1e-10,atol=1e-12)
    for clock in (.1,.3,.5,.7,.9):
        panel=next(r for r in panels if r['law']==law and abs(float(r['time'])-clock)<1e-8)
        with np.load(banks/f'panel_{clock:.1f}.npz') as data:truth=data['oracle_jacobian']
        with np.load(out/f'{law}-midpoint-panel-{clock:.1f}.npz') as data:jac=data['jacobian']
        assert len(jac)==int(panel['anchors'])==512
        value=float(np.mean(np.sum((jac-truth)**2,axis=(1,2))))
        np.testing.assert_allclose(value,float(panel['midpoint_counterfactual_J_error']),rtol=1e-10,atol=1e-12)
assert checked==80000
assert all(float(r['C_tangent_rms'])>1e-10 and float(r['max_absolute_difference'])<1e-10 for r in tangents)
result=dict(status='PASS',scope='saved-array decomposition, selected-case and counterfactual-Jacobian metrics; not an independent second model implementation',
    field_rows=checked,panels=25,cases=15,tangent_receipts=20,max_decomposition_absolute_error=largest)
(root/'array-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
