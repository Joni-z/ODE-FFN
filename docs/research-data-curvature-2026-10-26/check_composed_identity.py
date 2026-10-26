"""Verify the post-result complete-prefix corollary without changing weights."""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT,DEST=map(Path,sys.argv[1:3])
sys.path.insert(0,str(ROOT/'source'))
import curvature_study as study
from curvature_math import local,left,frame,reduced
from teacher_support import TEACHERS,STUDENTS,TIMES,load,write,sha

torch.set_num_threads(1)
assert load(ROOT/'audit.json')['status']=='PASS'
with np.load(ROOT/'output/cache/prefix.npz') as f:bank={k:f[k] for k in f.files}
checks,projected=[],[]
for teacher in TEACHERS:
    for position in range(3):
        folder=ROOT/'output/cases'/f'{teacher}--{position}'
        models=study.subjects(teacher,position)
        arrays={}
        for key in ['target',*STUDENTS]:
            with np.load(folder/(key+'.npz')) as f:arrays[key]={k:f[k] for k in f.files}
        for key,(module,family) in models.items():
            if family!='ode_k2':continue
            for anchor in range(80):
                z,u,hz=(torch.from_numpy(bank[k][anchor,position]) for k in ('z','jacobian','hessian'))
                with torch.no_grad():
                    data=local(module,z,u,family);definition=frame(data,u)
                    frozen=data['frozen_hessian']+left(data['frozen_jac'],hz)
                    actual=torch.from_numpy(arrays[key]['hessian_total'][anchor])
                    difference=reduced(actual-frozen,definition)
                native_j=torch.func.jacrev(lambda state:module(state[None])[0])(z).detach()
                delta=native_j-data['frozen_jac']
                singular=torch.linalg.svdvals(delta)
                b=definition['output_basis']
                outside=delta-b@(b.T@delta)
                torch.testing.assert_close(difference,torch.zeros_like(difference),rtol=2e-8,atol=5e-10)
                torch.testing.assert_close(outside,torch.zeros_like(outside),rtol=2e-8,atol=5e-10)
                checks.append(dict(teacher=teacher,position=position,subject=key,anchor=anchor,time=float(bank['time'][anchor]),
                    composed_identity_max_abs=float(difference.abs().max()),jacobian_projection_max_abs=float(outside.abs().max()),
                    third_jacobian_difference_singular_value=float(singular[2])))
                if key=='ode_k2':
                    truth=reduced(torch.from_numpy(arrays['target']['hessian_total'][anchor]),definition)
                    row=dict(teacher=teacher,position=position,anchor=anchor,time=float(bank['time'][anchor]),
                             target_retained_energy=float(truth.square().sum()),
                             target_full_energy=float(np.sum(arrays['target']['hessian_total'][anchor]**2)))
                    for family2 in STUDENTS:
                        value=reduced(torch.from_numpy(arrays[family2]['hessian_total'][anchor]),definition)
                        row[family2+'_error_energy']=float((value-truth).square().sum())
                    projected.append(row)

def csv_file(name,rows):
    with (DEST/name).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

rows,aggregate=[],[]
for teacher in TEACHERS:
    for position in range(3):
        for family in STUDENTS:
            for time in [None,*TIMES]:
                selected=[r for r in projected if r['teacher']==teacher and r['position']==position and (time is None or r['time']==time)]
                energy=sum(r['target_retained_energy'] for r in selected);full=sum(r['target_full_energy'] for r in selected)
                error=sum(r[family+'_error_energy'] for r in selected)
                row=dict(teacher=teacher,position=position,student=family,target_retained_energy=energy,target_full_energy=full,
                         target_retained_fraction=energy/full,error_energy=error,relative_squared_error=error/energy)
                if time is None:aggregate.append(row)
                else:rows.append(dict(time=time,**row))
csv_file('composed-identity-checks.csv',checks)
csv_file('reduced-composed.csv',rows)
csv_file('reduced-composed-aggregate.csv',aggregate)
receipt=dict(status='PASS',identities=len(checks),native_jacobian_checks=len(checks),
             maximum_composed_hessian_difference=max(r['composed_identity_max_abs'] for r in checks),
             maximum_jacobian_projection_difference=max(r['jacobian_projection_max_abs'] for r in checks),
             maximum_third_jacobian_difference_singular_value=max(r['third_jacobian_difference_singular_value'] for r in checks),
             source_sha256=sha(__file__),audit_sha256=sha(ROOT/'audit.json'))
write(DEST/'composed-identity-receipt.json',receipt)
print(receipt)
for teacher in TEACHERS:
    for position in range(3):
        selected=[r for r in aggregate if r['teacher']==teacher and r['position']==position]
        print(teacher,position+1,'teacher retained',selected[0]['target_retained_fraction'],
              'RMS errors %',[round(100*math.sqrt(r['relative_squared_error']),3) for r in selected])
