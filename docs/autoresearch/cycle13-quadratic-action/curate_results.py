"""Audited copies and reporting-only tangent/normal split of the same intervention."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np

parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--results',type=Path,required=True);args=parser.parse_args()
root=args.root;out=args.results;out.mkdir(exist_ok=True)
load=lambda p:json.loads(p.read_text())
assert load(root/'audit/audit.json')['status']=='PASS'
for p in (root/'audit').iterdir():
    if p.is_file(): shutil.copyfile(p,out/p.name)
parent=Path(load(root/'settings.json')['parent']);frame=Path(load(parent/'settings.json')['parent'])
settings=load(frame/'settings.json');basis=np.load(frame/'basis.npy')
with (out/'complete.csv').open() as stream: complete=list(csv.DictReader(stream))
rows=[]
for entry in settings['laws']:
    with np.load(Path(entry['banks'])/'field.npz') as bank: truth=bank['oracle']
    for record in [r for r in complete if r['law']==entry['law']]:
        with np.load(root/'output/raw'/f"{entry['law']}-{record['family']}-{int(record['step']):07d}-predictions.npz") as data:
            native=data['native'].astype(float);changed=data['changed'].astype(float)
        def parts(v):
            plane=v@basis
            return (v,plane,v-plane@basis.T)
        for scope,dimension,old,new,target in zip(('full','tangent','normal'),(128,2,126),parts(native),parts(changed),parts(truth)):
            error=old-target;delta=new-old
            energy=lambda v:float(np.sum(v*v)/(len(v)*dimension))
            a=energy(error);b=energy(new-target);d=energy(delta);cross=float(2*np.sum(error*delta)/(len(delta)*dimension))
            np.testing.assert_allclose(b-a,d+cross,atol=1e-10,rtol=1e-8)
            if scope=='full':
                np.testing.assert_allclose(a,float(record['original_field_mse']),atol=1e-12,rtol=1e-12)
                np.testing.assert_allclose(b,float(record['changed_field_mse']),atol=1e-12,rtol=1e-12)
            base=energy(old)
            rows.append(dict(law=entry['law'],family=record['family'],step=int(record['step']),scope=scope,
                dimension=dimension,count=len(old),original_mse=a,changed_mse=b,
                relative_risk_change=(b-a)/a,displacement_mse=d,cross_term=cross,
                relative_displacement_rms=float(np.sqrt(d/base)) if base>0 else None,
                registration_status='registered_full' if scope=='full' else 'postregistration_reporting_split'))
assert len(rows)==72
with (out/'intervention-parts.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print('Copied audited evidence and verified 72 reporting decomposition rows; no new intervention')
