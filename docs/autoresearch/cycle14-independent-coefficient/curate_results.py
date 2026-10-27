"""Publish audited summaries and exact wide spectra/windowed telemetry."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--results',type=Path,required=True);args=parser.parse_args()
root=args.root;out=args.results;out.mkdir(exist_ok=True)
load=lambda p:json.loads(p.read_text())
assert load(root/'audit/audit.json')['status']=='PASS'
for p in (root/'audit').iterdir():
    if p.is_file():shutil.copyfile(p,out/p.name)
settings=load(root/'settings.json');reference=Path(settings['geometry_reference'])
with (out/'comparison.csv').open() as stream:comparison=list(csv.DictReader(stream))
spectra=[];telemetry=[]
for record in comparison:
    law,family=record['law'],record['family'];fresh=record['new_training']=='True'
    folder=(root if fresh else reference)/'output/runs'/law/(family+'_clean')
    for line in (folder/'metrics.jsonl').read_text().splitlines():
        row=json.loads(line)
        telemetry.append(dict(law=law,family=family,new_training=fresh,step=row['step'],
            **{k:row[k] for k in ('train_loss','grad_norm','clip_fraction','oracle_mse',
                'tangent_mse_per_dimension','normal_mse_per_dimension') if k in row}))
    if fresh and family!='swiglu258':
        for step in (0,1000,2500,5000):
            for row in load(folder/f'generator_{step:07d}.json'):
                result=dict(law=law,family=family,step=step,layer=row['layer'],coefficient_mode=row['coefficient_mode'])
                for prefix,key in [('P','P_singular_values'),('reference_B','reference_B_singular_values'),('C','C_singular_values')]:
                    result.update({f'{prefix}_sigma_{i:02d}':v for i,v in enumerate(row[key])})
                spectra.append(result)
assert len(spectra)==192 and len(telemetry)==154
def table(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
table(out/'coefficient-singular-values.csv',spectra);table(out/'training-telemetry.csv',telemetry)

# Existing stronger neighboring-width reference: context only, never a new selection rule.
with (Path(settings['parent_cycle11'])/'audit-v2/comparison.csv').open() as stream:
    context=[r for r in csv.DictReader(stream) if r['family']=='swiglu257']
assert len(context)==2
for row in context:
    row['new_training_in_cycle11']=row.pop('new_training')
    row.update(new_training_in_cycle14=False,registered_cycle14_reference=False,
               provenance='sealed cycle11 audit-v2; same geometry banks; context only')
table(out/'historical-swi257-context.csv',context)
print('192 wide P/B-reference/C spectra and 154 exact windowed training rows; historical Swi257 is context only')
