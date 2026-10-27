"""Descriptive trained fixed-dictionary operator ranks; no model selection."""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch

parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();root=args.root;out=args.output;out.mkdir(exist_ok=False);sys.path.insert(0,str(root/'source'))
import study
settings,native,_,_=study.setup(root);study.verify_inputs(root,native);rows=[]
for entry in settings['laws']:
    model=study.build(root,entry['law'],'transport')
    native.restore_model(model,torch.load(root/'output/runs'/entry['law']/'transport/step_0005000.pt',weights_only=False))
    for index,block in enumerate(model.raw.blocks):
        module=block.ffn;u=module.w3.weight.detach().double().numpy()
        v=module.w12.weight.detach().double().numpy()[module.w3.in_features:]
        c=module.transport.detach().double().numpy();r=module.w3.in_features;m=module.left_width
        mix=np.eye(r);mix[:m,m:]+=.5*c
        for j in range(r):
            coefficient=(u*mix[:,j])@v
            singular=np.linalg.svd(coefficient,compute_uv=False)
            energy=np.sum(singular**2)
            effective=energy**2/max(float(np.sum(singular**4)),1e-300)
            numerical=int(np.count_nonzero(singular>1e-8*singular[0])) if singular[0] else 0
            if j<m:assert numerical==1 and abs(effective-1)<1e-10
            assert numerical<=m+1
            rows.append(dict(law=entry['law'],layer=index,gate=j,role='receiver' if j<m else 'source',
                numerical_rank=numerical,rank_tolerance_relative=1e-8,participation_rank=float(effective),
                coefficient_frobenius=float(np.sqrt(energy)),linear_part_only=True))
with (out/'gate-coefficient-ranks.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
summary=[]
for entry in settings['laws']:
    values=[row for row in rows if row['law']==entry['law'] and row['role']=='source']
    summary.append(dict(law=entry['law'],source_gates=len(values),numerical_rank_min=min(row['numerical_rank'] for row in values),
        numerical_rank_max=max(row['numerical_rank'] for row in values),
        participation_rank_median=float(np.median([row['participation_rank'] for row in values])),
        participation_rank_max=max(row['participation_rank'] for row in values)))
study.write(out/'summary.json',dict(post_outcome_descriptive=True,fixed_dictionary_not_network_rank=True,by_law=summary))
study.verify_inputs(root,native);print(summary)
