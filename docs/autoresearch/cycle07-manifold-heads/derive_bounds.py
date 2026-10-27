"""Integrate the proved population bounds; these are not finite-bank floors."""
import argparse
import csv
from pathlib import Path
import sys

import numpy as np

parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
sys.path.insert(0,str(args.root/'source'));import benchmark_core as core
rows=[]
for law,variance in [('thin',0.),('thick',.04)]:
    estimates=[]
    for order in (64,128):
        nodes,weights=np.polynomial.legendre.leggauss(order);values={}
        for low,high in [(0.,.95),(.95,1.)]:
            t=low+(nodes+1)*(high-low)/2;bounds=core.bottleneck_bounds(t[:,None],variance)
            for key,value in bounds.items():values[key]=values.get(key,0.)+float(weights@value[:,0]*(high-low)/2)
        estimates.append(values)
    for key in estimates[0]:np.testing.assert_allclose(estimates[0][key],estimates[1][key],rtol=1e-11,atol=1e-12)
    rows.append(dict(law=law,normal_variance=variance,**{f'uniform_time_average_{key}':value for key,value in estimates[-1].items()},population_not_finite_bank=True))
with args.output.open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(rows)
