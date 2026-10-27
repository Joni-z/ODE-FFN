"""Reporting-only conditional Monte Carlo precision; no selection tests."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--results',type=Path,required=True);args = parser.parse_args()
import json
settings = json.loads((args.root/'settings.json').read_text());basis = np.load(Path(settings['parent'])/'basis.npy')
rows = []
with (args.results/'endpoints.csv').open() as stream:endpoints = list(csv.DictReader(stream))
for record in endpoints:
    law,family = record['law'],record['family']
    with np.load(args.root/'output/raw'/f'{law}-{family}.npz') as data:
        error = data['bridges'][-1].astype(np.float64)-data['bridges'][0]
        projector = data['forced_projector']
    plane = error@basis;normal = error-plane@basis.T
    for scope,value,dimension in [('full',error,128),('tangent',plane,2),('normal',normal,126)]:
        samples = np.sum(value*value,axis=1)/dimension;mean = float(samples.mean());sd = float(samples.std(ddof=1));se = sd/np.sqrt(len(samples))
        np.testing.assert_allclose(mean,float(record[scope+'_paired_endpoint_mse']),rtol=1e-12,atol=1e-12)
        rows.append(dict(law=law,family=family,quantity='paired_endpoint_mse',scope=scope,count=len(samples),
            mean=mean,source_sd=sd,standard_error=se,lower95=mean-1.959963984540054*se,
            upper95=mean+1.959963984540054*se,interval='pointwise_CLT_fixed_predictors'))
    numerator = np.sum((error@projector)**2,axis=1)/126;denominator = np.sum(normal**2,axis=1)/126
    ratio = float(numerator.mean()/denominator.mean());influence = (numerator-ratio*denominator)/denominator.mean()
    se = float(influence.std(ddof=1)/np.sqrt(len(error)))
    np.testing.assert_allclose(ratio,float(record['forced_fraction_of_normal_paired_error']),rtol=1e-12,atol=1e-12)
    rows.append(dict(law=law,family=family,quantity='forced_fraction_of_normal_error',scope='normal',count=len(error),
        mean=ratio,source_sd=None,standard_error=se,lower95=ratio-1.959963984540054*se,
        upper95=ratio+1.959963984540054*se,interval='pointwise_delta_fixed_predictors'))
assert len(rows)==48
with (args.results/'conditional-mc.csv').open('w',newline='') as stream:
    writer = csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print('48 reporting-only conditional Monte Carlo summaries; no hypothesis/selection tests')
