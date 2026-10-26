"""Verify descriptive code-membership diagnostics on all saved generated samples."""
from pathlib import Path
import argparse
import itertools
import json
import sys

import numpy as np

parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
root=args.root.resolve();sys.path.insert(0,str(root/'source'))
import joint_study as study
study.setup(root);target=study.distribution(root)
quadruples=np.array([q for q in itertools.combinations(range(16),4) if q[0]^q[1]^q[2]^q[3]==0]);assert len(quadruples)==140
valid_integers=np.sum((target.signs>0).astype(np.int64)*(1<<np.arange(16)),axis=1)


def metrics(x):
    q=x.astype(np.float64)@target.rotation;sign=np.where(q>=0,1.,-1.)
    hamming=np.sum(sign[:,None]!=target.signs[None],axis=2).min(1)
    valid=hamming==0
    integer=np.sum((q>=0).astype(np.int64)*(1<<np.arange(16)),axis=1)
    np.testing.assert_array_equal(valid,np.isin(integer,valid_integers))
    distance=np.sum((q[:,None]-target.means[None])**2,axis=2)
    independent=np.sum(q**2,axis=1)[:,None]+np.sum(target.means**2,axis=1)-2*q@target.means.T
    np.testing.assert_allclose(distance,independent,rtol=1e-9,atol=1e-10)
    nearest=distance.argmin(1);frequency=np.bincount(nearest,minlength=32)/len(nearest)
    row=dict(valid_code_fraction=float(valid.mean()),mean_four_sign_correlation=float(np.prod(sign[:,quadruples],axis=-1).mean()),
        nearest_mean_mse=float(distance.min(1).mean()/16),mode_count=int((frequency>0).sum()),mode_frequency_tv=float(np.abs(frequency-1/32).sum()/2))
    details=dict(hamming_histogram=np.bincount(hamming,minlength=17).tolist(),mode_frequencies=frequency.tolist())
    bad=np.flatnonzero(~valid)
    if len(bad):
        k=int(bad[0]);details['first_invalid_example']=dict(source_index=k,normalized_latent=(q[k]/target.locations).tolist(),
            sign_pattern=sign[k].astype(int).tolist(),nearest_mean_index=int(nearest[k]),
            nearest_mean_sign=target.signs[nearest[k]].astype(int).tolist(),minimum_hamming=int(hamming[k]))
    return row,details


rows=[];details=[]
for family in study.FAMILIES:
    with np.load(root/'output/runs'/family/'diagnostics/samples.npz') as data:row,detail=metrics(data['generated'])
    rows.append(dict(family=family,**row));details.append(dict(family=family,**detail))
with np.load(root/'banks/sampling.npz') as data:
    for name in ['target','target2']:
        row,detail=metrics(data[name]);rows.append(dict(family=name,**row));details.append(dict(family=name,**detail))
summary=root/'joint-sample-case-study.json'
if summary.exists():
    previous=study.load(summary)
    for old,new in zip(previous['results'],rows):
        assert old.keys()==new.keys()
        for key in old:
            if isinstance(old[key],str):assert old[key]==new[key]
            else:np.testing.assert_allclose(old[key],new[key],rtol=1e-12,atol=1e-12)
else:
    study.write(summary,dict(scope='post-outcome descriptive case study; no change to registered decisions',
                quadruples=quadruples.tolist(),results=rows))
study.write(root/'joint-sample-case-details.json',dict(status='PASS',scope='post-outcome descriptive case study; unchanged decision',results=rows,details=details))
print(json.dumps(dict(status='PASS',families=len(rows),sample_count=2048,independent_bit_membership=True,independent_distance=True)),flush=True)
