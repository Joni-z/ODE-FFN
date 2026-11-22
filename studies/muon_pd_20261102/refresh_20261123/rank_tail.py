"""Partition empirical SW2 squared across central and extreme projection ranks."""
import json
import os
from pathlib import Path
import sys
import numpy as np
from odd_trial import base


def read(path):
    return json.loads(path.read_text())


def decompose(folder,step):
    binding=read(folder/'binding.json')
    assets=binding['assets']
    assert base.digest(assets['file'])==assets['sha256'] and binding['split']=='selection'
    with np.load(assets['file']) as arrays:
        dirs=arrays['directions'].astype(float)
        target=np.concatenate([arrays[f'target_{i}'] for i in range(8)]).astype(float)
    y=np.concatenate([np.load(folder/f'{step}-b{i}.npy') for i in range(8)]).astype(float)
    assert len(y)==32768 and len(dirs)==256
    k=32
    middle=tail=0.
    for d in np.array_split(dirs,16):
        delta2=np.square(np.sort(y@d.T,axis=0)-np.sort(target@d.T,axis=0))
        tail+=delta2[:k].sum()+delta2[-k:].sum()
        middle+=delta2[k:-k].sum()
    measured,=[s for s in read(folder/'summary.json') if s['step']==step]
    sw2=np.sqrt((middle+tail)/(len(y)*len(dirs)))
    np.testing.assert_allclose(sw2,measured['pooled_sw2'],rtol=1e-10)
    return dict(dataset=binding['source_config']['cell']['dataset'],family=binding['source_config']['cell']['family'],
                step=step,pooled_sw2=sw2,central_rank_rms=np.sqrt(middle/((len(y)-2*k)*len(dirs))),
                extreme_fraction_of_squared_sw2=tail/(middle+tail),k_per_tail=k,
                n=len(y),directions=len(dirs),assets_sha256=assets['sha256'],
                checkpoint_sha256=measured['checkpoint_sha256'])


def run(prep,output):
    os.nice(10)
    rows=[]
    for tag in ('odd-field','bounded-odd','odd-control'):
        for row in read(prep/f'{tag}-20261123/selection.json'):
            rows.append(decompose(prep/f'{tag}-20261123/selection'/row['dataset'],row['step']))
    original=prep/'coefficient-20261116/analysis/selection'
    for case in ('gmm_ring8_2d__free_isru__*','rotated_product_bimodal_16d__swiglu192__*'):
        folder,=original.glob(case)
        rows.append(decompose(folder,20000))
    base.dump(output,rows)


if __name__=='__main__':
    run(Path(sys.argv[1]),Path(sys.argv[2]))
