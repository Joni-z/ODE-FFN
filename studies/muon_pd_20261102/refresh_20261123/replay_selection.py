"""Independent pooled SW2 and radius summaries from each saved selection endpoint."""
import argparse
import json
from pathlib import Path
import numpy as np
from odd_trial import base


def replay(work, out, split="selection"):
    table=[]
    for directory in sorted((work/split).glob('*')):
        if not (directory/'COMPLETE.json').exists():
            continue
        load=lambda p: json.loads(p.read_text())
        done=load(directory/'COMPLETE.json')
        binding=load(directory/'binding.json')
        row,=load(directory/'summary.json')
        assert done['passed'] and done['steps']==[row['step']] and not done['witness']
        assert binding['split']==('confirmation' if split=='archive-confirmation' else 'selection')
        assets=binding['assets']; assert base.digest(assets['file'])==assets['sha256']
        with np.load(assets['file']) as inputs:
            target=np.concatenate([inputs[f'target_{i}'] for i in range(8)]).astype(float)
            dirs=inputs['directions'].astype(float)
        paths=[directory/f"{row['step']}-b{i}.npy" for i in range(8)]
        y=np.concatenate([np.load(p) for p in paths]).astype(float)
        assert y.shape==target.shape and len(y)==32768 and len(dirs)==256
        summed=0.
        for proj in np.array_split(dirs,16):
            d=np.sort(y@proj.T,axis=0)-np.sort(target@proj.T,axis=0)
            summed+=np.square(d).sum()
        sw2=float(np.sqrt(summed/(len(y)*len(dirs))))
        np.testing.assert_allclose(sw2,row['pooled_sw2'],rtol=1e-10)
        radius=np.linalg.norm(y,axis=1); r_target=np.linalg.norm(target,axis=1)
        table.append(dict(dataset=directory.name,family=binding['source_config']['cell']['family'],
                          seed=0,step=row['step'],pooled_sw2=sw2,field_mse=row['field_mse'],
                          bank_sw2=row['bank_sw2'],checkpoint_sha256=row['checkpoint_sha256'],
                          assets_sha256=assets['sha256'],files_sha256=[base.digest(p) for p in paths],
                          radius_quantiles=[.5,.99,1.],
                          generated_radius=np.quantile(radius,[.5,.99,1.]).tolist(),
                          target_radius=np.quantile(r_target,[.5,.99,1.]).tolist(),
                          beyond_twice_target_max=int(np.sum(radius>2*r_target.max()))))
    if not table:
        raise ValueError('No completed evaluation')
    base.dump(out,table)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('work',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--split',choices=('selection','archive-confirmation'),default='selection')
    args=p.parse_args()
    replay(args.work,args.output,args.split)
