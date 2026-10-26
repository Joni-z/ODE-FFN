"""Common-input analytic field/Jacobian evaluation in arbitrary dimension."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.check_gate_geometry import jacobian,oracle_geometry
from synthetic.models import VelocityNet
from synthetic.train import load_distribution,load_targets,restore_model,sha256


def check(checkpoint: Path,config: Path,data_root: Path,output: Path,points: int=512,
          split: str='validation') -> dict:
    if split not in ('validation','test'):raise ValueError('split must be validation or test')
    torch.set_num_threads(1)
    output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(config.read_text())
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['config_sha256']==sha256(config)
    assert state['dataset_manifest_sha256']==sha256(data_root/'manifest.json')
    dataset,kind=state['identity']['dataset'],state['identity']['model']
    manifest=json.loads((data_root/'manifest.json').read_text())
    dist=load_distribution(manifest['datasets'][dataset])
    dim=dist.dim
    model=VelocityNet(dim,kind,**cfg['model_options']).double().eval()
    restore_model(model,state)
    for parameter in model.parameters():parameter.requires_grad_(False)
    targets=load_targets(data_root/dataset/f'{split}.npz',dim)[:points].astype(np.float64)
    source=np.random.default_rng(0).standard_normal(targets.shape)
    rows=[]
    for t in (.1,.3,.5,.7,.9):
        x=torch.from_numpy((1-t)*source+t*targets)
        tt=torch.full((len(x),1),t,dtype=torch.float64)
        truth=oracle_geometry(dist,x.numpy(),t)
        prediction,derivative=jacobian(model,x,tt)
        pred,jac=prediction.numpy(),derivative.numpy()
        row=dict(time=t,field_mse=float(np.mean((pred-truth['oracle'])**2)),
            jacobian_error=float(np.mean(np.sum((jac-truth['oracle_jacobian'])**2,axis=(1,2)))))
        rows.append(row)
        np.savez_compressed(output/f'common-{t}.npz',x=x.numpy(),prediction=pred,
            jacobian=jac,oracle=truth['oracle'],oracle_jacobian=truth['oracle_jacobian'])
    result=dict(dataset=dataset,model=kind,dimension=dim,step=state['step'],
        seed=state['identity']['seed'],checkpoint_sha256=sha256(checkpoint),
        config_sha256=sha256(config),manifest_sha256=sha256(data_root/'manifest.json'),
        split=split,targets_sha256=sha256(data_root/dataset/f'{split}.npz'),
        rows=rows,field_mse=float(np.mean([r['field_mse'] for r in rows])),
        jacobian_error=float(np.mean([r['jacobian_error'] for r in rows])),
        arrays_sha256={p.name:sha256(p) for p in sorted(output.glob('*.npz'))})
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','config','data-root','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--split',choices=('validation','test'),default='validation')
    args=parser.parse_args()
    print(json.dumps(check(args.checkpoint,args.config,args.data_root,args.output,split=args.split),indent=2))
