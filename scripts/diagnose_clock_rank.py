"""Conditional complete-field clock-rank bound on shared validation anchors."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from synthetic.models import VelocityNet
from synthetic.train import load_distribution,load_targets,restore_model,sha256
from scripts.check_gate_geometry import gate_context,jacobian,oracle_geometry


def rank_tail_floor(residual: np.ndarray,rank: int) -> np.ndarray:
    singular=np.linalg.svd(residual,compute_uv=False)
    return np.sum(singular[...,rank:]**2,axis=-1)


def diagnose(study: Path,data_root: Path,output: Path) -> dict:
    torch.set_num_threads(1)
    assert json.loads((study/'status.json').read_text())['state']=='COMPLETE'
    plan=json.loads((study/'provenance.json').read_text())['plan']
    manifest=json.loads((data_root/'manifest.json').read_text())
    output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for dataset in plan['datasets']:
        dist=load_distribution(manifest['datasets'][dataset])
        targets=load_targets(data_root/dataset/'validation.npz',dist.dim)[:512].astype(np.float64)
        source=np.random.default_rng(0).standard_normal(targets.shape)
        for arm in plan['arms']:
            if arm['label'] not in ('scalar_k1','vector_k1'):continue
            cfg_path=study/'configs'/f'{arm["label"]}.json';cfg=json.loads(cfg_path.read_text())
            checkpoint=study/'runs'/cfg['experiment_id']/dataset/arm['model']/'seed000'/f'checkpoints/step_{plan["steps"]:07d}.pt'
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert state['config_sha256']==sha256(cfg_path)
            model=VelocityNet(dist.dim,arm['model'],**cfg['model_options']).double().eval();restore_model(model,state)
            destination=output/dataset/arm['label'];destination.mkdir(parents=True)
            for time in (.1,.3,.5,.7,.9):
                x=torch.from_numpy((1-time)*source+time*targets)
                t=torch.full((len(x),1),time,dtype=torch.float64)
                truth=oracle_geometry(dist,x.numpy(),time)
                values={}
                with gate_context(model,values,capture=True):native,full=jacobian(model,x,t)
                with gate_context(model,values):anchored,frozen=jacobian(model,x,t)
                torch.testing.assert_close(native,anchored,atol=1e-11,rtol=1e-11)
                j0=frozen.numpy();j=full.numpy();correction=j-j0
                singular=np.linalg.svd(correction,compute_uv=False)
                rank=min(dist.dim,sum(v.shape[-1] for v in values.values()))
                observed=np.sum(singular>np.maximum(1e-12,1e-8*singular[:,:1]),axis=1)
                assert np.all(observed<=rank)
                residual=truth['oracle_jacobian']-j0
                floor=rank_tail_floor(residual,rank)
                error=np.sum((j-truth['oracle_jacobian'])**2,axis=(1,2))
                frozen_error=np.sum(residual**2,axis=(1,2))
                assert np.all(error+1e-9>=floor)
                rows.append(dict(dataset=dataset,label=arm['label'],time=time,
                    rank_upper_bound=rank,observed_correction_rank_max=int(observed.max()),
                    field_jacobian_error=float(error.mean()),anchored_jacobian_error=float(frozen_error.mean()),
                    conditional_rank_floor=float(floor.mean()),floor_share_of_error=float(floor.sum()/max(error.sum(),1e-30)),
                    error_median=float(np.median(error)),error_q95=float(np.quantile(error,.95))))
                np.savez_compressed(destination/f'common-{time}.npz',x=x.numpy(),full_jacobian=j,
                    anchored_jacobian=j0,oracle_jacobian=truth['oracle_jacobian'],
                    correction_singular_values=singular,conditional_rank_floor=floor)
    result=dict(state='COMPLETE',split='validation',points=512,rows=rows,
        scope='Conditional rank floor at trained anchored base; not an irreducible architecture bound.',
        source_sha256=sha256(Path(__file__)),manifest_sha256=sha256(data_root/'manifest.json'),
        files_sha256={p.relative_to(output).as_posix():sha256(p) for p in sorted(output.rglob('*.npz'))})
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('study','data-root','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=diagnose(args.study,args.data_root,args.output)
    print(json.dumps(dict(state=result['state'],rows=len(result['rows']))))
