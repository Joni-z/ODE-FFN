"""Frozen final product16 fields and joint/marginal sampling diagnostics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from synthetic.data import RotatedProductMixture
from synthetic.models import VelocityNet
from synthetic.train import generated_samples,load_distribution,load_targets,restore_model,sha256,sliced_w2
from scripts.check_vector_field import check


def sample_metrics(samples: torch.Tensor,target: torch.Tensor,directions: torch.Tensor,
                   dist: RotatedProductMixture) -> dict:
    latent=samples.numpy().astype(np.float64)@dist.rotation
    reference=target.numpy().astype(np.float64)@dist.rotation
    marginal=np.sqrt(np.mean((np.sort(latent,axis=0)-np.sort(reference,axis=0))**2,axis=0))
    signs=np.where(latent>=0,1.,-1.)
    second=signs.T@signs/len(signs)
    off_diagonal=second[~np.eye(dist.dim,dtype=bool)]
    _,covariance=dist.moments()
    return dict(sw2=sliced_w2(samples,target,directions),
        mean_marginal_w2=float(np.mean(marginal)),marginal_w2=marginal.tolist(),
        off_well_coordinate_fraction=float(np.mean(np.abs(np.abs(latent)-dist.locations)>.6)),
        mean_sign_imbalance=float(np.mean(np.abs(np.mean(signs,axis=0)))/2),
        per_coordinate_positive_fraction=np.mean(signs>0,axis=0).tolist(),
        sign_second_moment_off_diagonal_rms=float(np.sqrt(np.mean(off_diagonal**2))),
        covariance_relative_error=float(np.linalg.norm(np.cov(samples.numpy(),rowvar=False,ddof=0)-covariance)/np.linalg.norm(covariance)),
        mean_target_negative_log_density=float(-np.mean(dist.log_density(samples.numpy()))))


def evaluate(study: Path,data_root: Path,output: Path) -> dict:
    torch.set_num_threads(1)
    assert json.loads((study/'status.json').read_text())['state']=='COMPLETE'
    provenance=json.loads((study/'provenance.json').read_text())
    plan=provenance['plan']
    assert [a['label'] for a in plan['arms']]==['scalar_k1','vector_k1','swiglu_budget','swiglu']
    output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((data_root/'manifest.json').read_text())
    rows=[];references=[]
    for dataset in plan['datasets']:
        dist=load_distribution(manifest['datasets'][dataset])
        assert isinstance(dist,RotatedProductMixture)
        targets=load_targets(data_root/dataset/'test.npz',dist.dim)
        n=8192;assert len(targets)>=2*n
        rng=np.random.default_rng(0)
        source=rng.standard_normal((n,dist.dim)).astype(np.float32)
        directions=rng.standard_normal((128,dist.dim)).astype(np.float32)
        directions/=np.linalg.norm(directions,axis=1,keepdims=True)
        source_tensor=torch.from_numpy(source);direction_tensor=torch.from_numpy(directions)
        target_tensor=torch.from_numpy(targets[:n])
        references.append(dict(dataset=dataset,**sample_metrics(torch.from_numpy(targets[n:2*n]),target_tensor,direction_tensor,dist)))
        np.savez_compressed(output/f'{dataset}-inputs.npz',source=source,directions=directions,
            target=targets[:n],target2=targets[n:2*n])
        for arm in plan['arms']:
            label,kind=arm['label'],arm['model']
            config=study/'configs'/f'{label}.json';cfg=json.loads(config.read_text())
            checkpoint=study/'runs'/cfg['experiment_id']/dataset/kind/'seed000'/f'checkpoints/step_{plan["steps"]:07d}.pt'
            destination=output/dataset/label
            field=check(checkpoint,config,data_root,destination/'field',split='test')
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model=VelocityNet(dist.dim,kind,**cfg['model_options']).eval();restore_model(model,state)
            latent_error=[]
            for path in sorted((destination/'field').glob('common-*.npz')):
                with np.load(path,allow_pickle=False) as a:
                    error=(a['prediction']-a['oracle'])@dist.rotation
                    latent_error.append(np.mean(error**2,axis=0))
            row=dict(dataset=dataset,label=label,step=state['step'],checkpoint_sha256=sha256(checkpoint),
                test_field_mse=field['field_mse'],test_jacobian_error=field['jacobian_error'],
                test_latent_coordinate_field_mse=np.mean(latent_error,axis=0).tolist(),sample_count=n,sampling=[])
            for steps in (32,128):
                started=time.monotonic();samples=generated_samples(model,source_tensor,steps)
                elapsed=time.monotonic()-started
                row['sampling'].append(dict(heun_steps=steps,nfe=2*steps,generation_seconds=elapsed,
                    **sample_metrics(samples,target_tensor,direction_tensor,dist)))
                np.savez_compressed(destination/f'samples-heun{steps}.npz',samples=samples.numpy())
            rows.append(row)
            (output/'summary.json').write_text(json.dumps(dict(state='RUNNING',rows=rows),indent=2,allow_nan=False)+'\n')
    result=dict(state='COMPLETE',rows=rows,empirical_references=references,
        interpretation='Frozen final checkpoints. Marginal W2 is not joint W2; lower negative log density alone is not better generation.',
        manifest_sha256=sha256(data_root/'manifest.json'),source_sha256=sha256(Path(__file__)),
        files_sha256={p.relative_to(output).as_posix():sha256(p) for p in sorted(output.rglob('*.npz'))})
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('study','data-root','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=evaluate(args.study,args.data_root,args.output)
    print(json.dumps(dict(state=result['state'],rows=len(result['rows']))))
