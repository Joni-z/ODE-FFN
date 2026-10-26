"""Frozen final architecture comparison on heldout fields and sampling."""
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
from synthetic.data import RotatedGaussianMixture
from synthetic.models import VelocityNet
from synthetic.train import generated_samples,load_distribution,load_targets,restore_model,sha256,sliced_w2
from scripts.check_vector_field import check


def evaluate(study: Path,data_root: Path,output: Path) -> dict:
    torch.set_num_threads(1)
    status=json.loads((study/'status.json').read_text())
    assert status['state']=='COMPLETE'
    provenance=json.loads((study/'provenance.json').read_text())
    plan=provenance['plan']
    assert [a['label'] for a in plan['arms']]==['scalar_k1','vector_k1','swiglu']
    output.mkdir(parents=True,exist_ok=False)
    rows=[]
    manifests=json.loads((data_root/'manifest.json').read_text())
    for dataset in plan['datasets']:
        dist=load_distribution(manifests['datasets'][dataset])
        targets=load_targets(data_root/dataset/'test.npz',dist.dim)
        n=4096
        assert len(targets)>=2*n
        rng=np.random.default_rng(0)
        source=rng.standard_normal((n,dist.dim)).astype(np.float32)
        directions=rng.standard_normal((128,dist.dim)).astype(np.float32)
        directions/=np.linalg.norm(directions,axis=1,keepdims=True)
        direction_tensor=torch.from_numpy(directions)
        source_tensor=torch.from_numpy(source)
        first,second=torch.from_numpy(targets[:n]),torch.from_numpy(targets[n:2*n])
        reference_floor=sliced_w2(first,second,direction_tensor)
        np.savez_compressed(output/f'{dataset}-inputs.npz',source=source,directions=directions,
            target=targets[:n],target2=targets[n:2*n])
        for arm in plan['arms']:
            label,kind=arm['label'],arm['model']
            cfg_path=study/'configs'/f'{label}.json'
            cfg=json.loads(cfg_path.read_text())
            checkpoint=study/'runs'/cfg['experiment_id']/dataset/kind/'seed000'/f'checkpoints/step_{plan["steps"]:07d}.pt'
            destination=output/dataset/label
            geometry=check(checkpoint,cfg_path,data_root,destination/'field',split='test')
            model=VelocityNet(dist.dim,kind,**cfg['model_options']).eval()
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            restore_model(model,state)
            row=dict(dataset=dataset,label=label,checkpoint_step=state['step'],
                checkpoint_sha256=sha256(checkpoint),test_field_mse=geometry['field_mse'],
                test_jacobian_error=geometry['jacobian_error'],sample_count=n,
                reference_sw2_floor=reference_floor,sampling=[])
            if isinstance(dist,RotatedGaussianMixture):
                r=dist.rotation
                mode_errors=[]
                background_errors=[]
                for path in sorted((destination/'field').glob('common-*.npz')):
                    with np.load(path,allow_pickle=False) as a:
                        err=(a['prediction']-a['oracle'])@r
                        mode_errors.append(float(np.mean(err[:,:2]**2)))
                        background_errors.append(float(np.mean(err[:,2:]**2)))
                row.update(test_mode_field_mse=float(np.mean(mode_errors)),
                    test_background_field_mse=float(np.mean(background_errors)))
            for steps in (32,128):
                started=time.monotonic()
                samples=generated_samples(model,source_tensor,steps)
                entry=dict(heun_steps=steps,nfe=2*steps,
                    sw2=sliced_w2(samples,first,direction_tensor),seconds=time.monotonic()-started)
                if isinstance(dist,RotatedGaussianMixture):
                    latent=samples.numpy().astype(np.float64)@dist.rotation
                    reference=targets[:n].astype(np.float64)@dist.rotation
                    centers=dist.base.means[:,:2]
                    closest=np.sum((latent[:,None,:2]-centers[None])**2,axis=2)
                    target_closest=np.sum((reference[:,None,:2]-centers[None])**2,axis=2)
                    mode_rng=np.random.default_rng(0)
                    mode_directions=mode_rng.standard_normal((128,2))
                    mode_directions/=np.linalg.norm(mode_directions,axis=1,keepdims=True)
                    entry.update(mode_sw2=sliced_w2(torch.from_numpy(latent[:,:2]),
                        torch.from_numpy(reference[:,:2]),torch.from_numpy(mode_directions)),
                        off_mode_fraction=float(np.mean(np.min(closest,axis=1)>.6**2)),
                        target_off_mode_fraction=float(np.mean(np.min(target_closest,axis=1)>.6**2)),
                        mode_mass=(np.bincount(np.argmin(closest,axis=1),minlength=8)/n).tolist(),
                        background_covariance_relative_error=float(np.linalg.norm(np.cov(latent[:,2:],rowvar=False,ddof=0)-np.diag(dist.base.variances[0,2:]))/np.linalg.norm(dist.base.variances[0,2:])))
                np.savez_compressed(destination/f'samples-heun{steps}.npz',samples=samples.numpy())
                row['sampling'].append(entry)
            rows.append(row)
            (output/'summary.json').write_text(json.dumps(dict(state='RUNNING',rows=rows),indent=2,allow_nan=False)+'\n')
    summary=dict(state='COMPLETE',rows=rows,
        heldout_scope='Frozen architectures and final step; test split not used for this stage selection.',
        source_sha256=sha256(Path(__file__)),manifest_sha256=sha256(data_root/'manifest.json'),
        files_sha256={str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*.npz'))})
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('study','data-root','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(evaluate(args.study,args.data_root,args.output),indent=2))
