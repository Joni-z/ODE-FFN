"""Replay product16 endpoint fields and recompute saved-sample/rank metrics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch


def audit(study: Path,source: Path,data_root: Path,output: Path) -> dict:
    sys.path.insert(0,str(source))
    from synthetic.models import VelocityNet
    from synthetic.train import load_distribution,restore_model,sha256,sliced_w2
    from scripts.check_gate_geometry import jacobian,oracle_geometry
    torch.set_num_threads(1)
    provenance=json.loads((study/'provenance.json').read_text())
    for path,digest in provenance['source_sha256'].items():
        assert sha256(source/path.replace('\\','/'))==digest,path
    heldout=json.loads((study/'heldout/summary.json').read_text())
    ranks=json.loads((study/'clock-rank/summary.json').read_text())
    assert heldout['state']==ranks['state']=='COMPLETE'
    assert sha256(data_root/'manifest.json')==heldout['manifest_sha256']
    manifest=json.loads((data_root/'manifest.json').read_text())
    for folder,summary in [('heldout',heldout),('clock-rank',ranks)]:
        for path,digest in summary['files_sha256'].items():assert sha256(study/folder/path.replace('\\','/'))==digest
    largest_forward=largest_derivative=0.;arrays=0
    for row in heldout['rows']:
        dataset,label=row['dataset'],row['label']
        cfg_path=study/'configs'/f'{label}.json';cfg=json.loads(cfg_path.read_text())
        checkpoint=study/'runs'/cfg['experiment_id']/dataset/cfg['models'][0]/'seed000/checkpoints/step_0020000.pt'
        assert sha256(checkpoint)==row['checkpoint_sha256']
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert state['config_sha256']==sha256(cfg_path)
        dist=load_distribution(manifest['datasets'][dataset])
        model=VelocityNet(dist.dim,cfg['models'][0],**cfg['model_options']).double().eval();restore_model(model,state)
        errors=[]
        for path in sorted((study/'heldout'/dataset/label/'field').glob('common-*.npz')):
            with np.load(path,allow_pickle=False) as a:
                time=float(path.stem.removeprefix('common-'))
                truth=oracle_geometry(dist,a['x'],time)
                np.testing.assert_allclose(a['oracle'],truth['oracle'],atol=1e-11,rtol=1e-11)
                np.testing.assert_allclose(a['oracle_jacobian'],truth['oracle_jacobian'],atol=1e-10,rtol=1e-11)
                pred,derivative=jacobian(model,torch.from_numpy(a['x']),torch.full((len(a['x']),1),time,dtype=torch.float64))
                np.testing.assert_allclose(a['prediction'],pred.numpy(),atol=1e-10,rtol=1e-10)
                np.testing.assert_allclose(a['jacobian'],derivative.numpy(),atol=1e-9,rtol=1e-10)
                largest_forward=max(largest_forward,float(np.max(np.abs(a['prediction']-pred.numpy()))))
                largest_derivative=max(largest_derivative,float(np.max(np.abs(a['jacobian']-derivative.numpy()))))
                errors.append([np.mean((a['prediction']-a['oracle'])**2),np.mean(np.sum((a['jacobian']-a['oracle_jacobian'])**2,axis=(1,2)))])
            arrays+=1
        np.testing.assert_allclose(np.mean(errors,axis=0),[row['test_field_mse'],row['test_jacobian_error']],atol=1e-12,rtol=1e-12)
        with np.load(study/'heldout'/f'{dataset}-inputs.npz',allow_pickle=False) as bank:
            target=bank['target'];directions=torch.from_numpy(bank['directions'])
            for sampling in row['sampling']:
                with np.load(study/'heldout'/dataset/label/f'samples-heun{sampling["heun_steps"]}.npz',allow_pickle=False) as a:
                    samples=a['samples'];latent=samples.astype(np.float64)@dist.rotation;reference=target.astype(np.float64)@dist.rotation
                    measured=[sliced_w2(torch.from_numpy(samples),torch.from_numpy(target),directions),
                        float(np.sqrt(np.mean((np.sort(latent,axis=0)-np.sort(reference,axis=0))**2,axis=0)).mean()),
                        float(np.mean(np.abs(np.abs(latent)-dist.locations)>.6)),float(np.mean(np.abs(np.mean(latent>=0,axis=0)-.5)))]
                    np.testing.assert_allclose(measured,[sampling[k] for k in ['sw2','mean_marginal_w2','off_well_coordinate_fraction','mean_sign_imbalance']],atol=1e-7,rtol=1e-6)
    for row in ranks['rows']:
        path=study/'clock-rank'/row['dataset']/row['label']/f'common-{row["time"]}.npz'
        with np.load(path,allow_pickle=False) as a:
            correction=a['full_jacobian']-a['anchored_jacobian']
            singular=np.linalg.svd(correction,compute_uv=False)
            np.testing.assert_allclose(singular,a['correction_singular_values'],atol=1e-10,rtol=1e-9)
            residual=a['oracle_jacobian']-a['anchored_jacobian']
            tail=np.sum(np.linalg.svd(residual,compute_uv=False)[:,row['rank_upper_bound']:]**2,axis=1)
            error=np.sum((a['full_jacobian']-a['oracle_jacobian'])**2,axis=(1,2))
            np.testing.assert_allclose(tail,a['conditional_rank_floor'],atol=1e-9,rtol=1e-9)
            assert np.all(error+1e-9>=tail)
            np.testing.assert_allclose([error.mean(),tail.mean()],[row['field_jacobian_error'],row['conditional_rank_floor']],atol=1e-9,rtol=1e-9)
    result=dict(state='PASS',source_verified=True,checkpoints_reloaded=4,heldout_fields_recomputed=arrays,
        rank_arrays_recomputed=len(ranks['rows']),saved_sample_metrics_recomputed=True,
        maximum_forward_difference=largest_forward,maximum_derivative_difference=largest_derivative,
        scope='Final test fields and saved-sample/rank metrics; no sampling-trajectory or intermediate-training replay.')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','source','data-root','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();print(json.dumps(audit(args.study,args.source,args.data_root,args.output),indent=2))
