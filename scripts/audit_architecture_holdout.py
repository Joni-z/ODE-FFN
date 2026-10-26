"""Replay heldout fields from frozen source, weights and recorded runtime buffers."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch


def audit(root: Path,source: Path,data: Path,buffers_path: Path,output: Path) -> list[dict]:
    sys.path.insert(0,str(source))
    from synthetic.models import VelocityNet
    from synthetic.train import sha256,load_distribution,sliced_w2
    from scripts.check_gate_geometry import oracle_geometry,jacobian
    torch.set_num_threads(1)
    buffers=json.loads(buffers_path.read_text())
    reports=[]
    for name,dsroot in [('long_ring_20261024','v1'),('rotated16_20261024','rotated16_v1')]:
        study=root/name
        summary=json.loads((study/'heldout/summary.json').read_text())
        assert summary['state']=='COMPLETE'
        data_root=data/dsroot
        manifest=json.loads((data_root/'manifest.json').read_text())
        assert sha256(data_root/'manifest.json')==summary['manifest_sha256']
        provenance=json.loads((study/'provenance.json').read_text())
        for file,digest in provenance['source_sha256'].items():
            assert sha256(source/file.replace('\\','/'))==digest,file
        for file,digest in summary['files_sha256'].items():
            assert sha256(study/'heldout'/file.replace('\\','/'))==digest,file
        count=0
        common={}
        maximum_forward_difference=maximum_jacobian_difference=0.
        for row in summary['rows']:
            label,dataset=row['label'],row['dataset']
            cfg_path=study/'configs'/f'{label}.json'
            cfg=json.loads(cfg_path.read_text())
            checkpoint=study/'runs'/cfg['experiment_id']/dataset/cfg['models'][0]/'seed000/checkpoints/step_0020000.pt'
            assert sha256(checkpoint)==row['checkpoint_sha256']
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert state['config_sha256']==sha256(cfg_path)
            model=VelocityNet(manifest['datasets'][dataset]['dim'],cfg['models'][0],**cfg['model_options']).double().eval()
            model.load_state_dict(state['model'],strict=True)
            # The original frequency buffer was nonpersistent. Windows and macOS
            # float32 exp/linspace differ slightly; preserve the actual Windows buffer.
            model.time.frequencies.copy_(torch.tensor(buffers['time.frequencies'],dtype=torch.float64))
            dist=load_distribution(manifest['datasets'][dataset])
            errors=[]
            for path in sorted((study/'heldout'/dataset/label/'field').glob('common-*.npz')):
                with np.load(path,allow_pickle=False) as a:
                    time=float(path.stem.removeprefix('common-'))
                    if path.name in common:np.testing.assert_array_equal(a['x'],common[path.name])
                    else:common[path.name]=a['x'].copy()
                    truth=oracle_geometry(dist,a['x'],time)
                    np.testing.assert_allclose(a['oracle'],truth['oracle'],atol=1e-12,rtol=1e-12)
                    np.testing.assert_allclose(a['oracle_jacobian'],truth['oracle_jacobian'],atol=1e-11,rtol=1e-11)
                    prediction,derivative=jacobian(model,torch.from_numpy(a['x']),
                        torch.full((len(a['x']),1),time,dtype=torch.float64))
                    maximum_forward_difference=max(maximum_forward_difference,float(np.max(np.abs(a['prediction']-prediction.numpy()))))
                    maximum_jacobian_difference=max(maximum_jacobian_difference,float(np.max(np.abs(a['jacobian']-derivative.numpy()))))
                    np.testing.assert_allclose(a['prediction'],prediction.numpy(),atol=1e-10,rtol=1e-10)
                    np.testing.assert_allclose(a['jacobian'],derivative.numpy(),atol=1e-9,rtol=1e-9)
                    errors.append([np.mean((a['prediction']-a['oracle'])**2),
                        np.mean(np.sum((a['jacobian']-a['oracle_jacobian'])**2,axis=(1,2)))])
                count+=1
            np.testing.assert_allclose(np.mean(errors,axis=0),[row['test_field_mse'],row['test_jacobian_error']],atol=1e-12,rtol=1e-12)
            with np.load(study/'heldout'/f'{dataset}-inputs.npz',allow_pickle=False) as bank:
                directions=torch.from_numpy(bank['directions'])
                target=torch.from_numpy(bank['target'])
                for sampling in row['sampling']:
                    with np.load(study/'heldout'/dataset/label/f'samples-heun{sampling["heun_steps"]}.npz',allow_pickle=False) as sample:
                        sw2=sliced_w2(torch.from_numpy(sample['samples']),target,directions)
                        np.testing.assert_allclose(sw2,sampling['sw2'],atol=1e-7,rtol=1e-6)
            if dist.dim==16:
                components=[]
                for path in sorted((study/'heldout'/dataset/label/'field').glob('common-*.npz')):
                    with np.load(path,allow_pickle=False) as a:
                        error=(a['prediction']-a['oracle'])@dist.rotation
                        components.append([np.mean(error[:,:2]**2),np.mean(error[:,2:]**2)])
                np.testing.assert_allclose(np.mean(components,axis=0),
                    [row['test_mode_field_mse'],row['test_background_field_mse']],atol=1e-12,rtol=1e-12)
        reports.append(dict(study=name,state='PASS',field_arrays=count,weights_reloaded=3,
            maximum_forward_difference=maximum_forward_difference,maximum_jacobian_difference=maximum_jacobian_difference,
            source_verified=True,forward_and_jacobian_recomputed=True,oracles_recomputed=True,
            saved_sample_sw2_recomputed=True,runtime_buffers_sha256=sha256(buffers_path),source=str(source),
            scope='Final heldout fields and saved-sample metrics; sampling trajectories were not replayed. Training-trajectory audit is separate.'))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(reports,indent=2,allow_nan=False)+'\n')
    return reports


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','source','data','buffers','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.root,args.source,args.data,args.buffers,args.output),indent=2))
