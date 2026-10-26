"""Separate clock-count and fixed gate-response subspace limits at trained anchors."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch


def project_response(response: np.ndarray,matrix: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    left,singular,_=np.linalg.svd(response,full_matrices=False)
    keep=singular>np.maximum(1e-12,1e-12*singular[:,:1])
    basis=left*keep[:,None,:]
    return basis@(basis.transpose(0,2,1)@matrix),keep.sum(axis=1)


def clock_response(model,x,t):
    from scripts.check_gate_geometry import gate_context
    values={}
    with gate_context(model,values,capture=True):
        native=model(x,t).detach()
    names=[name for name,_ in model.ode_layers()]
    assert all(values[name].shape[-1]==1 for name in names)
    clocks=torch.cat([values[name] for name in names],dim=-1).detach().requires_grad_(True)
    injected={name:clocks[:,i:i+1] for i,name in enumerate(names)}
    with gate_context(model,injected):
        prediction=model(x,t)
        response=torch.stack([torch.autograd.grad(prediction[:,i].sum(),clocks,
            retain_graph=i+1<prediction.shape[1])[0] for i in range(prediction.shape[1])],dim=1)
    torch.testing.assert_close(prediction,native,atol=1e-11,rtol=1e-11)
    return response.detach().numpy()


def diagnose(study: Path,source: Path,data_root: Path,output: Path) -> dict:
    sys.path.insert(0,str(source))
    from synthetic.models import VelocityNet
    from synthetic.train import restore_model,sha256
    torch.set_num_threads(1)
    original=json.loads((study/'clock-rank/summary.json').read_text())
    assert original['state']=='COMPLETE'
    assert original['manifest_sha256']==sha256(data_root/'manifest.json')
    manifest=json.loads((data_root/'manifest.json').read_text())
    config=study/'configs/scalar_k1.json';cfg=json.loads(config.read_text())
    output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for dataset in json.loads((study/'provenance.json').read_text())['plan']['datasets']:
        checkpoint=study/'runs'/cfg['experiment_id']/dataset/'ode_original/seed000/checkpoints/step_0020000.pt'
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert state['config_sha256']==sha256(config)
        dimension=manifest['datasets'][dataset]['dim']
        model=VelocityNet(dimension,'ode_original',**cfg['model_options']).double().eval();restore_model(model,state)
        for row in [r for r in original['rows'] if r['label']=='scalar_k1' and r['dataset']==dataset]:
            relative=f'{dataset}/scalar_k1/common-{row["time"]}.npz'
            path=study/'clock-rank'/relative
            assert sha256(path)==original['files_sha256'][relative]
            with np.load(path,allow_pickle=False) as a:
                response=clock_response(model,torch.from_numpy(a['x']),
                    torch.full((len(a['x']),1),row['time'],dtype=torch.float64))
                residual=a['oracle_jacobian']-a['anchored_jacobian']
                correction=a['full_jacobian']-a['anchored_jacobian']
                projected,ranks=project_response(response,residual)
                projected_c,_=project_response(response,correction)
                leak=correction-projected_c
                assert np.max(np.abs(leak))<1e-9
                floor=np.sum((residual-projected)**2,axis=(1,2))
                error=np.sum((a['full_jacobian']-a['oracle_jacobian'])**2,axis=(1,2))
                assert np.all(floor+1e-8>=a['conditional_rank_floor'])
                assert np.all(floor<=error+1e-8)
                rows.append(dict(dataset=dataset,time=row['time'],response_rank_max=int(ranks.max()),
                    actual_error=float(error.mean()),rank_only_floor=float(a['conditional_rank_floor'].mean()),
                    fixed_response_floor=float(floor.mean()),response_floor_share=float(floor.sum()/error.sum()),
                    correction_projection_max_error=float(np.max(np.abs(leak)))))
                np.savez_compressed(output/f'common-{row["time"]}.npz',response=response,
                    fixed_response_floor=floor,rank_only_floor=a['conditional_rank_floor'])
    result=dict(state='COMPLETE',rows=rows,source_sha256=sha256(Path(__file__)),
        rank_summary_sha256=sha256(study/'clock-rank/summary.json'),
        interpretation='Fixed weights, clock values and response subspace. Allows arbitrary total clock derivatives; not a retrained-architecture lower bound.',
        files_sha256={p.name:sha256(p) for p in sorted(output.glob('*.npz'))})
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','source','data-root','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();print(json.dumps(diagnose(args.study,args.source,args.data_root,args.output),indent=2))
