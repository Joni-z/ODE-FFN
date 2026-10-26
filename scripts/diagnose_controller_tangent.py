"""Exact frozen FFN/controller tangent diagnosis on exposed validation anchors."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import psutil
import torch

if __package__:
    from .controller_tangent import kernels
else:
    from controller_tangent import kernels


def sha(path: Path) -> str:return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path,value: dict) -> None:
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def ridge_summary(kernel: np.ndarray,residual: np.ndarray,fit: np.ndarray,evaluation: np.ndarray) -> tuple[dict,dict]:
    fit_coordinates=np.concatenate([np.arange(i*residual.shape[1],(i+1)*residual.shape[1]) for i in fit])
    eval_coordinates=np.concatenate([np.arange(i*residual.shape[1],(i+1)*residual.shape[1]) for i in evaluation])
    target=residual[fit].reshape(-1);check=residual[evaluation].reshape(-1)
    ff=kernel[np.ix_(fit_coordinates,fit_coordinates)];ef=kernel[np.ix_(eval_coordinates,fit_coordinates)]
    scale=float(np.trace(ff)/len(ff));original_fit=float(np.mean(target**2));original_eval=float(np.mean(check**2))
    if scale<1e-30:
        return dict(trace_scale=scale,zero_response=True,original_calibration_mse=original_fit,original_evaluation_mse=original_eval),{}
    ff=ff/scale;ef=ef/scale
    eigen,vectors=np.linalg.eigh((ff+ff.T)/2)
    minimum=float(eigen[0]);assert minimum>=-1e-9*max(1.,float(eigen[-1])),minimum
    eigen=np.maximum(eigen,0);coordinates=vectors.T@target
    rows=[];arrays=dict(calibration_eigenvalues=eigen)
    for penalty in (1e-4,1e-3,1e-2,.1,1.,10.):
        coefficient=vectors@(coordinates/(eigen+penalty))
        delta_fit=ff@coefficient;delta_eval=ef@coefficient
        rows.append(dict(penalty=penalty,calibration_mse=float(np.mean((target-delta_fit)**2)),
            evaluation_mse=float(np.mean((check-delta_eval)**2)),
            calibration_ratio=float(np.mean((target-delta_fit)**2))/original_fit,
            evaluation_ratio=float(np.mean((check-delta_eval)**2))/original_eval,
            feature_update_norm=float(np.sqrt(max(0.,coefficient@ff@coefficient/scale)))))
        arrays[f'evaluation_correction_{penalty}']=delta_eval.reshape(len(evaluation),residual.shape[1])
    alignment=float(target@ff@target/(np.dot(target,target)*len(target)))
    return dict(trace_scale=scale,zero_response=False,original_calibration_mse=original_fit,
        original_evaluation_mse=original_eval,normalized_eigen_min=minimum,
        normalized_eigen_max=float(eigen[-1]),normalized_residual_alignment=alignment,
        effective_rank=float(np.sum(eigen)**2/max(np.dot(eigen,eigen),1e-300)),rows=rows),arrays


def run(studies: Path,source: Path,output: Path) -> dict:
    sys.path.insert(0,str(source))
    from synthetic.models import VelocityNet
    from synthetic.train import restore_model
    torch.set_num_threads(1)
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();rows=[];files={};process=psutil.Process()
    studies_found=[]
    for study in sorted(studies.iterdir()):
        if not (study/'audit/audit.json').exists():continue
        audit=json.loads((study/'audit/audit.json').read_text());assert audit['state']=='PASS'
        provenance=json.loads((study/'provenance.json').read_text());assert provenance['device']=='cpu'
        for name,digest in provenance['source_sha256'].items():assert sha(source/name)==digest,name
        studies_found.append((study,provenance['plan']))
        files[str(study/'audit/audit.json')]=sha(study/'audit/audit.json')
    cases=[]
    for study,plan in studies_found:
        for dataset in plan['datasets']:
            if dataset not in ('gmm_ring8_2d','rotated_product_bimodal_16d'):continue
            for arm in plan['arms']:
                for step in (1000,5000):cases.append((study,dataset,arm,step))
    assert len(cases)==24
    for index,(study,dataset,arm,step) in enumerate(cases):
        label,kind=arm['label'],arm['model']
        write(output/'status.json',dict(state='RUNNING',completed=index,total=len(cases),
            active=f'{dataset}/{label}/{step}',pid=os.getpid(),rss_mib=process.memory_info().rss/1024**2,
            time=datetime.now(timezone.utc).isoformat()))
        cfg_path=study/'configs'/f'{label}.json';cfg=json.loads(cfg_path.read_text())
        cp_path=study/'runs'/cfg['experiment_id']/dataset/kind/'seed000/checkpoints'/f'step_{step:07d}.pt'
        state=torch.load(cp_path,map_location='cpu',weights_only=False)
        assert state['step']==step and state['identity']==dict(dataset=dataset,model=kind,seed=0)
        assert state['config_sha256']==sha(cfg_path)
        summary=json.loads((study/'diagnostics'/dataset/label/'summary.json').read_text())
        assert summary['manifest_sha256']==state['dataset_manifest_sha256']
        batches=[];times=[];truth=[];fit=[];evaluation=[]
        for time_value in (.1,.3,.5,.7,.9):
            path=study/'diagnostics'/dataset/label/f'common-{time_value}.npz'
            assert sha(path)==summary['arrays_sha256'][path.name]
            with np.load(path) as data:
                selected=np.r_[0:16,256:272]
                start=sum(map(len,batches));fit.extend(range(start,start+16));evaluation.extend(range(start+16,start+32))
                batches.append(data['x'][selected]);truth.append(data['oracle'][selected]);times.append(np.full((32,1),time_value))
        x=torch.from_numpy(np.concatenate(batches));t=torch.from_numpy(np.concatenate(times));oracle=np.concatenate(truth)
        model=VelocityNet(x.shape[1],kind,**cfg['model_options']).double().eval();restore_model(model,state)
        for parameter in model.parameters():parameter.requires_grad_(False)
        measurement=kernels(model,x,t);prediction=measurement['prediction'].numpy();residual=oracle-prediction
        fit,evaluation=np.array(fit),np.array(evaluation)
        record=dict(dataset=dataset,label=label,step=step,checkpoint_sha256=sha(cp_path),config_sha256=sha(cfg_path),
            dataset_manifest_sha256=state['dataset_manifest_sha256'],
            raw_oracle_mse=float(np.mean(residual**2)),layers=measurement['layers'],components={})
        arrays=dict(x=x.numpy(),t=t.numpy(),oracle=oracle,prediction=prediction,fit_indices=fit,evaluation_indices=evaluation)
        if 'free_gate_response' in measurement:
            response=measurement['free_gate_response'].numpy();left,singular,_=np.linalg.svd(response,full_matrices=False)
            keep=singular>np.maximum(1e-12,singular[:,:1]*1e-12)
            basis=left*keep[:,None,:];projected=(basis@(basis.transpose(0,2,1)@residual[:,:,None]))[:,:,0]
            record['free_gate_rank_min']=int(keep.sum(1).min());record['free_gate_rank_max']=int(keep.sum(1).max())
            record['free_gate_residual_floor_share']=float(np.sum((residual-projected)**2)/np.sum(residual**2))
            arrays['free_gate_singular_values']=singular
        for name,kernel in measurement['components'].items():
            matrix=kernel.numpy();assert np.allclose(matrix,matrix.T,atol=2e-12,rtol=2e-12)
            result,stored=ridge_summary(matrix,residual,fit,evaluation)
            record['components'][name]=result
            arrays.update({name+'__'+key:value for key,value in stored.items()})
        target=output/dataset/label/f'step-{step}';target.mkdir(parents=True)
        np.savez_compressed(target/'response.npz',**arrays)
        write(target/'summary.json',record);rows.append(record)
        files[str(cp_path)]=sha(cp_path);files[str(target/'response.npz')]=sha(target/'response.npz')
        write(output/'summary.json',dict(state='RUNNING',rows=rows,files_sha256=files))
        print(json.dumps(dict(completed=index+1,total=len(cases),active=f'{dataset}/{label}/{step}',
            elapsed_seconds=time.monotonic()-started,rss_mib=process.memory_info().rss/1024**2)),flush=True)
    result=dict(state='COMPLETE',rows=rows,files_sha256=files,script_sha256=sha(Path(__file__)),
        helper_sha256=sha(Path(__file__).with_name('controller_tangent.py')),elapsed_seconds=time.monotonic()-started,
        scope='Read-only privileged-label tangent models on exposed validation anchors. No weight updates, new holdout or trained architecture success.')
    write(output/'summary.json',result);write(output/'status.json',dict(state='COMPLETE',completed=len(cases),total=len(cases),pid=os.getpid()))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('studies','source','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();run(args.studies,args.source,args.output)
