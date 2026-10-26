"""One readout factor, unchanged ODE/SwiGLU backbones and monitored seed0 training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from synthetic.conservative_readout import build, compute_budget
from synthetic.population import fingerprint_batch
from scripts import run_recurrent_depth as base

DOC='docs/research-conservative-readout-2026-10-26'
ARMS={f'{family}_{readout}':dict(family=family,readout=readout)
      for family in ('ode_k2','swiglu128','swiglu192') for readout in ('direct','gradient')}
ARMS['swiglu384_direct']=dict(family='swiglu384',readout='direct')


def diagnostics(model,banks,output):
    metrics=base.diagnostics(model,banks,output,rollout=False)
    records=[]
    for clock in (.1,.3,.5,.7,.9):
        with np.load(output/f'panel_{clock:.1f}.npz') as data:
            jac=data['jacobian'].astype(np.float64)
        symmetric=.5*(jac+jac.transpose(0,2,1))
        skew=.5*(jac-jac.transpose(0,2,1))
        relative=np.linalg.norm(jac-jac.transpose(0,2,1),axis=(1,2))/np.maximum(1.,np.linalg.norm(jac,axis=(1,2)))
        if model.readout=='gradient' and float(relative.max())>1e-4:
            raise AssertionError('complete potential Hessian is not numerically symmetric')
        with np.load(banks/f'panel_{clock:.1f}.npz') as data:
            truth=data['oracle_jacobian']
        records.append(dict(time=clock,symmetric_error=float(np.mean(np.sum((symmetric-truth)**2,axis=(1,2)))),
            skew_error=float(np.mean(np.sum(skew**2,axis=(1,2)))),maximum_relative_asymmetry=float(relative.max())))
    base.write(output/'symmetric-skew.json',records)
    return metrics


def train_cell(config_path,output,dataset,arm,steps=5000,*,build_model=None,arm_settings=None,final_diagnostics=None,visit_collector=None):
    torch.set_num_threads(1)
    config=json.loads(config_path.read_text())
    setting=(ARMS if arm_settings is None else arm_settings)[arm]
    folder=output/'runs'/dataset/arm
    folder.mkdir(parents=True,exist_ok=False)
    base.write(folder/'status.json',dict(state='RUNNING',step=0,pid=os.getpid(),time=base.now()))
    try:
        root,spec,_=base.load_data(config,dataset)
        targets=torch.from_numpy(base.load_targets(root/dataset/'train.npz',spec['dim']))
        model=(build if build_model is None else build_model)(spec['dim'],setting['family'],setting['readout'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0.)
        rng=torch.Generator().manual_seed(100000)
        digest=hashlib.sha256(b'').hexdigest()
        costs=compute_budget(model)
        banks=output/'banks'/dataset
        base.write(folder/'provenance.json',dict(dataset=dataset,arm=arm,setting=setting,seed=0,
            actual_device='cpu',torch=torch.__version__,costs=costs,
            config_sha256=base.sha256(config_path),manifest_sha256=base.sha256(root/'manifest.json')))
        base.checkpoint(model,optimizer,rng,0,digest,folder)
        collect_visits=base.visit_diagnostics if visit_collector is None else visit_collector
        base.write(folder/'visits_0000000.json',collect_visits(model,banks))
        base.append(folder/'metrics.jsonl',dict(step=0,**base.field_evaluation(model,banks)))
        started=time.monotonic()
        loss_sum=gradient_sum=0.
        count=clipped=0
        for step in range(1,steps+1):
            target=targets[torch.randint(len(targets),(512,),generator=rng)]
            source=torch.randn(target.shape,generator=rng)
            clock=torch.rand((512,1),generator=rng)
            state=(1-clock)*source+clock*target
            digest=fingerprint_batch(digest,target,source,clock)
            optimizer.zero_grad(set_to_none=True)
            loss=(model(state,clock)-(target-source)).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite training loss')
            loss.backward()
            gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(gradient):
                raise FloatingPointError('nonfinite training gradient')
            optimizer.step()
            loss_sum+=loss.item()
            gradient_sum+=gradient.item()
            clipped+=int(gradient>1)
            count+=1
            if step%500==0 or step==steps:
                record=dict(step=step,train_loss=loss_sum/count,grad_norm=gradient_sum/count,
                    clip_fraction=clipped/count,wall_seconds=time.monotonic()-started,time=base.now())
                if step in (1000,2500,steps):
                    record.update(base.field_evaluation(model,banks))
                    base.checkpoint(model,optimizer,rng,step,digest,folder)
                    base.write(folder/f'visits_{step:07d}.json',collect_visits(model,banks))
                base.append(folder/'metrics.jsonl',record)
                base.write(folder/'status.json',dict(state='RUNNING',step=step,pid=os.getpid(),time=base.now()))
                print(json.dumps(record),flush=True)
                count=clipped=0
                loss_sum=gradient_sum=0.
        result=(diagnostics if final_diagnostics is None else final_diagnostics)(model,banks,folder/'diagnostics')
        base.write(folder/'result.json',dict(dataset=dataset,arm=arm,**setting,step=steps,
            input_stream_digest=digest,**costs,**result))
        base.write(folder/'status.json',dict(state='COMPLETE',step=steps,pid=os.getpid(),time=base.now()))
    except BaseException as error:
        base.write(folder/'status.json',dict(state='FAILED',error=repr(error),pid=os.getpid(),time=base.now()))
        raise


def run_study(output,config,*,script_relative='scripts/run_conservative_readout.py',protocol_relative=None):
    hashes=json.loads((ROOT/'source-sha256.json').read_text())
    for name,value in hashes.items():
        assert base.sha256(ROOT/name)==value
    output.mkdir(parents=True,exist_ok=False)
    base.write(output/'study.json',config)
    base.write(output/'provenance.json',dict(time=base.now(),python=sys.version,torch=torch.__version__,
        source_root=str(ROOT),source_sha256=hashes,protocol_sha256=base.sha256(ROOT/(protocol_relative or DOC+'/protocol.md'))))
    progress=dict(state='RUNNING',phase='prepare',completed=0,total=len(config['datasets'])*len(config['arms']),time=base.now())
    base.write(output/'status.json',progress)
    stop,aborted,lock=threading.Event(),threading.Event(),threading.Lock()
    watcher=threading.Thread(target=base.monitor,args=(output,stop),daemon=True)
    watcher.start()
    processes=set()
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        PYTHONUNBUFFERED='1',PYTHONUTF8='1',CUDA_VISIBLE_DEVICES='')
    def launch(job):
        dataset,arm=job
        command=[sys.executable,str(ROOT/script_relative),'cell',
            '--output',str(output),'--config',str(output/'study.json'),'--dataset',dataset,'--arm',arm]
        with (output/f'{dataset}-{arm}.log').open('w') as log:
            with lock:
                if aborted.is_set():raise RuntimeError('queue aborted')
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                processes.add(child)
            code=child.wait()
            with lock:processes.remove(child)
        if code:raise RuntimeError('failed cell: '+dataset+'/'+arm)
        return job
    try:
        for dataset in config['datasets']:base.prepare_banks(config,dataset,output)
        progress.update(phase='training',time=base.now())
        base.write(output/'status.json',progress)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures=[executor.submit(launch,(dataset,arm)) for dataset in config['datasets'] for arm in config['arms']]
            try:
                for future in as_completed(futures):
                    job=future.result()
                    progress.update(completed=progress['completed']+1,latest=list(job),time=base.now())
                    base.write(output/'status.json',progress)
            except BaseException:
                aborted.set()
                for future in futures:future.cancel()
                with lock:
                    for child in processes:child.terminate()
                raise
        for name,value in hashes.items():assert base.sha256(ROOT/name)==value
        progress.update(state='COMPLETE',phase='complete',time=base.now())
    except BaseException as error:
        progress.update(state='FAILED',error=repr(error),time=base.now())
        raise
    finally:
        base.write(output/'status.json',progress)
        stop.set()
        watcher.join(timeout=15)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('run','cell'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--dataset',choices=base.DATASETS)
    parser.add_argument('--arm',choices=tuple(ARMS))
    parser.add_argument('--data-v1',type=Path)
    parser.add_argument('--data-product',type=Path)
    parser.add_argument('--steps',type=int,default=5000)
    parser.add_argument('--datasets',nargs='+',choices=base.DATASETS,default=list(base.DATASETS))
    parser.add_argument('--arms',nargs='+',choices=tuple(ARMS),default=list(ARMS))
    args=parser.parse_args()
    if args.action=='cell':
        cell_steps=int(json.loads(args.config.read_text())['steps'])
        train_cell(args.config.resolve(),args.output.resolve(),args.dataset,args.arm,steps=cell_steps)
    else:
        config=dict(seed=0,steps=args.steps,workers=2,device='cpu',datasets=args.datasets,
            arms={name:ARMS[name] for name in args.arms},
            data_roots={d:str((args.data_product if d==base.DATASETS[2] else args.data_v1).resolve()) for d in args.datasets})
        run_study(args.output.resolve(),config)
