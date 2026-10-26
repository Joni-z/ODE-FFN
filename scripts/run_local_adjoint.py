"""Isolated monitored cold comparison of local primal/adjoint FFNs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from synthetic.local_adjoint import build,LocalAdjointODEFFN
from synthetic.conservative_readout import compute_budget
from synthetic.explicit_adjoint import ode_primal,ode_adjoint
from synthetic.population import fingerprint_batch
from scripts import run_recurrent_depth as base

DATASET="rotated_product_bimodal_16d"
ARMS=("original","local_full","local_frozen_clock","tied128","tied192","swiglu128","swiglu192")
DOC="docs/research-local-adjoint-2026-10-26"


@torch.no_grad()
def visits(model,banks):
    rows=base.visit_diagnostics(model,banks)
    hooks=[]
    for name,module in model.named_modules():
        if isinstance(module,LocalAdjointODEFFN):
            def capture(ffn,args,result,name=name):
                h=args[0];y,cache=ode_primal(ffn.ode1,h);q=F.silu(y)
                frozen=ode_adjoint(cache,q,include_clock=False)
                full=ode_adjoint(cache,q,include_clock=True)
                rms=lambda x:float(x.square().mean().sqrt())
                rows.append(dict(layer=name,kind="local_adjoint",include_clock=ffn.include_clock,
                    encoder=ffn.ode1.activation_stats(h),output_rms=rms(result),
                    clock_adjoint_rms=rms(full-frozen),frozen_adjoint_rms=rms(frozen)))
            hooks.append(module.register_forward_hook(capture))
    if hooks:
        try:
            with np.load(banks/"probe.npz") as bank:model(torch.from_numpy(bank["x"][:32]),torch.from_numpy(bank["t"][:32]))
        finally:
            for hook in hooks:hook.remove()
        assert len([r for r in rows if r.get("kind")=="local_adjoint"])==3
    return rows


def cell(config_path,output,arm,steps):
    torch.set_num_threads(1)
    config=base.json.loads(config_path.read_text())
    folder=output/"runs"/DATASET/arm;folder.mkdir(parents=True,exist_ok=False)
    base.write(folder/"status.json",dict(state="RUNNING",step=0,pid=os.getpid(),time=base.now()))
    try:
        root,spec,_=base.load_data(config,DATASET)
        targets=torch.from_numpy(base.load_targets(root/DATASET/"train.npz",spec["dim"]))
        model=build(spec["dim"],arm)
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0.)
        rng=torch.Generator().manual_seed(100000);digest=hashlib.sha256(b"").hexdigest()
        costs=compute_budget(model);banks=output/"banks"/DATASET
        base.write(folder/"provenance.json",dict(arm=arm,dataset=DATASET,seed=0,costs=costs,
            config_sha256=base.sha256(config_path),manifest_sha256=base.sha256(root/"manifest.json"),torch=torch.__version__))
        base.checkpoint(model,optimizer,rng,0,digest,folder)
        base.write(folder/"visits_0000000.json",visits(model,banks))
        base.append(folder/"metrics.jsonl",dict(step=0,**base.field_evaluation(model,banks)))
        started=time.monotonic();loss_sum=gradient_sum=0.;count=clipped=0
        for step in range(1,steps+1):
            target=targets[torch.randint(len(targets),(512,),generator=rng)]
            source=torch.randn(target.shape,generator=rng);clock=torch.rand((512,1),generator=rng)
            state=(1-clock)*source+clock*target
            digest=fingerprint_batch(digest,target,source,clock)
            optimizer.zero_grad(set_to_none=True)
            loss=(model(state,clock)-(target-source)).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError("nonfinite training loss")
            loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(gradient):raise FloatingPointError("nonfinite training gradient")
            optimizer.step();loss_sum+=loss.item();gradient_sum+=gradient.item();clipped+=int(gradient>1);count+=1
            if step%500==0 or step==steps:
                record=dict(step=step,train_loss=loss_sum/count,grad_norm=gradient_sum/count,
                    clip_fraction=clipped/count,wall_seconds=time.monotonic()-started,time=base.now())
                if step in (1000,2500,steps):
                    record.update(base.field_evaluation(model,banks))
                    base.checkpoint(model,optimizer,rng,step,digest,folder)
                    base.write(folder/f"visits_{step:07d}.json",visits(model,banks))
                base.append(folder/"metrics.jsonl",record)
                base.write(folder/"status.json",dict(state="RUNNING",step=step,pid=os.getpid(),time=base.now()))
                print(json.dumps(record),flush=True)
                count=clipped=0;loss_sum=gradient_sum=0.
        result=base.diagnostics(model,banks,folder/"diagnostics",rollout=False)
        base.write(folder/"result.json",dict(dataset=DATASET,arm=arm,step=steps,input_stream_digest=digest,**costs,**result))
        base.write(folder/"status.json",dict(state="COMPLETE",step=steps,pid=os.getpid(),time=base.now()))
    except BaseException as error:
        base.write(folder/"status.json",dict(state="FAILED",error=repr(error),pid=os.getpid(),time=base.now()));raise


def run(reference,output,steps):
    assert steps in (3,5000)
    hashes=json.loads((ROOT/"source-sha256.json").read_text())
    for name,value in hashes.items():assert base.sha256(ROOT/name)==value
    config=json.loads((reference/"study.json").read_text())
    assert json.loads((reference/"status.json").read_text())["state"]=="COMPLETE"
    config.update(arms=list(ARMS),datasets=[DATASET],steps=steps,workers=2,reference_output=str(reference))
    output.mkdir(parents=True,exist_ok=False);base.write(output/"study.json",config)
    target=output/"banks"/DATASET;target.parent.mkdir(parents=True)
    shutil.copytree(reference/"banks"/DATASET,target)
    base.write(output/"provenance.json",dict(source_root=str(ROOT),source_sha256=hashes,
        reference_provenance_sha256=base.sha256(reference/"provenance.json"),protocol_sha256=base.sha256(ROOT/DOC/"protocol.md")))
    progress=dict(state="RUNNING",completed=0,total=len(ARMS),time=base.now())
    base.write(output/"status.json",progress)
    stop=threading.Event();watcher=threading.Thread(target=base.monitor,args=(output,stop),daemon=True);watcher.start()
    env=os.environ.copy();env.update(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",PYTHONUNBUFFERED="1",CUDA_VISIBLE_DEVICES="")
    processes=set();lock=threading.Lock();aborted=threading.Event()
    def launch(arm):
        command=[sys.executable,str(ROOT/"scripts/run_local_adjoint.py"),"cell","--output",str(output),
                 "--config",str(output/"study.json"),"--arm",arm,"--steps",str(steps)]
        with (output/(arm+".log")).open("w") as log:
            with lock:
                if aborted.is_set():raise RuntimeError("queue aborted")
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);processes.add(child)
            code=child.wait()
            with lock:processes.remove(child)
        if code:raise RuntimeError("failed arm "+arm)
        return arm
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures=[executor.submit(launch,arm) for arm in ARMS]
            try:
                for future in as_completed(futures):
                    arm=future.result();progress.update(completed=progress["completed"]+1,latest=arm,time=base.now())
                    base.write(output/"status.json",progress);print(json.dumps(progress),flush=True)
            except BaseException:
                aborted.set()
                for future in futures:future.cancel()
                with lock:
                    for child in processes:child.terminate()
                raise
        for name,value in hashes.items():assert base.sha256(ROOT/name)==value
        progress.update(state="COMPLETE",time=base.now())
    except BaseException as error:
        progress.update(state="FAILED",error=repr(error),time=base.now());raise
    finally:
        base.write(output/"status.json",progress);stop.set();watcher.join(timeout=15)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="mode",required=True)
    a=sub.add_parser("run");a.add_argument("--reference",type=Path,required=True);a.add_argument("--output",type=Path,required=True)
    a.add_argument("--steps",type=int,choices=(3,5000),default=5000)
    a=sub.add_parser("cell");a.add_argument("--config",type=Path,required=True);a.add_argument("--output",type=Path,required=True)
    a.add_argument("--arm",choices=ARMS,required=True);a.add_argument("--steps",type=int,choices=(3,5000),required=True)
    args=parser.parse_args()
    if args.mode=="run":run(args.reference.resolve(),args.output.resolve(),args.steps)
    else:cell(args.config.resolve(),args.output.resolve(),args.arm,args.steps)
