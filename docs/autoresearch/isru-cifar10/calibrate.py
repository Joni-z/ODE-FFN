"""Bounded throughput/memory check, not a model-quality experiment."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from model import build_all, costs

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--native', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--fast', action='store_true')
    ap.add_argument('--foreach', action='store_true')
    args = ap.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(1)
    assert torch.backends.mps.is_available()
    models = build_all(args.native, fast_attention=args.fast)
    rng = torch.Generator().manual_seed(0)
    target = torch.randn(args.batch,3,32,32,generator=rng).to('mps')
    noise = torch.randn(args.batch,3,32,32,generator=rng).to('mps')
    t = torch.sigmoid(.8*torch.randn(args.batch,generator=rng)-.8).to('mps')
    x = t[:,None,None,None]*target+(1-t[:,None,None,None])*noise
    rows = []
    for arm, model in models.items():
        row = dict(arm=arm, **costs(model))
        model.to('mps').train()
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.9,.95),weight_decay=0,foreach=args.foreach)
        timings=[]
        for i in range(7):
            torch.mps.synchronize();tick=time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            pred=model(x,t,torch.zeros(args.batch,dtype=torch.long,device='mps'))
            loss=((pred-target)/(1-t[:,None,None,None]).clamp_min(.05)).square().mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=False);optimizer.step()
            torch.mps.synchronize()
            if i>=2:timings.append(time.monotonic()-tick)
        assert torch.isfinite(loss)
        row.update(batch=args.batch,median_update_seconds=float(np.median(timings)),
                   SDPA=args.fast,optimizer_foreach=args.foreach,
                   MPS_allocated_bytes=torch.mps.current_allocated_memory(),
                   MPS_driver_bytes=torch.mps.driver_allocated_memory(),kernel='real full JiT forward/backward/Adam',
                   input='seeded synthetic image tensors; no quality outcomes')
        rows.append(row);print(json.dumps(row),flush=True)
        model.cpu();del optimizer;torch.mps.empty_cache()
    value=dict(status='PASS',device='mps',width=192,depth=6,rows=rows,
               estimated_cohort_update_seconds=sum(v['median_update_seconds'] for v in rows),
               formal_quality_training_started=False)
    args.output.write_text(json.dumps(value,indent=2)+'\n')
