"""Actual operation counts and isolated timings, without optimizer updates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from synthetic.local_adjoint import build
from synthetic.conservative_readout import compute_budget


def profile(output: Path) -> None:
    torch.set_num_threads(1)
    settings=("original","local_full","local_frozen_clock","tied128","tied192","swiglu128","swiglu192")
    rows=[]
    x=torch.linspace(-2,2,512*16).reshape(512,16)
    t=torch.linspace(.05,.95,512).reshape(512,1)
    target=torch.cos(x)
    for setting in settings:
        model=build(16,setting)
        budget=compute_budget(model)
        with torch.no_grad():
            head=model.raw.output.weight
            head.copy_(torch.linspace(-.03,.04,head.numel()).reshape_as(head))
        model.eval()
        def inference() -> None:
            with torch.no_grad():model(x,t)
        def backward() -> None:
            model.zero_grad(set_to_none=True)
            loss=(model(x,t)-target).square().mean()
            loss.backward()
        for _ in range(8):inference()
        for _ in range(3):backward()
        forward_times=[];backward_times=[]
        for _ in range(5):
            before=time.perf_counter()
            for _ in range(20):inference()
            forward_times.append((time.perf_counter()-before)/20)
            before=time.perf_counter()
            for _ in range(10):backward()
            backward_times.append((time.perf_counter()-before)/10)
        rows.append(dict(setting=setting,**budget,inference_ms=statistics.median(forward_times)*1000,
            loss_backward_ms=statistics.median(backward_times)*1000,
            inference_group_seconds=forward_times,loss_backward_group_seconds=backward_times))
    index={r["setting"]:r for r in rows}
    candidate,parent,sw=index["local_full"],index["original"],index["swiglu192"]
    checks=dict(parameters_at_most_80_percent=candidate["parameters"]<=.8*parent["parameters"],
        inference_MACs_at_most_110_percent_swiglu192=candidate["dense_forward_macs"]<=1.1*sw["dense_forward_macs"],
        loss_backward_MACs_at_most_110_percent_original=candidate["dense_training_forward_backward_macs"]<=1.1*parent["dense_training_forward_backward_macs"])
    source={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ("synthetic/local_adjoint.py","synthetic/explicit_adjoint.py","synthetic/models.py",
             "synthetic/conservative_readout.py","tests/test_local_adjoint.py",
             "docs/research-local-adjoint-2026-10-26/prototype-protocol.md")}
    result=dict(settings=rows,resource_checks=checks,resource_screen_passed=all(checks.values()),
        quality_experiment_selected=False,optimizer_updates=0,batch=512,threads=1,
        timing_scope="median of five groups; no optimizer step; nonzero receiving head; pointwise work included by timing only",
        source_sha256=source,profiler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with output.open("x") as handle:handle.write(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(resource_checks=checks,resource_screen_passed=all(checks.values()),
        settings=[{key:row[key] for key in ("setting","parameters","dense_forward_macs",
            "dense_training_forward_backward_macs","inference_ms","loss_backward_ms")} for row in rows]),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();profile(args.output.resolve())
