"""Monitored cold-training screen for a time-dependent original K2 generator."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from synthetic.time_generator import build,time_coordinate
from scripts import run_recurrent_depth as base

DOC="docs/research-time-generator-2026-10-26"
ARMS={f"{family}_{mode}_final":dict(family=family,mode=mode,supervision="final")
      for mode in ("static","time") for family in ("ode_k2","swiglu192")}


def diagnostics(model,banks: Path,output: Path,*,rollout=False):
    metrics=base.diagnostics(model,banks,output,rollout=False)
    trajectory={}
    for step in (0,1000,2500,5000):
        path=output.parent/f"step_{step:07d}.pt"
        if not path.exists():continue
        state=torch.load(path,map_location="cpu",weights_only=False)["model"]
        records={}
        for name,layer in model.ode_layers():
            p=state[name+".proj.weight"].double()
            up=state[name+".adapter.up.weight"].double()
            down=state[name+".adapter.down.weight"].double()
            delta=up@down
            values=[]
            for time in (.1,.3,.5,.7,.9):
                coefficient=float(time_coordinate(torch.tensor(time,dtype=torch.float64))) if model.variant=="time" else 1.
                matrix=p+coefficient*delta
                eigen=torch.linalg.eigvals(matrix)
                values.append(dict(time=time,singular_values=torch.linalg.svdvals(matrix).tolist(),
                    eigenvalue_real=eigen.real.tolist(),eigenvalue_imag=eigen.imag.tolist()))
            records[name]=dict(delta_frobenius=float(delta.norm()),base_frobenius=float(p.norm()),times=values)
        trajectory[str(step)]=records
    base.write(output.parent/"effective-generator-trajectory.json",trajectory)
    return metrics


def train_cell(config: Path,output: Path,dataset: str,arm: str,steps=5000):
    base.train_cell(config,output,dataset,arm,steps=steps,rollout=False,build_model=build,
                    arm_settings=ARMS,final_diagnostics=diagnostics)


def freeze(output: Path):
    output.mkdir(parents=True,exist_ok=False)
    shutil.copytree(ROOT/"synthetic",output/"synthetic",ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    for name in ("scripts/run_time_generator.py","scripts/preflight_time_generator.py",
                 "scripts/run_recurrent_depth.py","scripts/run_prediction_bridge.py",
                 "scripts/check_gate_geometry.py","tests/test_time_generator.py",DOC+"/protocol.md"):
        target=output/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    base.write(output/"source-sha256.json",{p.relative_to(output).as_posix():base.sha256(p)
                                           for p in output.rglob("*") if p.is_file()})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("freeze","run","cell"));parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--config",type=Path);parser.add_argument("--dataset",choices=base.DATASETS)
    parser.add_argument("--arm",choices=tuple(ARMS));parser.add_argument("--datasets",choices=base.DATASETS,nargs="+")
    parser.add_argument("--data-v1",type=Path);parser.add_argument("--data-product",type=Path);args=parser.parse_args()
    if args.action=="freeze":freeze(args.output.resolve())
    elif args.action=="cell":train_cell(args.config.resolve(),args.output.resolve(),args.dataset,args.arm)
    else:
        if not args.datasets or not args.data_v1 or not args.data_product:parser.error("run requires datasets and both data roots")
        config=dict(seed=0,steps=5000,workers=2,device="cpu",datasets=args.datasets,arms=ARMS,
            data_roots={d:str((args.data_product if d==base.DATASETS[2] else args.data_v1).resolve()) for d in args.datasets})
        base.run(args.output.resolve(),config,script_relative="scripts/run_time_generator.py",
                 protocol_relative=DOC+"/protocol.md")
