"""Native frozen-predictor evaluation under independent-coordinate redraws."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import psutil
import torch


DATASET = "rotated_product_bimodal_16d"
TIMES = (0.1, 0.3, 0.5, 0.7, 0.9)
N = 4096


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def components(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, ...]:
    total = (first**2 + second**2) / 2
    own = first * second
    other = (first - second)**2 / 2
    np.testing.assert_allclose(total, own + other, rtol=1e-11, atol=1e-12)
    return total, own, other


def self_check() -> dict[str, Any]:
    # Exhaustive independent ternary inputs: a nonlinear own-coordinate error
    # plus both additive and multiplicative dependence on another coordinate.
    grid = np.array(list(itertools.product((-1., 0., 1.), repeat=3)))
    first = np.repeat(grid, len(grid), axis=0)
    second = np.tile(grid, (len(grid), 1))
    def error(q: np.ndarray, i: int) -> np.ndarray:
        return q[:, i]**2 - 2/3 + (.5 + .2*q[:, i])*q[:, (i+1)%3]
    for i in range(3):
        hybrid = second.copy(); hybrid[:, i] = first[:, i]
        t, c, d = components(error(first, i), error(hybrid, i))
        np.testing.assert_allclose(c.mean(), 2/9, rtol=0, atol=1e-14)
        np.testing.assert_allclose(d.mean(), (.25+.04*2/3)*2/3, rtol=0, atol=1e-14)
        np.testing.assert_allclose(t.mean(), c.mean()+d.mean(), rtol=0, atol=1e-14)
        oracle_first=np.tanh(first[:, i])+first[:, i]/3
        oracle_hybrid=np.tanh(hybrid[:, i])+hybrid[:, i]/3
        np.testing.assert_array_equal(oracle_first, oracle_hybrid)
        np.testing.assert_allclose((error(first,i)-error(hybrid,i))**2/2,
            ((error(first,i)+oracle_first)-(error(hybrid,i)+oracle_hybrid))**2/2, atol=1e-14)
    return dict(status="PASS", exhaustive_pairs=len(first), own_nonlinearity=True,
                other_coordinate_variance=True, target_cancellation=True, finite_identity=True)


def field(model: torch.nn.Module, x: np.ndarray, clock: float, *, chunk: int=256) -> np.ndarray:
    dtype=next(model.parameters()).dtype
    result=[]
    with torch.no_grad():
        for begin in range(0,len(x),chunk):
            xx=torch.from_numpy(np.ascontiguousarray(x[begin:begin+chunk])).to(dtype=dtype)
            tt=torch.full((len(xx),1),clock,dtype=dtype)
            result.append(model(xx,tt).detach().cpu().numpy())
    return np.concatenate(result)


def main(raw: Path, output: Path) -> None:
    torch.set_num_threads(1)
    source=Path(__file__).resolve().parent
    manifest=load(source/"source-sha256.json")
    for name,digest in manifest.items():assert sha(source/name)==digest
    provenance=load(raw/"provenance.json")
    old_source=Path(provenance["source_root"])
    for name,digest in provenance["source_sha256"].items():assert sha(old_source/name)==digest
    sys.path.insert(0,str(old_source))
    from synthetic.conservative_readout import build
    from synthetic.train import restore_model
    from synthetic.data import suite
    from explicit_adjoint import ExplicitAdjointNet

    assert load(raw/"status.json")["state"]=="COMPLETE"
    config=load(raw/"study.json");assert len(config["arms"])==7
    population=suite()[DATASET]
    mu,variance,rotation=population.locations,population.variances,population.rotation
    np.testing.assert_array_equal(mu,np.geomspace(.75,2.5,16))
    np.testing.assert_array_equal(variance,np.full(16,.04))
    np.testing.assert_allclose(rotation.T@rotation,np.eye(16),atol=1e-12,rtol=0)
    output.mkdir(parents=True,exist_ok=False);(output/"fields").mkdir();(output/"banks").mkdir()
    start=time.monotonic();process=psutil.Process();telemetry=[];bindings={};models={}
    native_count=0;native_max=0.;explicit_max=0.
    def progress(**extra: Any) -> None:
        rec=dict(pid=process.pid,elapsed_seconds=time.monotonic()-start,
                 rss_MiB=process.memory_info().rss/2**20,available_GiB=psutil.virtual_memory().available/2**30,**extra)
        telemetry.append(rec)
        write(output/"status.json",dict(state="RUNNING",**rec))
        with (output/"telemetry.jsonl").open("a") as handle:handle.write(json.dumps(rec)+"\n")
        print(json.dumps(rec),flush=True)
    for path in (raw/"provenance.json",raw/"study.json",old_source/"source-sha256.json"):
        bindings[str(path)]=sha(path)
    for arm,setting in config["arms"].items():
        progress(phase="native compatibility",arm=arm)
        folder=raw/"runs"/DATASET/arm
        assert load(folder/"status.json")["state"]=="COMPLETE"
        for name in ("step_0005000.pt","result.json"):
            path=folder/name;bindings[str(path)]=sha(path)
        snapshot=torch.load(folder/"step_0005000.pt",map_location="cpu",weights_only=False)
        assert snapshot["step"]==5000
        model=build(16,setting["family"],setting["readout"]).eval()
        restore_model(model,snapshot)
        for parameter in model.parameters():parameter.requires_grad_(False)
        for clock in TIMES:
            bp=raw/"banks"/DATASET/f"panel_{clock:.1f}.npz"
            pp=folder/"diagnostics"/f"panel_{clock:.1f}.npz"
            bindings[str(bp)]=sha(bp);bindings[str(pp)]=sha(pp)
            with np.load(bp) as bank,np.load(pp) as saved:
                # These saved panels were generated together with input
                # Jacobians, so the readout used create_graph=True. Match
                # that actual evaluation context for bitwise native replay.
                with torch.enable_grad():
                    xx=torch.from_numpy(bank["x"]).requires_grad_(True)
                    prediction=model(xx,torch.from_numpy(bank["t"])).detach().numpy()
                np.testing.assert_array_equal(prediction,saved["prediction"])
                native_max=max(native_max,float(np.max(np.abs(prediction-saved["prediction"]))))
            native_count+=1
        models[arm]=model.double()
    rng=np.random.default_rng(0)
    draws={}
    for label in ("A","B"):
        draws["source_"+label]=rng.standard_normal((N,16))
        draws["sign_"+label]=2*rng.integers(0,2,(N,16))-1
        draws["target_noise_"+label]=rng.standard_normal((N,16))
    np.savez_compressed(output/"standardized-draws.npz",**draws)
    write(output/"population.json",population.to_dict())
    banks={}
    for clock in TIMES:
        a=(1-clock)*draws["source_A"]+clock*(draws["sign_A"]*mu+.2*draws["target_noise_A"])
        b=(1-clock)*draws["source_B"]+clock*(draws["sign_B"]*mu+.2*draws["target_noise_B"])
        c=(1-clock)**2+clock**2*variance
        oracle=(clock*variance-(1-clock))/c*a+(1-clock)*mu/c*np.tanh(clock*mu*a/c)
        physical=a@rotation.T
        np.testing.assert_allclose(physical@rotation,a,atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(population.velocity(physical,clock)@rotation,oracle,atol=1e-11,rtol=1e-11)
        banks[clock]=(a,b,oracle)
        np.savez_compressed(output/"banks"/f"time-{clock:.1f}.npz",A=a,B=b,oracle=oracle)
    summaries=[];explicit_checks=0
    for arm,model in models.items():
        setting=config["arms"][arm]
        explicit=ExplicitAdjointNet(model.raw).eval() if setting["readout"]=="gradient" else None
        for clock in TIMES:
            progress(phase="coordinate redraws",arm=arm,time=clock)
            a,b,oracle=banks[clock]
            physical=a@rotation.T
            va=field(model,physical,clock)@rotation
            vh=np.empty_like(va)
            for i in range(16):
                hybrid=b.copy();hybrid[:,i]=a[:,i]
                hybrid_physical=hybrid@rotation.T
                hybrid_output=field(model,hybrid_physical,clock)@rotation
                vh[:,i]=hybrid_output[:,i]
                if explicit is not None and i==0:
                    for x in (physical[:16],hybrid_physical[:16]):
                        standard=field(model,x,clock)
                        alternate=field(explicit,x,clock)
                        np.testing.assert_allclose(standard,alternate,rtol=1e-10,atol=1e-10)
                        explicit_max=max(explicit_max,float(np.max(np.abs(standard-alternate))))
                        explicit_checks+=1
            ea,eh=va-oracle,vh-oracle
            total,own,other=components(ea,eh)
            np.testing.assert_allclose(other,.5*(va-vh)**2,rtol=1e-10,atol=1e-12)
            case=output/"fields"/arm;case.mkdir(exist_ok=True)
            np.savez_compressed(case/f"time-{clock:.1f}.npz",native=va,hybrid_diagonal=vh,total=total,own=own,other=other)
            summaries.append(dict(arm=arm,time=clock,total=float(total.mean()),own=float(own.mean()),
                                  other=float(other.mean()),native_A_mse=float((ea**2).mean())))
    write(output/"summaries.json",summaries)
    for name,digest in bindings.items():assert sha(Path(name))==digest
    for name,digest in manifest.items():assert sha(source/name)==digest
    write(output/"input-bindings.json",bindings)
    audit=dict(status="PASS",models=len(models),native_panels_replayed=native_count,
        native_max_error=native_max,new_model_time_cases=len(summaries),pairs=N,dimension=16,
        explicit_adjoint_cases=explicit_checks,explicit_max_error=explicit_max,
        source_manifest_sha256=sha(source/"source-sha256.json"),elapsed_seconds=time.monotonic()-start,
        peak_sampled_RSS_MiB=max(r["rss_MiB"] for r in telemetry),
        minimum_available_GiB=min(r["available_GiB"] for r in telemetry))
    write(output/"audit.json",audit);write(output/"status.json",dict(state="COMPLETE",**audit))
    print(json.dumps(audit),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check",action="store_true")
    parser.add_argument("--raw",type=Path);parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check()))
    elif args.raw is None or args.output is None:
        parser.error("--raw and --output required")
    else:
        try:
            main(args.raw.resolve(),args.output.resolve())
        except Exception as error:
            if args.output.is_dir():
                write(args.output/"status.json",dict(state="FAILED",error=repr(error)))
            raise
