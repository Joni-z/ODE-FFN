"""Cross-fitted time-dependent matrix interventions at frozen checkpoints."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from synthetic.recurrent_depth import build
from synthetic.train import restore_model

DOC = "docs/research-time-operator-2026-10-26"
ARMS = {"ode_k2_native_final": ("ode_k2", "native"),
        "swiglu192_native_final": ("swiglu192", "native"),
        "swiglu128_native_deep": ("swiglu128", "native")}
TIMES = (.1,.3,.5,.7,.9)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def directions(gradients: torch.Tensor, phi: torch.Tensor, rho: float = .01) -> tuple[dict,dict]:
    if gradients.ndim!=2 or phi.shape!=(len(gradients),) or rho<=0:
        raise ValueError("invalid gradient/time coordinate shape or budget")
    if abs(float(phi.mean()))>1e-12 or abs(float(phi.square().mean())-1)>1e-12:
        raise ValueError("time coordinate must have zero mean and unit mean square")
    g0, g1 = gradients.mean(0), (gradients*phi[:,None]).mean(0)
    n0, n1 = g0.norm(), g1.norm()
    joint = torch.sqrt(n0.square()+n1.square())
    common = -rho*g0/n0 if n0>0 else torch.zeros_like(g0)
    zero = torch.zeros_like(gradients)
    values = dict(shared=common.expand_as(gradients).clone(),
        time_linear=-rho*(g0+phi[:,None]*g1)/joint if joint>0 else zero.clone(),
        time_reverse=-rho*(g0-phi[:,None]*g1)/joint if joint>0 else zero.clone())
    norms = gradients.norm(dim=1)
    denominator = norms[:,None]*norms[None,:]
    cosine = torch.where(denominator>0, (gradients@gradients.T)/denominator.clamp_min(1e-300),
                         torch.zeros_like(denominator))
    info = dict(pooled_gradient_norm=float(n0), temporal_gradient_norm=float(n1),
        mean_time_gradient_norm=float(norms.mean()), per_time_gradient_norm=norms.tolist(),
        cosine=cosine.tolist(), zero_pooled_gradient=bool(n0==0), zero_joint_gradient=bool(joint==0),
        first_order={k:float((gradients*v).sum(1).mean()) for k,v in values.items()})
    expected = dict(shared=-rho*float(n0), time_linear=-rho*float(joint),
        time_reverse=-rho*float((n0.square()-n1.square())/joint) if joint>0 else 0.)
    for key,value in values.items():
        np.testing.assert_allclose(info["first_order"][key], expected[key], rtol=1e-12, atol=1e-14)
        target = rho**2 if (n0>0 if key=="shared" else joint>0) else 0.
        np.testing.assert_allclose(float(value.square().sum(1).mean()),target,rtol=1e-12,atol=1e-16)
    return values,info


def selected_parameters(model, family: str) -> list[tuple[str,torch.Tensor]]:
    if family=="ode_k2":
        result = [(name+".proj.weight",layer.proj.weight) for name,layer in model.ode_layers()]
    else:
        result = [(f"raw.blocks.{i}.ffn.{name}.weight",getattr(block.ffn,name).weight)
                  for i,block in enumerate(model.raw.blocks) for name in ("w12","w3")]
    if len(result)!=6:
        raise ValueError("expected six core matrices")
    return result


def put(parameters: list, native: list, norms: list, displacement: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for parameter,value,norm in zip(parameters,native,norms):
            count = parameter.numel()
            parameter.copy_(value+norm*displacement[offset:offset+count].reshape_as(value))
            offset += count
    assert offset==len(displacement)


def csv_write(path: Path, rows: list[dict]) -> None:
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def run(inputs: list[Path], source: Path, output: Path) -> None:
    torch.set_num_threads(1)
    hashes=read(source/"source-sha256.json")
    for name,value in hashes.items():
        assert sha(source/name)==value
    for name in ("synthetic/models.py","synthetic/recurrent_depth.py","synthetic/train.py"):
        assert sha(ROOT/name)==hashes[name]
    output.mkdir(parents=True,exist_ok=False)
    write(output/"status.json",dict(state="RUNNING",pid=os.getpid(),completed=0,total=24))
    bindings={"script":sha(Path(__file__)),"protocol":sha(ROOT/DOC/"protocol.md"),
              "runtime_source":hashes,"inputs":{}}
    rows,info_rows,metadata=[],[],[]
    jobs=[]
    for group in inputs:
        config=read(group/"study.json")
        assert read(group/"status.json")["state"]=="COMPLETE"
        assert read(group/"provenance.json")["source_sha256"]==hashes
        for dataset in config["datasets"]:
            if dataset=="gaussian_aniso_2d":continue
            root=Path(config["data_roots"][dataset]);manifest=read(root/"manifest.json")
            target_file=root/dataset/"validation.npz"
            assert sha(target_file)==manifest["files"][dataset+"/validation.npz"]["sha256"]
            with np.load(target_file) as data:targets=data["x1"][:512].astype(np.float64)
            for path in (root/"manifest.json",target_file):bindings["inputs"][str(path)]=sha(path)
            panels=[]
            for t in TIMES:
                path=group/"banks"/dataset/f"panel_{t:.1f}.npz"
                with np.load(path) as data:
                    x,tensor_t,oracle=[data[key] for key in ("x","t","oracle")]
                assert x.shape[0]==512
                tt=tensor_t.astype(np.float64)
                pair=targets-(x.astype(np.float64)-tt*targets)/(1-tt)
                panels.append(dict(x=x,t=tensor_t,oracle=oracle,pair=pair))
                bindings["inputs"][str(path)]=sha(path)
            for arm in ARMS:
                run_path=group/"runs"/dataset/arm
                checkpoint=run_path/"step_0005000.pt"
                bindings["inputs"][str(checkpoint)]=sha(checkpoint)
                state=torch.load(checkpoint,map_location="cpu",weights_only=False)
                family,mode=ARMS[arm]
                model=build(targets.shape[1],family,mode)
                restore_model(model,state);model.eval()
                # A float32 replay validates the actual saved runtime before any double analysis.
                for time,panel in zip(TIMES,panels):
                    path=run_path/"diagnostics"/f"panel_{time:.1f}.npz"
                    with torch.no_grad():native=model(torch.from_numpy(panel["x"]),torch.from_numpy(panel["t"])).numpy()
                    with np.load(path) as data:np.testing.assert_array_equal(native,data["prediction"])
                    bindings["inputs"][str(path)]=sha(path)
                jobs.append((dataset,arm,family,model.double(),panels))
    assert len(jobs)==6
    write(output/"provenance.json",bindings)
    completed=0
    for dataset,arm,family,model,panels in jobs:
        named=selected_parameters(model,family)
        parameters=[p for _,p in named]
        selected_ids={id(p) for p in parameters}
        for parameter in model.parameters():parameter.requires_grad_(id(parameter) in selected_ids)
        native=[p.detach().clone() for p in parameters]
        norms=[p.norm().detach() for p in native]
        assert all(v>0 for v in norms)
        times=torch.tensor([float(p["t"][0,0]) for p in panels],dtype=torch.float64)
        phi=(times-times.mean())/torch.sqrt((times-times.mean()).square().mean())
        dimension=sum(p.numel() for p in parameters)
        zero=torch.zeros(dimension,dtype=torch.float64)
        data=[{k:torch.from_numpy(v).double() for k,v in panel.items()} for panel in panels]
        with torch.no_grad():
            native_predictions=[model(p["x"],p["t"]).detach() for p in data]
        metadata.append(dict(dataset=dataset,arm=arm,dimension=dimension,
            parameter_names=[n for n,_ in named],norms=[float(n) for n in norms],phi=phi.tolist()))
        for teacher in ("oracle","pair"):
            for fold in (0,1):
                calibration=slice(256*fold,256*(fold+1));evaluation=slice(256*(1-fold),256*(2-fold))
                gradients=[]
                for panel in data:
                    prediction=model(panel["x"][calibration],panel["t"][calibration])
                    loss=(prediction-panel[teacher][calibration]).square().mean()
                    values=torch.autograd.grad(loss,parameters)
                    gradients.append(torch.cat([(g*n).reshape(-1) for g,n in zip(values,norms)]).detach())
                gradients=torch.stack(gradients)
                variants,info=directions(gradients,phi)
                info.update(dataset=dataset,arm=arm,teacher=teacher,fold=fold,dimension=dimension)
                tag=f"{dataset}-{arm}-{teacher}-{fold}"
                torch.save(dict(gradient=gradients,phi=phi,norms=norms,
                    parameter_names=[n for n,_ in named],displacements=variants),output/(tag+".pt"))
                predictions=[]
                for index,panel in enumerate(data):
                    unit=variants["time_linear"][index]/.01
                    losses=[]
                    for sign in (1.,-1.):
                        put(parameters,native,norms,sign*1e-5*unit)
                        with torch.no_grad():
                            value=model(panel["x"][calibration],panel["t"][calibration])
                            losses.append(float((value-panel[teacher][calibration]).square().mean()))
                    derivative=(losses[0]-losses[1])/2e-5
                    expected=float(gradients[index]@unit)
                    np.testing.assert_allclose(derivative,expected,rtol=2e-4,atol=1e-8)
                    for variant,displacement in variants.items():
                        put(parameters,native,norms,displacement[index])
                        with torch.no_grad():prediction=model(panel["x"],panel["t"]).detach()
                        assert torch.isfinite(prediction).all()
                        predictions.append(prediction.numpy())
                        for split,selection in (("calibration",calibration),("evaluation",evaluation)):
                            base=native_predictions[index][selection];after=prediction[selection]
                            truth=panel["oracle"][selection]
                            before_loss=float((base-truth).square().mean())
                            after_loss=float((after-truth).square().mean())
                            alignment=float(2*((base-truth)*(after-base)).mean())
                            energy=float((after-base).square().mean())
                            np.testing.assert_allclose(after_loss-before_loss,alignment+energy,rtol=1e-10,atol=1e-12)
                            rows.append(dict(dataset=dataset,arm=arm,teacher=teacher,fold=fold,
                                time=float(times[index]),variant=variant,split=split,
                                native_oracle_mse=before_loss,changed_oracle_mse=after_loss,
                                oracle_difference=after_loss-before_loss,oracle_alignment=alignment,
                                finite_change_energy=energy,fit_loss=float((after-panel[teacher][selection]).square().mean()),
                                native_fit_loss=float((base-panel[teacher][selection]).square().mean())))
                np.savez_compressed(output/(tag+"-predictions.npz"),prediction=np.stack(predictions),
                    native=np.stack([p.numpy() for p in native_predictions]),
                    oracle=np.stack([p["oracle"].numpy() for p in data]))
                info_rows.append(info)
                put(parameters,native,norms,zero)
                assert all(torch.equal(p,v) for p,v in zip(parameters,native))
                completed+=1
                write(output/"status.json",dict(state="RUNNING",pid=os.getpid(),completed=completed,total=24,
                      latest=[dataset,arm,teacher,fold]))
                process=psutil.Process()
                cpu=process.cpu_times()
                with (output/"monitor.jsonl").open("a") as handle:
                    handle.write(json.dumps(dict(completed=completed,pid=os.getpid(),cpu_seconds=cpu.user+cpu.system,
                        rss_bytes=process.memory_info().rss,available_memory_bytes=psutil.virtual_memory().available))+"\n")
                print(json.dumps(dict(completed=completed,total=24,dataset=dataset,arm=arm,teacher=teacher,fold=fold)),flush=True)
    csv_write(output/"risk.csv",rows)
    write(output/"gradients.json",info_rows)
    write(output/"parameter-metadata.json",metadata)
    for path,value in bindings["inputs"].items():assert sha(Path(path))==value
    assert sha(Path(__file__))==bindings["script"]
    assert sha(ROOT/DOC/"protocol.md")==bindings["protocol"]
    write(output/"status.json",dict(state="COMPLETE",completed=24,total=24,pid=os.getpid(),risk_rows=len(rows)))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs",type=Path,nargs="+",required=True)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    try:
        run([p.resolve() for p in args.inputs],args.source.resolve(),args.output.resolve())
    except BaseException as error:
        path=args.output.resolve()/"status.json"
        if path.exists():
            progress=read(path)
            if progress.get("pid")==os.getpid() and progress.get("state")=="RUNNING":
                write(path,dict(progress,state="FAILED",error=repr(error)))
        raise
