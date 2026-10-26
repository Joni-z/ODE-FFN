"""Replay all seven actual cold endpoints, trajectories and the fixed screen."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F


DATASET="rotated_product_bimodal_16d"


def load(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
def table(path,rows):
    with Path(path).open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def exact(a,b):
    if isinstance(a,torch.Tensor):assert isinstance(b,torch.Tensor) and torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:exact(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert type(a)==type(b) and len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b


def main(runtime,output):
    torch.set_num_threads(1)
    raw=runtime/"output";source=runtime/"source-v2"
    assert load(raw/"status.json")["state"]=="COMPLETE"
    manifest=load(source/"source-sha256.json")
    for name,value in manifest.items():assert sha(source/name)==value
    config=load(raw/"study.json");reference=Path(config["reference_output"])
    assert config["steps"]==5000 and config["seed"]==0 and len(config["arms"])==7
    sys.path.insert(0,str(source))
    from synthetic.local_adjoint import build,LocalAdjointODEFFN
    from synthetic.train import restore_model,load_targets,generated_samples
    from synthetic.conservative_readout import compute_budget
    from synthetic.population import fingerprint_batch
    from scripts.check_gate_geometry import jacobian
    output.mkdir(parents=True,exist_ok=False)
    bindings={};metrics=[];curves=[];spectra=[];activity=[];finite=[]
    for path in (raw/"study.json",raw/"provenance.json",runtime/"profile.json",runtime/"READY-cold.json"):
        bindings[str(path)]=sha(path)
    banks=raw/"banks"/DATASET
    for path in banks.glob("*.npz"):
        bindings[str(path)]=sha(path)
        assert path.read_bytes()==(reference/"banks"/DATASET/path.name).read_bytes()
    data=Path(config["data_roots"][DATASET]);train=data/DATASET/"train.npz"
    bindings[str(train)]=sha(train);bindings[str(data/"manifest.json")]=sha(data/"manifest.json")
    targets=torch.from_numpy(load_targets(train,16))
    rng=torch.Generator().manual_seed(100000);digest=hashlib.sha256(b"").hexdigest()
    expected={0:(rng.get_state().clone(),digest)}
    for step in range(1,5001):
        y=targets[torch.randint(len(targets),(512,),generator=rng)]
        z=torch.randn(y.shape,generator=rng);t=torch.rand((512,1),generator=rng)
        digest=fingerprint_batch(digest,y,z,t)
        if step in (1000,2500,5000):expected[step]=(rng.get_state().clone(),digest)
    old_arms=dict(original="ode_k2_direct",swiglu128="swiglu128_direct",swiglu192="swiglu192_direct")
    checked_states=old_states=local_checks=0
    local_value_error=local_gradient_error=0.
    for arm in config["arms"]:
        folder=raw/"runs"/DATASET/arm
        assert load(folder/"status.json")["state"]=="COMPLETE"
        record=load(folder/"result.json");bindings[str(folder/"result.json")]=sha(folder/"result.json")
        initial=None
        for step in (0,1000,2500,5000):
            path=folder/f"step_{step:07d}.pt";bindings[str(path)]=sha(path)
            snapshot=torch.load(path,map_location="cpu",weights_only=False)
            assert snapshot["step"]==step
            exact(snapshot["data_rng"],expected[step][0]);assert snapshot["input_stream_digest"]==expected[step][1]
            if initial is None:initial=snapshot["model"]
            if arm in old_arms:
                old_path=reference/"runs"/DATASET/old_arms[arm]/path.name
                old=torch.load(old_path,map_location="cpu",weights_only=False);bindings[str(old_path)]=sha(old_path)
                for key in ("model","optimizer","runtime_buffers","data_rng","torch_rng","input_stream_digest"):
                    exact(snapshot[key],old[key])
                old_states+=1
            model=build(16,arm).eval();restore_model(model,snapshot)
            saved=load(folder/f"spectrum_{step:07d}.json")
            bindings[str(folder/f"spectrum_{step:07d}.json")]=sha(folder/f"spectrum_{step:07d}.json")
            for name,value in saved.items():
                matrix=snapshot["model"]["raw."+name+".proj.weight"].double().numpy()
                singular=np.linalg.svd(matrix,compute_uv=False)
                eigen=np.linalg.eigvals(matrix)
                stored=np.array(value["eigenvalue_real"])+1j*np.array(value["eigenvalue_imag"])
                distances=np.abs(eigen[:,None]-stored[None,:])
                assert max(distances.min(0).max(),distances.min(1).max())<1e-8
                np.testing.assert_allclose(singular,value["singular_values"],rtol=1e-10,atol=1e-11)
                p0=initial["raw."+name+".proj.weight"].double().numpy()
                spectra.append(dict(arm=arm,step=step,layer=name,sigma_max=float(singular[0]),
                    spectral_radius=float(np.abs(eigen).max()),frobenius=float(np.linalg.norm(matrix)),
                    drift=float(np.linalg.norm(matrix-p0))))
            if arm.startswith("local"):
                doubled=copy.deepcopy(model).double();captures={};hooks=[]
                for name,module in doubled.named_modules():
                    if isinstance(module,LocalAdjointODEFFN):
                        def capture(ffn,args,name=name):captures[name]=args[0].detach().clone()
                        hooks.append(module.register_forward_pre_hook(capture))
                try:
                    with np.load(banks/"probe.npz") as bank:
                        doubled(torch.from_numpy(bank["x"][:3]).double(),torch.from_numpy(bank["t"][:3]).double())
                finally:
                    for hook in hooks:hook.remove()
                for name,h in captures.items():
                    ffn=dict(doubled.named_modules())[name];encoder=copy.deepcopy(ffn.ode1)
                    h=h.requires_grad_(True);hh=h.detach().clone().requires_grad_(True)
                    actual=ffn(h);y=encoder(hh);q=F.silu(y)
                    desired=torch.autograd.grad(y,hh,q,create_graph=True)[0]
                    if not ffn.include_clock:
                        a=encoder.step_size(hh);ga=torch.autograd.grad(a.sum(),hh,create_graph=True)[0]
                        ph=encoder.proj(hh);u=ph+a*encoder.proj(ph)
                        desired=desired-ga*(u*q).sum(-1,keepdim=True)
                    torch.testing.assert_close(actual,desired,rtol=1e-10,atol=1e-10)
                    g1=torch.autograd.grad(actual.square().mean(),(h,*ffn.parameters()))
                    g2=torch.autograd.grad(desired.square().mean(),(hh,*encoder.parameters()))
                    local_value_error=max(local_value_error,float((actual-desired).abs().max()))
                    for a,b in zip(g1,g2):
                        torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-9)
                        local_gradient_error=max(local_gradient_error,float((a-b).abs().max()))
                    local_checks+=1
                activity.extend(dict(arm=arm,step=step,**r) for r in load(folder/f"visits_{step:07d}.json"))
            checked_states+=1
        costs=compute_budget(model)
        for key in ("parameters","dense_forward_macs","dense_training_forward_backward_macs"):assert costs[key]==record[key]
        for p in model.parameters():p.requires_grad_(False)
        with np.load(banks/"field.npz") as bank:
            values=[]
            with torch.no_grad():
                for start in range(0,len(bank["x"]),1024):
                    values.append(model(torch.from_numpy(bank["x"][start:start+1024]),torch.from_numpy(bank["t"][start:start+1024])).numpy())
            values=np.concatenate(values)
            with np.load(folder/"diagnostics/field.npz") as saved:np.testing.assert_array_equal(values,saved["prediction"])
            assert float(np.mean((values-bank["oracle"])**2))==record["oracle_mse"]
            assert float(np.mean((values-bank["pair"])**2))==record["pair_mse"]
        panel_errors=[]
        for clock in (.1,.3,.5,.7,.9):
            with np.load(banks/f"panel_{clock:.1f}.npz") as bank:
                value,derivative=jacobian(model,torch.from_numpy(bank["x"]),torch.from_numpy(bank["t"]))
                with np.load(folder/f"diagnostics/panel_{clock:.1f}.npz") as saved:
                    np.testing.assert_array_equal(value.numpy(),saved["prediction"])
                    np.testing.assert_array_equal(derivative.numpy(),saved["jacobian"])
                error=float(np.mean(np.sum((derivative.numpy()-bank["oracle_jacobian"])**2,axis=(1,2))))
                field_error=float(np.mean((value.numpy()-bank["oracle"])**2))
            expected_panel=next(r for r in record["panels"] if r["time"]==clock)
            assert error==expected_panel["jacobian_error"] and field_error==expected_panel["field_mse"]
            panel_errors.append(error)
        assert float(np.mean(panel_errors))==record["jacobian_error"]
        with np.load(banks/"sampling.npz") as bank:
            sampled=generated_samples(model,torch.from_numpy(bank["source"]),32)
            with np.load(folder/"diagnostics/samples.npz") as saved:np.testing.assert_array_equal(sampled.numpy(),saved["generated"])
            a=sampled.numpy().astype(np.float64)@bank["directions"].astype(np.float64).T
            b=bank["target"].astype(np.float64)@bank["directions"].astype(np.float64).T
            sw=float(np.sqrt(np.mean((np.sort(a,axis=0)-np.sort(b,axis=0))**2)))
            assert np.isclose(sw,record["sw2"],rtol=1e-5,atol=1e-7)
        if arm=="local_full":
            doubled=copy.deepcopy(model).double()
            for p in doubled.parameters():p.requires_grad_(True)
            with np.load(banks/"probe.npz") as bank:x,t=[torch.from_numpy(bank[k][:8]).double() for k in ("x","t")]
            for name in ("raw.blocks.0.ffn.ode1.proj.weight","raw.blocks.0.ffn.ode1.t_func.2.weight"):
                p=dict(doubled.named_parameters())[name];old=p.detach().clone()
                direction=torch.linspace(-1,1,p.numel(),dtype=p.dtype).reshape_as(p);direction/=direction.norm()
                g=torch.autograd.grad(doubled(x,t).square().mean(),p)[0];analytic=float((g*direction).sum())
                eps=1e-5
                try:
                    with torch.no_grad():
                        p.copy_(old+eps*direction);plus=float(doubled(x,t).square().mean())
                        p.copy_(old-eps*direction);minus=float(doubled(x,t).square().mean())
                finally:
                    with torch.no_grad():p.copy_(old)
                error=abs((plus-minus)/(2*eps)-analytic)/max(1.,abs(analytic));assert error<1e-6
                finite.append(dict(parameter=name,analytic=analytic,numeric=(plus-minus)/(2*eps),relative_error=error))
        with (folder/"metrics.jsonl").open() as f:
            curves.extend(dict(arm=arm,**json.loads(line)) for line in f)
        metrics.append(dict(arm=arm,oracle_mse=record["oracle_mse"],jacobian_error=record["jacobian_error"],
            sw2=record["sw2"],sw2_float64_recomputed=sw,**costs))
        print(json.dumps(dict(verified=arm,primary=record["oracle_mse"])),flush=True)
    by_arm={r["arm"]:r for r in metrics};candidate=by_arm["local_full"]
    generic=[by_arm[n] for n in ("tied128","tied192","swiglu128","swiglu192")]
    field=candidate["oracle_mse"]
    base_ok=field<=.9*by_arm["original"]["oracle_mse"] and field<=.95*by_arm["local_frozen_clock"]["oracle_mse"]
    quality=field<=.9*min(r["oracle_mse"] for r in generic)
    reuse=(field<=1.05*min(by_arm[n]["oracle_mse"] for n in ("swiglu128","swiglu192"))
        and field<=.9*min(by_arm[n]["oracle_mse"] for n in ("tied128","tied192"))
        and candidate["parameters"]<=.75*by_arm["swiglu128"]["parameters"])
    guards={key:candidate[key]<=1.1*min(r[key] for r in generic) for key in ("jacobian_error","sw2")}
    eligible=bool(base_ok and (quality or reuse) and all(guards.values()))
    decision=dict(eligible_for_other_tasks=eligible,base_contrasts=bool(base_ok),quality_route=bool(quality),
        parameter_reuse_route=bool(reuse),guards=guards,accepted_architecture=False,
        action="register fixed other-task replication" if eligible else "stop this local-adjoint setting without tuning or another control nomination")
    for name,value in bindings.items():assert sha(name)==value
    for name,value in manifest.items():assert sha(source/name)==value
    table(output/"comparison.csv",metrics)
    fields=list(dict.fromkeys(k for row in curves for k in row))
    table(output/"learning-curves.csv",[{k:r.get(k,"") for k in fields} for r in curves])
    table(output/"P-trajectories.csv",spectra);write(output/"local-activity.json",activity)
    table(output/"finite-directions.csv",finite);write(output/"decision.json",decision);write(output/"input-bindings.json",bindings)
    audit=dict(status="PASS",checkpoints=checked_states,old_control_states_bitwise=old_states,local_VJP_gradient_cases=local_checks,
        field_replays=7,panel_value_J_replays=35,sampling_replays=7,P_spectrum_records=len(spectra),
        finite_directions=len(finite),max_local_value_error=local_value_error,max_local_gradient_error=local_gradient_error,
        source_manifest_sha256=sha(source/"source-sha256.json"),analyzer_sha256=sha(__file__))
    write(output/"audit.json",audit);print(json.dumps(dict(audit=audit,decision=decision)),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--runtime",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    main(args.runtime.resolve(),args.output.resolve())
