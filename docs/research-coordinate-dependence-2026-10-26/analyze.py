"""Independent NumPy reconstruction and paired row-bootstrap analysis."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


TIMES=(0.1,0.3,0.5,0.7,0.9)


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path,value: Any) -> None:
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def table(path: Path,values: list[dict[str,Any]]) -> None:
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)


def ci(value: np.ndarray) -> tuple[float,float]:
    low,high=np.quantile(value,[.025,.975])
    return float(low),float(high)


def main(runtime: Path,output: Path) -> None:
    raw=runtime/"output-v2"
    assert load(raw/"status.json")["state"]=="COMPLETE"
    native_audit=load(raw/"audit.json")
    assert native_audit["native_panels_replayed"]==35 and native_audit["native_max_error"]==0
    assert native_audit["explicit_adjoint_cases"]==30
    for name,value in load(raw/"input-bindings.json").items():assert sha(Path(name))==value
    source=runtime/"source-v2"
    for name,value in load(source/"source-sha256.json").items():assert sha(source/name)==value
    assert sha(source/"source-sha256.json")==native_audit["source_manifest_sha256"]
    hashes={str(p.relative_to(runtime)):sha(p) for p in raw.rglob("*") if p.is_file()}
    output.mkdir(parents=True,exist_ok=False)
    spec=load(raw/"population.json")
    mu=np.array(spec["locations"]);var=np.array(spec["variances"]);rotation=np.array(spec["rotation"])
    np.testing.assert_allclose(rotation@rotation.T,np.eye(16),atol=1e-12)
    summaries=load(raw/"summaries.json")
    arms=list(dict.fromkeys(r["arm"] for r in summaries))
    assert len(arms)==7 and len(summaries)==35
    rows=np.zeros((len(arms),len(TIMES),4096,3),dtype=np.float64)
    max_summary_error=0.;cases=0
    with np.load(raw/"standardized-draws.npz") as draws:
        for ti,t in enumerate(TIMES):
            with np.load(raw/"banks"/f"time-{t:.1f}.npz") as bank:
                a=(1-t)*draws["source_A"]+t*(draws["sign_A"]*mu+.2*draws["target_noise_A"])
                b=(1-t)*draws["source_B"]+t*(draws["sign_B"]*mu+.2*draws["target_noise_B"])
                np.testing.assert_array_equal(a,bank["A"]);np.testing.assert_array_equal(b,bank["B"])
                scale=(1-t)**2+t**2*var
                oracle=((t*var-(1-t))*a+(1-t)*mu*np.tanh(t*mu*a/scale))/scale
                np.testing.assert_allclose(oracle,bank["oracle"],atol=1e-12,rtol=1e-12)
            for ai,arm in enumerate(arms):
                with np.load(raw/"fields"/arm/f"time-{t:.1f}.npz") as field:
                    ea=field["native"]-oracle;eh=field["hybrid_diagonal"]-oracle
                    # Reconstruct without importing the driver components helper.
                    total=(ea*ea+eh*eh)/2
                    other=(field["native"]-field["hybrid_diagonal"])**2/2
                    own=ea*eh
                    np.testing.assert_allclose(total,own+other,atol=1e-11,rtol=1e-10)
                    for key,expected in (("total",total),("own",own),("other",other)):
                        np.testing.assert_allclose(field[key],expected,atol=1e-11,rtol=1e-10)
                    row=next(r for r in summaries if r["arm"]==arm and r["time"]==t)
                    for index,(key,value) in enumerate((("total",total),("own",own),("other",other))):
                        observed=float(value.mean());assert np.isclose(observed,row[key],rtol=1e-10,atol=1e-12)
                        max_summary_error=max(max_summary_error,abs(observed-row[key]))
                        rows[ai,ti,:,index]=value.mean(axis=1)
                    cases+=1
    # Rows, not coordinate/time entries, are the independent Monte Carlo units.
    matrix=rows.transpose(2,0,1,3).reshape(4096,-1)
    rng=np.random.default_rng(0)
    indices=rng.integers(0,4096,size=(512,4096),dtype=np.uint16)
    bootstrap=np.empty((512,matrix.shape[1]))
    for i,index in enumerate(indices):
        weights=np.bincount(index,minlength=4096)
        bootstrap[i]=weights@matrix/4096
    bootstrap=bootstrap.reshape(512,len(arms),len(TIMES),3)
    np.savez_compressed(output/"row-metrics-and-bootstrap.npz",rows=rows,bootstrap_indices=indices,
                        bootstrap=bootstrap,arms=np.array(arms),times=np.array(TIMES))
    point=rows.mean(axis=2);pooled=point.mean(axis=1)
    pooled_bootstrap=bootstrap.mean(axis=2)
    mean_rows=[];time_rows=[]
    for ai,arm in enumerate(arms):
        row=dict(arm=arm)
        for j,key in enumerate(("total","own","other")):
            low,high=ci(pooled_bootstrap[:,ai,j])
            row.update({key:float(pooled[ai,j]),key+"_low":low,key+"_high":high})
        mean_rows.append(row)
        for ti,t in enumerate(TIMES):
            tr=dict(arm=arm,time=t)
            for j,key in enumerate(("total","own","other")):
                low,high=ci(bootstrap[:,ai,ti,j]);tr.update({key:float(point[ai,ti,j]),key+"_low":low,key+"_high":high})
            time_rows.append(tr)
    contrasts=[]
    for mode in ("direct","gradient"):
        parent="ode_k2_"+mode;pi=arms.index(parent)
        for ref in ("swiglu128_"+mode,"swiglu192_"+mode,"swiglu384_direct"):
            ri=arms.index(ref);delta=pooled[pi]-pooled[ri]
            boot_delta=pooled_bootstrap[:,pi]-pooled_bootstrap[:,ri]
            row=dict(parent=parent,reference=ref)
            for j,key in enumerate(("total","own","other")):
                low,high=ci(boot_delta[:,j]);row.update({"delta_"+key:float(delta[j]),"delta_"+key+"_low":low,"delta_"+key+"_high":high})
            positive=bool(delta[0]>0 and np.all(boot_delta[:,0]>0))
            share=float(delta[2]/delta[0]) if delta[0]>0 else None
            low,high=ci(boot_delta[:,2]/boot_delta[:,0]) if positive else (None,None)
            row.update(other_excess_share=share,share_low=low,share_high=high,
                       bootstrap_denominator_positive=positive)
            contrasts.append(row)
    primary=[r for r in contrasts if r["parent"]=="ode_k2_direct"]
    supported=all(r["share_low"] is not None and r["share_low"]>=.5 for r in primary)
    rejected=all(r["share_high"] is not None and r["share_high"]<.5 for r in primary)
    interpretation="supports majority extra-coordinate dependence" if supported else (
        "rejects majority extra-coordinate dependence" if rejected else "mixed or inconclusive")
    decision=dict(primary_interpretation=interpretation,primary_comparisons=primary,
                  architecture_selected=False,training_selected=False,
                  scope="Conditional fixed-model Monte Carlo intervals; no claim about training-run variability.")
    table(output/"means.csv",mean_rows);table(output/"times.csv",time_rows);table(output/"contrasts.csv",contrasts)
    write(output/"decision.json",decision)
    for name,value in hashes.items():assert sha(runtime/name)==value
    audit=dict(status="PASS",cases=cases,models=len(arms),independent_row_pairs=4096,bootstrap_resamples=512,
               maximum_summary_difference=max_summary_error,source_sha256=sha(Path(__file__)),
               frozen_output_sha256=hashes,denominator_rule="Ratio intervals undefined if bootstrap denominator crosses zero; no resamples discarded.")
    write(output/"audit.json",audit)
    print(json.dumps(dict(audit={k:v for k,v in audit.items() if k!="frozen_output_sha256"},decision=decision),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();main(args.runtime.resolve(),args.output.resolve())
