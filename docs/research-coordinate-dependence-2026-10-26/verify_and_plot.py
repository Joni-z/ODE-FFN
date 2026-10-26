"""Check saved redraw fields/bootstrap without importing experiment helpers."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path: Path) -> list[dict[str,str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main(runtime: Path,report: Path) -> None:
    raw=runtime/"output-v2";analysis=runtime/"analysis"
    audited=json.loads((analysis/"audit.json").read_text())
    for name,value in audited["frozen_output_sha256"].items():assert sha(runtime/name)==value
    with np.load(analysis/"row-metrics-and-bootstrap.npz") as data:
        arms=data["arms"].tolist();times=data["times"].tolist()
        expected_rows=data["rows"];indices=data["bootstrap_indices"];expected_boot=data["bootstrap"]
    rows=np.empty_like(expected_rows)
    cases=0
    for ai,arm in enumerate(arms):
        for ti,t in enumerate(times):
            with np.load(raw/"banks"/f"time-{t:.1f}.npz") as bank:
                oracle=bank["oracle"]
            with np.load(raw/"fields"/arm/f"time-{t:.1f}.npz") as data:
                a,h=data["native"],data["hybrid_diagonal"]
            ea,eh=a-oracle,h-oracle
            total=(np.mean(ea*ea,axis=1)+np.mean(eh*eh,axis=1))/2
            other=np.mean((a-h)**2,axis=1)/2
            # Subtraction here independently checks the cross-product estimator.
            own=total-other
            rows[ai,ti]=np.stack((total,own,other),axis=-1)
            cases+=1
    np.testing.assert_allclose(rows,expected_rows,rtol=1e-10,atol=2e-12)
    generated=np.random.default_rng(0).integers(0,4096,(512,4096),dtype=np.uint16)
    np.testing.assert_array_equal(generated,indices)
    boot=np.empty_like(expected_boot)
    for i,index in enumerate(indices):
        # Direct resampled-row mean, unlike the analyzer's bincount weights.
        boot[i]=rows[:,:,index,:].mean(axis=2)
    np.testing.assert_allclose(boot,expected_boot,rtol=1e-10,atol=2e-12)
    point=rows.mean(axis=2).mean(axis=1);draws=boot.mean(axis=2)
    means=table(analysis/"means.csv");time_rows=table(analysis/"times.csv");contrasts=table(analysis/"contrasts.csv")
    def check(x: float,value: str) -> None:
        assert np.isclose(x,float(value),rtol=1e-9,atol=2e-12),(x,value)
    for ai,arm in enumerate(arms):
        row=next(r for r in means if r["arm"]==arm)
        for j,key in enumerate(("total","own","other")):
            check(point[ai,j],row[key])
            bounds=np.percentile(draws[:,ai,j],[2.5,97.5])
            check(bounds[0],row[key+"_low"]);check(bounds[1],row[key+"_high"])
        for ti,t in enumerate(times):
            row=next(r for r in time_rows if r["arm"]==arm and float(r["time"])==t)
            for j,key in enumerate(("total","own","other")):
                check(rows[ai,ti,:,j].mean(),row[key])
                bounds=np.percentile(boot[:,ai,ti,j],[2.5,97.5])
                check(bounds[0],row[key+"_low"]);check(bounds[1],row[key+"_high"])
    primary=[]
    for row in contrasts:
        pi,ri=arms.index(row["parent"]),arms.index(row["reference"])
        delta=point[pi]-point[ri];sample=draws[:,pi]-draws[:,ri]
        for j,key in enumerate(("total","own","other")):
            check(delta[j],row["delta_"+key]);bounds=np.percentile(sample[:,j],[2.5,97.5])
            check(bounds[0],row["delta_"+key+"_low"]);check(bounds[1],row["delta_"+key+"_high"])
        if delta[0]>0:
            check(delta[2]/delta[0],row["other_excess_share"])
            assert np.all(sample[:,0]>0)
            bounds=np.percentile(sample[:,2]/sample[:,0],[2.5,97.5])
            check(bounds[0],row["share_low"]);check(bounds[1],row["share_high"])
        else:
            assert row["other_excess_share"]==row["share_low"]==row["share_high"]==""
        if row["parent"]=="ode_k2_direct":primary.append(row)
    assert len(primary)==3 and all(float(r["share_high"])<.5 for r in primary)
    decision=json.loads((analysis/"decision.json").read_text())
    assert decision["primary_interpretation"]=="rejects majority extra-coordinate dependence"
    audit=dict(status="PASS",cases=cases,bootstrap_resamples=len(indices),mean_rows=len(means),time_rows=len(time_rows),
        contrasts=len(contrasts),maximum_row_difference=float(np.max(np.abs(rows-expected_rows))),
        maximum_bootstrap_difference=float(np.max(np.abs(boot-expected_boot))),source_sha256=sha(Path(__file__)),
        experiment_audit_sha256=sha(raw/"audit.json"),analyzer_audit_sha256=sha(analysis/"audit.json"))
    (runtime/"independent-verification.json").write_text(json.dumps(audit,indent=2)+"\n")

    figures=report/"figures";figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    color_own,color_other="#0072B2","#E69F00"
    mmap={r["arm"]:r for r in means}
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.8),sharey=True)
    for ax,mode,title in zip(axes,("direct","gradient"),("Direct velocity readout","Complete potential-gradient readout")):
        names=("ode_k2_"+mode,"swiglu128_"+mode,"swiglu192_"+mode,"swiglu384_direct")
        own=[float(mmap[n]["own"]) for n in names];other=[float(mmap[n]["other"]) for n in names]
        x=np.arange(4)
        lower=ax.bar(x,own,color=color_own,label="Own-coordinate conditional error")
        upper=ax.bar(x,other,bottom=own,color=color_other,label="Dependence on other coordinates")
        totals=np.array([float(mmap[n]["total"]) for n in names])
        low=np.array([float(mmap[n]["total_low"]) for n in names]);high=np.array([float(mmap[n]["total_high"]) for n in names])
        ax.errorbar(x,totals,yerr=(totals-low,high-totals),fmt="none",ecolor="black",capsize=3,lw=1)
        if mode=="gradient":lower[-1].set_hatch("///");upper[-1].set_hatch("///")
        ax.set_xticks(x,("ODE","S128","S192","S384"));ax.set_title(title)
        ax.set_ylim(0,max(float(r["total_high"]) for r in means)*1.15)
    axes[0].set_ylabel("Paired population error; five-time mean")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,1.03),ncol=2,frameon=False,fontsize=9)
    fig.text(.5,.01,"Bars: conditional error decomposition. Whiskers: 95% paired row-bootstrap intervals for fixed trained functions. Hatched S384 is direct.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.95))
    for ext in ("png","pdf"):fig.savefig(figures/("all-model-coordinate-errors."+ext),dpi=180,bbox_inches="tight")
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.7))
    y=np.arange(3)
    shares=np.array([float(r["other_excess_share"])*100 for r in primary])
    lo=np.array([float(r["share_low"])*100 for r in primary]);hi=np.array([float(r["share_high"])*100 for r in primary])
    axes[0].errorbar(shares,y,xerr=(shares-lo,hi-shares),fmt="o",color=color_other,capsize=4)
    axes[0].axvline(0,color="black",lw=.8)
    axes[0].set_yticks(y,("vs SwiGLU128","vs SwiGLU192","vs SwiGLU384"))
    axes[0].set_xlim(-.9,.8);axes[0].set_ylim(-.5,2.5)
    axes[0].set_xlabel("Other-coordinate share of direct ODE excess (%)")
    axes[0].set_title("Every upper bound is below 0.55%\nMajority criterion: 50%",fontsize=11)
    for i,value in enumerate(shares):axes[0].annotate(f"{value:.2f}%",(value,i),xytext=(0,12),textcoords="offset points",ha="center",fontsize=9)
    direct=mmap["ode_k2_direct"];gradient=mmap["ode_k2_gradient"]
    x=np.arange(2);width=.34
    own=np.array([float(direct["own"]),float(gradient["own"])])
    other=np.array([float(direct["other"]),float(gradient["other"])])
    b0=axes[1].bar(x-width/2,own,width,color=color_own,label="Own-coordinate error")
    b1=axes[1].bar(x+width/2,other,width,color=color_other,label="Other-coordinate dependence")
    axes[1].bar_label(b0,fmt="%.4f",padding=3,fontsize=9);axes[1].bar_label(b1,fmt="%.4f",padding=3,fontsize=9)
    axes[1].set_xticks(x,("Direct ODE","Potential ODE"));axes[1].set_ylim(0,.36)
    axes[1].set_ylabel("Five-time mean contribution")
    axes[1].set_title("The readout gain changes own-coordinate fitting",fontsize=11)
    axes[1].legend(frameon=False,fontsize=8,loc="upper right")
    fig.text(.5,.01,"4096 independent redraw pairs, reused across all models and five times. Intervals describe probe sampling, not training-run variation.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.06,1,1))
    for ext in ("png","pdf"):fig.savefig(figures/("mechanism-and-readout."+ext),dpi=180,bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(audit,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime",type=Path,required=True);parser.add_argument("--report",type=Path,required=True)
    args=parser.parse_args();main(args.runtime.resolve(),args.report.resolve())
