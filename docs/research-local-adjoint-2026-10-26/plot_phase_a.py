"""Independent screen arithmetic and plots from verified seven-arm evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def read_csv(path):
    with Path(path).open(newline="") as handle:return list(csv.DictReader(handle))


def main(runtime,report):
    analysis=runtime/"analysis";raw=runtime/"output"
    audit=json.loads((analysis/"audit.json").read_text());assert audit["status"]=="PASS"
    rows=read_csv(analysis/"comparison.csv");by={r["arm"]:r for r in rows}
    for row in rows:
        result=json.loads((raw/"runs/rotated_product_bimodal_16d"/row["arm"]/"result.json").read_text())
        for key in ("oracle_mse","jacobian_error","sw2","parameters","dense_forward_macs","dense_training_forward_backward_macs"):
            assert float(row[key])==float(result[key])
    full=by["local_full"]
    generics=[by[n] for n in ("tied128","tied192","swiglu128","swiglu192")]
    ratios=dict(vs_original=float(full["oracle_mse"])/float(by["original"]["oracle_mse"]),
        vs_omitted=float(full["oracle_mse"])/float(by["local_frozen_clock"]["oracle_mse"]),
        vs_best_generic=float(full["oracle_mse"])/min(float(r["oracle_mse"]) for r in generics),
        vs_best_swiglu=float(full["oracle_mse"])/min(float(by[n]["oracle_mse"]) for n in ("swiglu128","swiglu192")),
        vs_best_tied=float(full["oracle_mse"])/min(float(by[n]["oracle_mse"]) for n in ("tied128","tied192")),
        parameter_ratio_swiglu128=float(full["parameters"])/float(by["swiglu128"]["parameters"]),
        jacobian_ratio=float(full["jacobian_error"])/min(float(r["jacobian_error"]) for r in generics),
        sw2_ratio=float(full["sw2"])/min(float(r["sw2"]) for r in generics))
    checks=dict(base_contrasts=ratios["vs_original"]<=.9 and ratios["vs_omitted"]<=.95,
        quality_route=ratios["vs_best_generic"]<=.9,
        parameter_reuse_route=ratios["vs_best_swiglu"]<=1.05 and ratios["vs_best_tied"]<=.9 and ratios["parameter_ratio_swiglu128"]<=.75,
        jacobian_guard=ratios["jacobian_ratio"]<=1.1,sw2_guard=ratios["sw2_ratio"]<=1.1)
    eligible=checks["base_contrasts"] and (checks["quality_route"] or checks["parameter_reuse_route"]) and checks["jacobian_guard"] and checks["sw2_guard"]
    actual=json.loads((analysis/"decision.json").read_text())
    assert eligible==actual["eligible_for_other_tasks"]
    for key in ("base_contrasts","quality_route","parameter_reuse_route"):assert checks[key]==actual[key]
    assert checks["jacobian_guard"]==actual["guards"]["jacobian_error"]
    assert checks["sw2_guard"]==actual["guards"]["sw2"]
    evidence=dict(status="PASS",ratios=ratios,checks=checks,eligible=eligible,
        plot_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        audit_sha256=hashlib.sha256((analysis/"audit.json").read_bytes()).hexdigest())
    (analysis/"independent-screen.json").write_text(json.dumps(evidence,indent=2)+"\n")
    for name in ("comparison.csv","learning-curves.csv","P-trajectories.csv","finite-directions.csv"):
        shutil.copy2(analysis/name,report/name)
    figures=report/"figures";figures.mkdir(exist_ok=True)
    order=("original","local_full","local_frozen_clock","tied128","tied192","swiglu128","swiglu192")
    labels=("Original ODE","Local full adjoint","Local omitted clock","Tied SiLU128","Tied SiLU192","SwiGLU128","SwiGLU192")
    colors={"original":"#777777","local_full":"#D55E00","local_frozen_clock":"#E69F00",
            "tied128":"#CC79A7","tied192":"#882255","swiglu128":"#56B4E9","swiglu192":"#0072B2"}
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    fig,axes=plt.subplots(1,3,figsize=(11.7,4.9),sharey=True)
    for ax,key,title in zip(axes,("oracle_mse","jacobian_error","sw2"),("Uniform-time field MSE","Five-panel Jacobian error","Common-source SW2")):
        values=[float(by[n][key]) for n in order]
        ax.scatter(values,np.arange(7),c=[colors[n] for n in order],s=42)
        ax.set_yticks(np.arange(7),labels);ax.invert_yaxis()
        if key=="oracle_mse":ax.set_xscale("log")
        else:ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_xlabel("Lower is better");ax.set_title(title)
        ax.margins(x=.25)
        for i,value in enumerate(values):ax.annotate(f"{value:.4g}",(value,i),xytext=(6,0),textcoords="offset points",va="center",fontsize=8)
    fig.suptitle("Local adjoint improves the original ODE but misses the generic frontier",fontsize=14)
    fig.text(.5,.01,"All seven fixed endpoints. No uncertainty bars or independent-repeat claim. Each metric retains its original evaluation definition.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.95))
    for ext in ("png","pdf"):fig.savefig(figures/("endpoint-comparison."+ext),dpi=180,bbox_inches="tight")
    plt.close(fig)
    curves=read_csv(analysis/"learning-curves.csv");spectra=read_csv(analysis/"P-trajectories.csv")
    activity=json.loads((analysis/"local-activity.json").read_text())
    fig,axes=plt.subplots(1,3,figsize=(12,4.6))
    for name,label in zip(order,labels):
        history=[r for r in curves if r["arm"]==name and r.get("oracle_mse","")!=""]
        axes[0].plot([int(r["step"]) for r in history],[float(r["oracle_mse"]) for r in history],
                     marker="o",ms=3,color=colors[name],label=label)
    axes[0].set_yscale("log");axes[0].set_xlabel("Training update");axes[0].set_ylabel("Validation oracle MSE")
    axes[0].set_title("All saved validation checkpoints")
    axes[0].legend(fontsize=7,frameon=False)
    for name,label in zip(order[:3],labels[:3]):
        mean=[];low=[];high=[]
        for step in (0,1000,2500,5000):
            values=[float(r["sigma_max"]) for r in spectra if r["arm"]==name and int(r["step"])==step]
            mean.append(np.mean(values));low.append(min(values));high.append(max(values))
        axes[1].plot((0,1000,2500,5000),mean,marker="o",color=colors[name],label=label)
        axes[1].fill_between((0,1000,2500,5000),low,high,color=colors[name],alpha=.12)
    axes[1].set_xlabel("Training update");axes[1].set_ylabel("Largest singular value of raw P")
    axes[1].set_title("Learned internal P; bands span layers")
    axes[1].legend(fontsize=7,frameon=False)
    for name,label,style in (("local_full","Full arm: executed clock term","-"),
                             ("local_frozen_clock","Omitted arm: hypothetical term","--")):
        values=[]
        for step in (0,1000,2500,5000):
            selected=[r for r in activity if r["arm"]==name and r["step"]==step]
            values.append(np.mean([r["clock_adjoint_rms"]/r["encoder"]["input_rms"] for r in selected]))
        axes[2].plot((0,1000,2500,5000),values,style,marker="o",color=colors[name],label=label)
    axes[2].set_xlabel("Training update");axes[2].set_ylabel("Clock-adjoint RMS / FFN-input RMS")
    axes[2].set_title("Clock response is active")
    axes[2].legend(fontsize=7,frameon=False)
    fig.text(.5,.01,"Internal probes do not bound the full field. Layer ranges are descriptive; the omitted arm does not execute its plotted counterfactual term.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.06,1,1))
    for ext in ("png","pdf"):fig.savefig(figures/("learning-and-P-trajectories."+ext),dpi=180,bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(evidence,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--runtime",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True);args=parser.parse_args()
    main(args.runtime.resolve(),args.report.resolve())
