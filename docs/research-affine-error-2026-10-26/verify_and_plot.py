"""Independent QR replay from original saved fields, then descriptive plots."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_square(a: np.ndarray) -> float:
    return float(np.sum(a*a)/a.size)


def qr_fit(x: np.ndarray, e: np.ndarray) -> np.ndarray:
    q,r=np.linalg.qr(x,mode="reduced")
    return np.linalg.solve(r,q.T@e)


def main(raw: Path, runtime: Path, report: Path) -> None:
    output=runtime/"output"
    assert json.loads((output/"status.json").read_text())["state"]=="COMPLETE"
    original_hashes={str(p.relative_to(runtime)):digest(p) for p in output.rglob("*") if p.is_file()}
    for name,value in json.loads((output/"input-bindings.json").read_text()).items():
        assert digest(Path(name))==value
    for name,value in json.loads((runtime/"source/source-sha256.json").read_text()).items():
        assert digest(runtime/"source"/name)==value
    panels=rows(output/"panels.csv");aggregates=rows(output/"aggregates.csv");gaps=rows(output/"gaps.csv")
    reconstructed=[];max_difference=0.0
    for row in panels:
        ds,arm,clock=row["dataset"],row["arm"],float(row["time"])
        with np.load(raw/"banks"/ds/f"panel_{clock:.1f}.npz") as bank:
            x=np.column_stack((np.ones(512),bank["x"].astype(np.float64)))
            teacher=bank["oracle"]
        with np.load(raw/"runs"/ds/arm/"diagnostics"/f"panel_{clock:.1f}.npz") as pred:
            error=pred["prediction"].astype(np.float64)-teacher
        coefficient=qr_fit(x,error);affine=x@coefficient
        oof=np.empty_like(error)
        fold_coefficients=[]
        for fit,test in ((np.arange(256),np.arange(256,512)),(np.arange(256,512),np.arange(256))):
            c=qr_fit(x[fit],error[fit]);oof[test]=x[test]@c;fold_coefficients.append(c)
        residual=error-affine
        values=dict(total=mean_square(error),affine=mean_square(affine),nonlinear=mean_square(residual),
                    crossfit_corrected=mean_square(error-oof),crossfit_reduction=mean_square(error)-mean_square(error-oof))
        for key,value in values.items():
            assert np.isclose(value,float(row[key]),rtol=1e-9,atol=1e-12)
            max_difference=max(max_difference,abs(value-float(row[key])))
        with np.load(output/"arrays"/f"{ds}-{arm}-{clock:.1f}.npz") as saved:
            np.testing.assert_array_equal(saved["error"],error)
            for name,value in (("affine",affine),("nonlinear",residual),("out_of_fold_affine",oof),
                               ("coefficient",coefficient),("fold_coefficients",np.stack(fold_coefficients))):
                np.testing.assert_allclose(value,saved[name],rtol=1e-9,atol=2e-11)
        reconstructed.append(dict(dataset=ds,arm=arm,time=clock,**values))
    amap={}
    for row in aggregates:
        subset=[r for r in reconstructed if r["dataset"]==row["dataset"] and r["arm"]==row["arm"]]
        assert len(subset)==5
        average={k:float(np.mean([r[k] for r in subset])) for k in ("total","affine","nonlinear","crossfit_corrected","crossfit_reduction")}
        for key,value in average.items():assert np.isclose(value,float(row[key]),rtol=1e-9,atol=1e-12)
        amap[(row["dataset"],row["arm"])]=average
    primary=[]
    for row in gaps:
        parent=amap[(row["dataset"],row["parent"])];ref=amap[(row["dataset"],row["reference"])]
        differences={k:parent[k]-ref[k] for k in ("total","affine","nonlinear","crossfit_reduction")}
        for key,value in differences.items():assert np.isclose(value,float(row[key]),rtol=1e-9,atol=1e-12)
        shares=(differences["affine"]/differences["total"],differences["crossfit_reduction"]/differences["total"]) if differences["total"]>0 else (None,None)
        for key,value in zip(("affine_excess_share","crossfit_excess_share"),shares):
            if value is None:assert row[key]==""
            else:assert np.isclose(value,float(row[key]),rtol=1e-9,atol=1e-12)
        passed=all(v is not None and v>=0.5 for v in shares)
        assert passed==(row["majority_affine"]=="True")
        if row["dataset"]=="rotated_product_bimodal_16d" and row["parent"]=="ode_k2_direct":primary.append(passed)
    decision=json.loads((output/"decision.json").read_text())
    assert len(primary)==3 and all(primary)==decision["mostly_affine_explanation_survives"]
    for name,value in original_hashes.items():assert digest(runtime/name)==value
    check=dict(status="PASS",panels=len(panels),aggregates=len(aggregates),gaps=len(gaps),
               saved_decompositions=len(panels),maximum_metric_difference=max_difference,
               source_sha256=digest(Path(__file__)),original_output_sha256=original_hashes,
               decision_reconstructed=decision["mostly_affine_explanation_survives"])
    (runtime/"independent-verification.json").write_text(json.dumps(check,indent=2)+"\n")

    figures=report/"figures";figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    datasets=("gaussian_aniso_2d","gmm_ring8_2d","rotated_product_bimodal_16d")
    names=("Gaussian2D","Ring8","Product16")
    color_affine,color_nonlin="#0072B2","#E69F00"
    fig,axes=plt.subplots(2,3,figsize=(11.5,6.9))
    for col,(ds,name) in enumerate(zip(datasets,names)):
        limit=max(v["total"] for (dataset,_),v in amap.items() if dataset==ds)*1.18
        for row,mode in enumerate(("direct","gradient")):
            arms=("ode_k2_"+mode,"swiglu128_"+mode,"swiglu192_"+mode,"swiglu384_direct")
            values=[amap[(ds,arm)] for arm in arms]
            affine=[v["affine"] for v in values];nonlin=[v["nonlinear"] for v in values]
            ax=axes[row,col];x=np.arange(4)
            bars1=ax.bar(x,affine,color=color_affine,label="Raw-affine error")
            bars2=ax.bar(x,nonlin,bottom=affine,color=color_nonlin,label="Orthogonal remainder")
            if mode=="gradient":
                bars1[-1].set_hatch("///");bars2[-1].set_hatch("///")
            ax.set_xticks(x,("ODE","S128","S192","S384"))
            ax.set_ylim(0,limit)
            ax.set_title(name+(" / direct" if mode=="direct" else " / potential gradient"))
            ax.ticklabel_format(axis="y",style="sci",scilimits=(-2,2))
            if col==0:ax.set_ylabel("Mean field MSE across five time panels")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=2,loc="upper center",bbox_to_anchor=(0.5,0.96),frameon=False)
    fig.suptitle("The direct ODE's Product16 gap lies mainly outside raw-affine error",y=1.0,fontsize=14)
    fig.text(0.5,0.01,"All 21 fixed predictors. Hatched S384 is the direct-readout cost control in both rows. No fitted correction is deployed.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,0.04,1,0.91))
    for ext in ("png","pdf"):fig.savefig(figures/("all-model-error-components."+ext),dpi=180,bbox_inches="tight")
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10.5,4.5))
    p=[r for r in gaps if r["dataset"]==datasets[2] and r["parent"]=="ode_k2_direct"]
    x=np.arange(3);affine=np.array([float(r["affine"]) for r in p]);nonlin=np.array([float(r["nonlinear"]) for r in p])
    axes[0].bar(x,affine,color=color_affine,label="Affine excess")
    axes[0].bar(x,nonlin,bottom=affine,color=color_nonlin,label="Nonlinear excess")
    for i,r in enumerate(p):axes[0].text(i,float(r["total"])+0.004,f'{100*float(r["affine_excess_share"]):.2f}% affine',ha="center",fontsize=9)
    axes[0].set_xticks(x,("vs S128","vs S192","vs S384"));axes[0].set_ylim(0,0.36)
    axes[0].set_ylabel("ODE minus reference five-panel MSE")
    axes[0].set_title("Only about 4% of excess error is affine")
    axes[0].legend(frameon=False,fontsize=9,loc="upper left")
    arms=("ode_k2_direct","swiglu128_direct","swiglu192_direct","swiglu384_direct")
    reductions=[amap[(datasets[2],arm)]["crossfit_reduction"] for arm in arms]
    bars=axes[1].bar(np.arange(4),reductions,color=("#D55E00",color_affine,color_affine,color_affine))
    axes[1].axhline(0,color="black",lw=0.8);axes[1].bar_label(bars,fmt="%.4f",padding=4,fontsize=9)
    axes[1].set_xticks(np.arange(4),("ODE","S128","S192","S384"))
    axes[1].set_ylim(-0.017,0.012);axes[1].set_ylabel("Native MSE minus cross-fit-corrected MSE")
    axes[1].set_title("Cross-fit affine subtraction does not help ODE")
    fig.text(0.5,0.01,"Product16, direct readouts, five fixed times. Cross-fit subtraction uses privileged diagnostic error labels; it is not a trained model.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,0.065,1,1))
    for ext in ("png","pdf"):fig.savefig(figures/("Product16-affine-falsifier."+ext),dpi=180,bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({k:v for k,v in check.items() if k!="original_output_sha256"},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw",type=Path,required=True);parser.add_argument("--runtime",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    args=parser.parse_args();main(args.raw.resolve(),args.runtime.resolve(),args.report.resolve())
