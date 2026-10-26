"""Independent functional replay and descriptive cross-fit temporal analysis."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.func import functional_call

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from synthetic.recurrent_depth import build
from synthetic.train import restore_model


def load(path: Path):return json.loads(path.read_text())
def sha(path: Path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path: Path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def csv_write(path: Path,rows: list[dict]):
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def analyze(raw: Path,groups: list[Path],output: Path) -> None:
    torch.set_num_threads(1)
    status=load(raw/"status.json")
    assert status["state"]=="COMPLETE" and status["completed"]==24 and status["risk_rows"]==720
    bindings=load(raw/"provenance.json")
    for path,value in bindings["inputs"].items():assert sha(Path(path))==value
    for name in ("synthetic/models.py","synthetic/recurrent_depth.py","synthetic/train.py"):
        assert sha(ROOT/name)==bindings["runtime_source"][name]
    with (raw/"risk.csv").open() as handle:risks=list(csv.DictReader(handle))
    information=load(raw/"gradients.json")
    meta=load(raw/"parameter-metadata.json")
    grouped={d:g for g in groups for d in load(g/"study.json")["datasets"]}
    aggregate,transfer,receipts=[],[],[]
    max_gradient_error=max_prediction_error=0.
    models={}
    for entry in meta:
        dataset,arm=entry["dataset"],entry["arm"]
        group=grouped[dataset];config=load(group/"study.json")
        setting=config["arms"][arm]
        data_root=Path(config["data_roots"][dataset])
        with np.load(data_root/dataset/"validation.npz") as data:target=data["x1"][:512].astype(np.float64)
        state=torch.load(group/"runs"/dataset/arm/"step_0005000.pt",map_location="cpu",weights_only=False)
        model=build(target.shape[1],setting["family"],setting["mode"])
        restore_model(model,state);model=model.double().eval()
        for p in model.parameters():p.requires_grad_(False)
        names=entry["parameter_names"];parameters=dict(model.named_parameters())
        reference={name:parameters[name].detach().clone() for name in names}
        panels=[]
        for time in (.1,.3,.5,.7,.9):
            with np.load(group/"banks"/dataset/f"panel_{time:.1f}.npz") as data:
                x,t,oracle=[data[k].astype(np.float64) for k in ("x","t","oracle")]
            pair=target-(x-t*target)/(1-t)
            panels.append({k:torch.from_numpy(v) for k,v in dict(x=x,t=t,oracle=oracle,pair=pair).items()})
        models[dataset,arm]=(model,names,reference,panels)
    for info in information:
        dataset,arm,teacher,fold=[info[k] for k in ("dataset","arm","teacher","fold")]
        tag=f"{dataset}-{arm}-{teacher}-{fold}"
        saved=torch.load(raw/(tag+".pt"),map_location="cpu",weights_only=False)
        opposite=torch.load(raw/f"{dataset}-{arm}-oracle-{1-fold}.pt",map_location="cpu",weights_only=False)
        gradients=saved["gradient"].numpy();phi=saved["phi"].numpy()
        np.testing.assert_allclose(phi.mean(),0.,atol=1e-12)
        np.testing.assert_allclose(np.mean(phi**2),1.,atol=1e-12)
        g0=gradients.mean(0);g1=np.mean(phi[:,None]*gradients,axis=0)
        norm0=np.linalg.norm(g0);joint=np.sqrt(np.sum(g0*g0)+np.sum(g1*g1))
        expected={"shared":np.repeat((-.01*g0/norm0)[None,:],5,axis=0),
            "time_linear":-.01*(g0+phi[:,None]*g1)/joint,
            "time_reverse":-.01*(g0-phi[:,None]*g1)/joint}
        for key,value in expected.items():
            np.testing.assert_allclose(value,saved["displacements"][key].numpy(),rtol=1e-11,atol=1e-14)
            np.testing.assert_allclose(np.mean(np.sum(value**2,axis=1)),.0001,rtol=1e-12,atol=1e-16)
        model,names,reference,panels=models[dataset,arm]
        with np.load(raw/(tag+"-predictions.npz")) as data:
            predictions=data["prediction"].reshape(5,3,512,-1)
            native=data["native"]
            truth=data["oracle"]
        calibration=slice(256*fold,256*(fold+1))
        for index,panel in enumerate(panels):
            # Independent differentiation through a scaled-coordinate functional model.
            b=torch.zeros(info["dimension"],dtype=torch.float64,requires_grad=True)
            updates={};offset=0
            for name,norm in zip(names,saved["norms"]):
                weight=reference[name];count=weight.numel()
                updates[name]=weight+norm*b[offset:offset+count].reshape_as(weight);offset+=count
            prediction=functional_call(model,updates,(panel["x"][calibration],panel["t"][calibration]))
            loss=(prediction-panel[teacher][calibration]).square().mean()
            actual=torch.autograd.grad(loss,b)[0]
            error=float((actual-saved["gradient"][index]).abs().max())
            max_gradient_error=max(max_gradient_error,error)
            torch.testing.assert_close(actual,saved["gradient"][index],rtol=2e-10,atol=2e-12)
            for vi,(variant,direction) in enumerate(saved["displacements"].items()):
                offset=0;updates={}
                for name,norm in zip(names,saved["norms"]):
                    weight=reference[name];count=weight.numel()
                    updates[name]=weight+norm*direction[index,offset:offset+count].reshape_as(weight);offset+=count
                with torch.no_grad():
                    replay=functional_call(model,updates,(panel["x"],panel["t"])).numpy()
                error=float(np.max(np.abs(replay-predictions[index,vi])))
                max_prediction_error=max(max_prediction_error,error)
                np.testing.assert_array_equal(replay,predictions[index,vi])
        for vi,variant in enumerate(("shared","time_linear","time_reverse")):
            selected=[r for r in risks if r["dataset"]==dataset and r["arm"]==arm and r["teacher"]==teacher
                      and int(r["fold"])==fold and r["variant"]==variant]
            for split,selection in (("calibration",calibration),("evaluation",slice(256*(1-fold),256*(2-fold)))):
                for index in range(5):
                    row=sorted((r for r in selected if r["split"]==split),key=lambda r:float(r["time"]))[index]
                    before=np.mean((native[index,selection]-truth[index,selection])**2)
                    after=np.mean((predictions[index,vi,selection]-truth[index,selection])**2)
                    np.testing.assert_allclose([before,after],[float(row["native_oracle_mse"]),float(row["changed_oracle_mse"])],rtol=1e-12,atol=1e-14)
            rows=[r for r in selected if r["split"]=="evaluation"]
            before=np.mean([float(r["native_oracle_mse"]) for r in rows]);after=np.mean([float(r["changed_oracle_mse"]) for r in rows])
            fit=[r for r in selected if r["split"]=="calibration"]
            actual_fit_change=np.mean([float(r["fit_loss"])-float(r["native_fit_loss"]) for r in fit])
            held_slope=float(np.mean(np.sum(opposite["gradient"].numpy()*saved["displacements"][variant].numpy(),axis=1)))
            aggregate.append(dict(dataset=dataset,arm=arm,teacher=teacher,fold=fold,variant=variant,
                native_evaluation_mse=before,changed_evaluation_mse=after,finite_evaluation_change=after-before,
                calibration_first_order=info["first_order"][variant],actual_calibration_change=actual_fit_change,
                evaluation_first_order=held_slope,
                evaluation_alignment=float(np.mean([float(r["oracle_alignment"]) for r in rows])),
                evaluation_change_energy=float(np.mean([float(r["finite_change_energy"]) for r in rows]))))
        receipts.append(dict(dataset=dataset,arm=arm,teacher=teacher,fold=fold,gradient="PASS",finite_functional_replay="bitwise PASS"))
    for dataset,arm in models:
        for teacher in ("oracle","pair"):
            selected=[r for r in aggregate if r["dataset"]==dataset and r["arm"]==arm and r["teacher"]==teacher]
            shared=[r["evaluation_first_order"] for r in selected if r["variant"]=="shared"]
            linear=[r["evaluation_first_order"] for r in selected if r["variant"]=="time_linear"]
            reverse=[r["evaluation_first_order"] for r in selected if r["variant"]=="time_reverse"]
            transfer.append(dict(dataset=dataset,arm=arm,teacher=teacher,
                shared_slope=float(np.mean(shared)),time_linear_slope=float(np.mean(linear)),
                reversed_slope=float(np.mean(reverse)),slope_gain_ratio=float(np.mean(linear)/np.mean(shared)),
                both_folds_linear_better=all(a<b and a<c for a,b,c in zip(linear,shared,reverse))))
    output.mkdir(parents=True,exist_ok=False)
    csv_write(output/"aggregate.csv",aggregate);csv_write(output/"infinitesimal-transfer.csv",transfer)
    write(output/"audit.json",dict(status="PASS",calibrations=24,risks=720,
        independently_recomputed_full_gradients=120,functional_prediction_arrays=360,
        max_gradient_absolute_error=max_gradient_error,max_prediction_absolute_error=max_prediction_error,
        receipts=receipts,input_provenance_sha256=sha(raw/"provenance.json"),script_sha256=sha(Path(__file__))))
    fig,axes=plt.subplots(2,3,figsize=(11,6))
    titles={"ode_k2_native_final":"ODE K2","swiglu192_native_final":"SwiGLU192","swiglu128_native_deep":"SwiGLU128 deep"}
    datasets=("gmm_ring8_2d","rotated_product_bimodal_16d")
    for di,dataset in enumerate(datasets):
        for ai,arm in enumerate(titles):
            data=[v for v in information if v["dataset"]==dataset and v["arm"]==arm and v["teacher"]=="oracle"]
            cosine=np.mean([v["cosine"] for v in data],axis=0)
            ax=axes[di,ai];im=ax.imshow(cosine,vmin=-1,vmax=1,cmap="coolwarm")
            ax.set_xticks(range(5),(".1",".3",".5",".7",".9"));ax.set_yticks(range(5),(".1",".3",".5",".7",".9"))
            ax.set_title(titles[arm]+(" / Ring8" if di==0 else " / Product16"),fontsize=10)
    fig.colorbar(im,ax=axes.ravel().tolist(),label="Gradient cosine; mean of the two calibration halves",shrink=.8)
    fig.savefig(output/"gradient-cosines.png",dpi=160,bbox_inches="tight");fig.savefig(output/"gradient-cosines.pdf",bbox_inches="tight");plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    labels=[("Ring8 " if r["dataset"]==datasets[0] else "Product16 ")+titles[r["arm"]] for r in transfer if r["teacher"]=="oracle"]
    for ax,teacher in zip(axes,("oracle","pair")):
        data=[r for r in transfer if r["teacher"]==teacher]
        for variant,key,offset,color in (("Shared","shared_slope",-.2,"#0072B2"),("Time linear","time_linear_slope",0,"#D55E00"),("Reverse","reversed_slope",.2,"#009E73")):
            ax.barh(np.arange(6)+offset,[-r[key] for r in data],height=.19,label=variant,color=color)
        ax.set_yticks(range(6),labels if teacher=="oracle" else []);ax.invert_yaxis()
        ax.set(title="Calibration: "+teacher,xlabel="Infinitesimal opposite-half risk decrease\n(scaled by the declared matrix budget)")
    axes[1].legend(fontsize=8);fig.tight_layout()
    fig.savefig(output/"transfer-slopes.png",dpi=160,bbox_inches="tight");fig.savefig(output/"transfer-slopes.pdf",bbox_inches="tight");plt.close(fig)
    finite=[r for r in aggregate if r["teacher"]=="oracle"]
    fig,ax=plt.subplots(figsize=(8,4))
    for variant,marker,color in (("shared","o","#0072B2"),("time_linear","s","#D55E00"),("time_reverse","^","#009E73")):
        selected=[r for r in finite if r["variant"]==variant]
        ax.scatter([-r["evaluation_first_order"] for r in selected],[r["finite_evaluation_change"] for r in selected],marker=marker,color=color,label=variant)
    ax.axhline(0,color="black",linewidth=.8);ax.set(xlabel="Predicted infinitesimal decrease",ylabel="Actual finite risk change at rho=.01",title="The fixed finite displacement overshoots across all three families")
    ax.legend();fig.tight_layout();fig.savefig(output/"finite-versus-local.png",dpi=160);fig.savefig(output/"finite-versus-local.pdf");plt.close(fig)
    lines=["# Time-dependent operator diagnostic","","The same-budget temporal tangent direction transfers across the two analysis halves, but the effect also occurs in both ordinary FFN comparators. The registered1% finite displacement overshoots even for the shared-gradient control. This makes the finite experiment inconclusive about whether a properly trained time-dependent operator would help; it is not evidence for rejecting time conditioning.","","| Dataset | Arm | Calibration teacher | Temporal/shared local gain | Better in both halves |","|---|---|---|---:|---|"]
    for r in transfer:lines.append(f"| {r['dataset']} | {r['arm']} | {r['teacher']} | {r['slope_gain_ratio']:.3f} | {r['both_folds_linear_better']} |")
    lines += ["","The displayed gain concerns an infinitesimal direction conditional on the frozen weights, a declared matrix metric and exposed panel inputs. It is not a trained quality improvement. The larger calibration tangent space already guarantees a local fit advantage; opposite-half transfer checks that this is not only calibration fitting. It still does not establish an ODE-specific mechanism, useful extra computation, or that conflicting time gradients cause the observed practical quality gap.","","![Cosines](gradient-cosines.png)","","Negative off-diagonal entries describe time-specific local gradient disagreement. Their presence is not sufficient to nominate a new architecture. Selected parameter dimensions and norms are preserved in parameter-metadata.json at the raw output.","","![Local transfer](transfer-slopes.png)","","Both loss-label definitions and both cross-fit orientations are retained. The generic ordinary FFNs display similar local temporal benefits. The actual training-bank objective was not optimized or reconstructed by this diagnostic.","","![Finite and local](finite-versus-local.png)","","A positive first-order descent prediction with a finite risk increase shows that the chosen radius is outside a reliable local approximation for these directions. No radius sweep, rank search, new loss or architecture is selected. The exact finite output changes and polarization components are in aggregate.csv; the original scope and stopping boundary remain intact.","","Decision: do not train a time-dependent-P variant from this evidence alone. Keep the shared temporal-direction question as generic conditional-computation evidence, not an explanation unique to the ODE gap. The broader research goal remains active."]
    (output/"analysis-report.md").write_text("\n".join(lines)+"\n")
    (output/"stats-appendix.md").write_text("# Analysis scope\n\nSix frozen training checkpoints, five times, two complementary analysis splits and two calibration-label definitions. Training seed0 is retained. The folds and time panels are dependent analysis units, not independent trained-model replications. Descriptive effects only; no p-values, confidence intervals over training variation or significance claims. The finite interventions and infinitesimal directions answer different questions. No fresh test inputs or generation were used.\n")
    (output/"figure-catalog.md").write_text("# Figures\n\n- gradient-cosines: all time-gradient pair similarities for the oracle calibration, averaging the two halves. Shows local disagreement in both ODE and ordinary controls. Does not establish a causal bottleneck.\n- transfer-slopes: local risk decrease on the opposite half for every checkpoint and both label definitions. Shows transferable temporal sensitivity, including generic controls. No error bars; exact fold values are in aggregate.csv.\n- finite-versus-local: local expected decrease against actual finite change, all oracle-calibrated folds and methods. Shows that the declared fixed radius overshoots even in the shared control, blocking a finite improvement or rejection claim.\n")
    print(json.dumps(dict(audit="PASS",gradient_error=max_gradient_error,prediction_error=max_prediction_error,training_selected=False)))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw",type=Path,required=True);parser.add_argument("--inputs",type=Path,nargs="+",required=True)
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    analyze(args.raw.resolve(),[p.resolve() for p in args.inputs],args.output.resolve())
