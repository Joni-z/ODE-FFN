"""Descriptive analysis only: one seed, fixed endpoints, no significance tests."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = {"gaussian_aniso_2d": "Gaussian", "gmm_ring8_2d": "Ring8", "two_moons_2d": "Moons"}
ORDERS = ["gate_k1_input", "gate_k2_input", "order_k4_input", "order_k8_input"]
COLORS = ["#0072B2", "#D55E00", "#CC79A7", "#009E73"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    args = parser.parse_args()
    p = args.report_root
    source = p / "stage-summary.json"
    data = json.loads(source.read_text())
    manifest = json.loads((p / "experiment-manifest.json").read_text())
    audit = json.loads((p / "completion-audit.json").read_text())
    assert data["complete"] and audit["all_hashes_valid"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == audit["summary_sha256"]
    index = {(d["dataset"], d["label"], d["step"]): d for d in data["diagnostics"]}
    curves = {(d["dataset"], d["label"]): d["records"] for d in data["curves"]}
    assert len(index) == 54 and len(curves) == 30
    out = p / "analysis-output"
    out.mkdir(exist_ok=True)
    figs = p / "figures"
    figs.mkdir(exist_ok=True)
    def sw(ds, label, step=5000):
        return index[(ds, label, step)]["sampling"]["32"]["sw2"]
    def layer(ds, label, key):
        return mean(l[key]["mean"] for l in index[(ds, label, 5000)]["layers"])
    effects = []
    for ds in NAMES:
        for k in (1, 2):
            a, b = sw(ds, f"gate_k{k}_input"), sw(ds, f"gate_k{k}_time")
            effects.append(dict(dataset=ds, comparison=f"time-minus-input-K{k}",
                reference=a, alternative=b, difference=b-a, percent_change=100*(b/a-1)))
        a, b = sw(ds, "order_k4_input"), sw(ds, "order_k8_input")
        effects.append(dict(dataset=ds, comparison="K8-minus-K4", reference=a, alternative=b,
                            difference=b-a, percent_change=100*(b/a-1)))
        for k, label in zip((1, 2, 4, 8), ORDERS):
            n = manifest["baseline_budgets"]["swiglu"]["matched_steps"][str(k)]
            a, b = sw(ds, label), sw(ds, "swiglu_budget", n)
            effects.append(dict(dataset=ds, comparison=f"MAC-SwiGLU-minus-ODE-K{k}",
                reference=a, alternative=b, difference=b-a, percent_change=100*(b/a-1)))
    with (out/"contrasts.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(effects[0]))
        writer.writeheader(); writer.writerows(effects)
    trajectory = []
    for ds in NAMES:
        for label in ORDERS:
            rows = curves[(ds, label)]
            first, last = rows[0], rows[-1]
            assert first["step"] == 0 and last["step"] == 5000
            def avg(r, category, key):
                return mean(l[category][key] for l in r["ode_layers"])
            trajectory.append(dict(dataset=ds, label=label,
                sigma_initial=avg(first, "effective", "sigma_max"),
                sigma_final=avg(last, "effective", "sigma_max"),
                stable_rank_initial=avg(first, "effective", "stable_rank"),
                stable_rank_final=avg(last, "effective", "stable_rank"),
                gate_mean_final=avg(last, "activation", "s_mean"),
                exp_relative_action=layer(ds,label,"relative_action_truncation_error"),
                full_jacobian=layer(ds,label,"full_layer_jacobian_norm")))
    with (out/"p-trajectory-summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trajectory[0]))
        writer.writeheader();writer.writerows(trajectory)
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False,
                         "pdf.fonttype":42})
    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(figs/(name+"."+ext), dpi=180, bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(2,3,figsize=(13,7),layout="constrained")
    for j, ds in enumerate(NAMES):
        for label,k,col in zip(ORDERS,(1,2,4,8),COLORS):
            rows = curves[(ds,label)]
            x = [r["step"] for r in rows]
            for ax,key in zip(axes[:,j],("sigma_max","stable_rank")):
                y = [mean(l["effective"][key] for l in r["ode_layers"]) for r in rows]
                ax.plot(x,y,label=f"K={k}",color=col)
                ax.set_xlabel("Training updates");ax.grid(alpha=.2)
        axes[0,j].set_title(NAMES[ds]);axes[0,j].set_ylabel("Mean largest singular value of P")
        axes[1,j].set_ylabel("Mean stable rank of P")
    axes[0,0].legend()
    fig.suptitle("Learned P over training | input gates | seed 0 | descriptive mean over six layers")
    save(fig,"p-training-dynamics")
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout="constrained")
    for j,ds in enumerate(NAMES):
        for label,col in [("silu_budget","#555555"),("swiglu_budget","#009E73")]:
            rows=[r for r in curves[(ds,label)] if r.get("sample") and r["step"]>=4000]
            axes[0,j].plot([r["step"] for r in rows],[r["sample"]["sliced_w2"] for r in rows],
                           "-o",markersize=3,label=label.replace("_budget",""),color=col)
        for label,k,col in zip(ORDERS,(1,2,4,8),COLORS):
            axes[0,j].axhline(sw(ds,label),color=col,alpha=.4,ls=":",label=f"ODE K{k}, step5000")
        axes[0,j].set_title(NAMES[ds]);axes[0,j].set_ylabel("Baseline SW2 (Heun32)")
        axes[0,j].set_xlabel("Baseline training updates");axes[0,j].grid(alpha=.2)
        for label,k,col in zip(ORDERS,(1,2,4,8),COLORS):
            d=index[(ds,label,5000)]
            steps=[8,16,32,64,128,256]
            axes[1,j].plot([2*n for n in steps],[d["sampling"][str(n)]["sw2"] for n in steps],
                           "-o",markersize=3,label=f"K={k}",color=col)
        axes[1,j].set_xscale("log",base=2);axes[1,j].set_xlabel("Sampling network evaluations (NFE)")
        axes[1,j].set_ylabel("ODE SW2 at step5000");axes[1,j].grid(alpha=.2)
    axes[0,0].legend(fontsize=7,ncol=2);axes[1,0].legend(fontsize=8)
    fig.suptitle("Endpoint sensitivity and solver check | fixed evaluation samples | seed 0")
    save(fig,"training-and-nfe")
    lines=["# Descriptive statistics appendix","",
           "Primary endpoint: SW2 at the predeclared checkpoint, Heun32, 2048 fixed generated and target points, 128 fixed directions; lower is better.",
           "Unit: one trained trajectory per dataset/configuration; seed0 only. Baseline prefix duplicates are technical reruns. Checkpoints and layers are correlated measurements, not independent repetitions.",
           "No across-seed SD, confidence intervals, p-values or multiple-comparison significance tests are estimable here. Effect sizes below are observed absolute/relative differences, not population estimates. No cross-dataset mean score is used.",
           "Baseline budget matching is a full-network forward-linear-MAC proxy, not measured end-to-end cost. Endpoint fluctuation is reported without replacing the prespecified endpoint with the best checkpoint.",
           "", "|Dataset|Contrast|Reference|Alternative|Difference|Relative change|",
           "|---|---|---:|---:|---:|---:|"]
    for e in effects:
        lines.append(f"|{NAMES[e['dataset']]}|{e['comparison']}|{e['reference']:.6f}|{e['alternative']:.6f}|{e['difference']:+.6f}|{e['percent_change']:+.2f}%|")
    lines += ["", "## P at initial and final checkpoint", "",
              "Layer means are descriptive; stable rank is ||P||_F^2 / sigma_max(P)^2, not algebraic rank.",
              "|Dataset|Model|Initial sigma|max sigma final|Initial stable rank|Final stable rank|Final mean s|Exp action difference|",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in trajectory:
        lines.append(f"|{NAMES[r['dataset']]}|{r['label']}|{r['sigma_initial']:.4f}|{r['sigma_final']:.4f}|{r['stable_rank_initial']:.3f}|{r['stable_rank_final']:.3f}|{r['gate_mean_final']:.3f}|{r['exp_relative_action']:.3e}|")
    lines+=["",f"Source summary SHA256: {audit['summary_sha256']}",
            "All54 checkpoint hashes,29 frozen source/config hashes and training config/source identities audited remotely; all30 runs complete. Six baseline prefix weight comparisons are exactly zero."]
    (out/"stats-appendix.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(contrasts=len(effects), p_trajectories=len(trajectory),
                         figures=["p-training-dynamics","training-and-nfe"])))


if __name__ == "__main__":
    main()
