"""Rebuild Report 1 displays from committed experimental evidence; no training."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAGE = ROOT / "docs/research-2026-10-20"
OLD = ROOT / "docs/research-2026-10-19"
DS = ["gaussian_aniso_2d", "gmm_ring8_2d", "two_moons_2d"]
NAMES = ["Gaussian", "Ring8", "Moons"]
COLORS = ["#4477AA", "#EE7733", "#228833", "#AA3377"]
LABELS = ["gate_k1_input", "gate_k2_input", "order_k4_input", "order_k8_input"]
K = [1, 2, 4, 8]


def main():
    data = json.loads((STAGE / "stage-summary.json").read_text())
    audit = json.loads((STAGE / "completion-audit.json").read_text())
    assert data["complete"] and audit["all_hashes_valid"]
    assert hashlib.sha256((STAGE / "stage-summary.json").read_bytes()).hexdigest() == audit["summary_sha256"]
    idx = {(r["dataset"], r["label"], r["step"]): r for r in data["diagnostics"]}
    curves = {(r["dataset"], r["label"]): r["records"] for r in data["curves"]}
    legacy = {d: json.loads((OLD / "diagnostics" / f"{d}.json").read_text())["runs"] for d in DS}
    assert len(idx) == 54 and len(curves) == 30
    assert all(r["seed"] == 0 for r in idx.values())

    def row(d, label, step=5000):
        return idx[d, label, step]

    def sw(d, label, step=5000):
        return row(d, label, step)["sampling"]["32"]["sw2"]

    def layer(r, key):
        return mean(x[key]["mean"] for x in r["layers"])

    def old(d, kind, order):
        return next(r for r in legacy[d] if r["model"] == kind and r["orders"] == order)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "savefig.facecolor": "white",
    })
    (HERE / "figures").mkdir(exist_ok=True)

    def save(fig, name):
        fig.savefig(HERE / "figures" / f"{name}.pdf", bbox_inches="tight", pad_inches=.04)
        plt.close(fig)

    examples = list(csv.DictReader((HERE / "data/target_examples.csv").open()))
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 3.65), layout="constrained")
    endpoints = []
    for j, d in enumerate(DS):
        points = np.array([[float(r["x"]), float(r["y"])] for r in examples if r["dataset"] == d])
        assert points.shape == (1000, 2)
        axes[0, j].scatter(*points.T, s=2, alpha=.35, color=COLORS[0], rasterized=True)
        extent = np.ceil(np.abs(points).max())
        axes[0, j].set(title=NAMES[j], xlabel="$x$", ylabel="$y$", aspect="equal",
                       xlim=(-extent, extent), ylim=(-extent, extent))
        values = [sw(d, "silu_5000"), sw(d, "swiglu_5000"),
                  sw(d, "gate_k2_input"), sw(d, "swiglu_budget", 6572)]
        ax = axes[1, j]
        bars = ax.bar(range(4), values, color=[COLORS[0], COLORS[1], COLORS[2], COLORS[1]], width=.66)
        bars[-1].set_hatch("///")
        bars[-1].set_edgecolor("#333333")
        bars[-1].set_linewidth(.5)
        ax.set_xticks(range(4), ["SiLU", "SwiGLU", "ODE K2", "SwiGLU\nbudget"], rotation=20)
        ax.set_ylabel("SW2 (lower is better)")
        ax.set_ylim(0, max(values)*1.3)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
        floor = curves[d, "gate_k2_input"][-1]["sample"]["reference_floor"]
        ax.axhline(floor, color=".55", ls=":", lw=1)
        endpoints.append(dict(dataset=d, silu=values[0], swiglu=values[1],
                              ode_k2=values[2], swiglu_k2_budget=values[3],
                              target_subset_reference=floor))
    save(fig, "01-data-and-comparison")

    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.35), layout="constrained")
    cap_records = []
    for ax, d, name in zip(axes, DS, NAMES):
        for k, color, marker in [(1, COLORS[0], "o"), (2, COLORS[1], "s")]:
            a, b = old(d, "ode_original", k), old(d, "ode_spectral_cap", k)
            x = [layer(r, "full_layer_jacobian_norm") for r in (a, b)]
            y = [r["sampling"]["32"]["sw2"] for r in (a, b)]
            ax.plot(x, y, color=color, lw=1)
            ax.scatter(x[0], y[0], color=color, marker=marker, s=35, label=f"K{k}: free P")
            ax.scatter(x[1], y[1], edgecolor=color, facecolor="white", marker=marker, s=35)
            ax.annotate("", xy=(x[1], y[1]), xytext=(x[0], y[0]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=.8))
            cap_records.append(dict(dataset=d, order=k, free_jacobian=x[0], cap_jacobian=x[1],
                                    free_sw2=y[0], cap_sw2=y[1]))
        ax.set(title=name, xlabel="Mean layer Jacobian norm", ylabel="SW2")
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=7)
    save(fig, "02-spectral-cap")

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.35), layout="constrained")
    gates = []
    for k, color, offset in [(1, COLORS[0], -.18), (2, COLORS[1], .18)]:
        scores, jac = [], []
        for d in DS:
            a, b = row(d, f"gate_k{k}_input"), row(d, f"gate_k{k}_time")
            ds = b["sampling"]["32"]["sw2"]-a["sampling"]["32"]["sw2"]
            dj = layer(b, "full_layer_jacobian_norm")-layer(a, "full_layer_jacobian_norm")
            assert layer(b, "gate_correction_norm") == 0
            scores.append(ds); jac.append(dj)
            gates.append(dict(dataset=d, order=k, delta_sw2=ds, delta_full_jacobian=dj))
        for ax, vals in zip(axes, (scores, jac)):
            ax.bar(np.arange(3)+offset, vals, width=.34, color=color, label=f"K={k}")
    for ax in axes:
        ax.set_xticks(range(3), NAMES)
        ax.axhline(0, color=".3", lw=.8)
        ax.grid(axis="y", alpha=.15)
    axes[0].set(title="Generation", ylabel="SW2: time gate minus input gate")
    axes[1].set(title="Internal sensitivity", ylabel="Jacobian norm: time minus input")
    axes[0].legend()
    save(fig, "03-gate-intervention")

    fig, axes = plt.subplots(2, 3, figsize=(6.8, 3.65), layout="constrained")
    orders = []
    for j, d in enumerate(DS):
        scores = [sw(d, label) for label in LABELS]
        errors = [layer(row(d, label), "relative_action_truncation_error") for label in LABELS]
        axes[0, j].plot(K, scores, "-o", color=COLORS[0], ms=4)
        axes[1, j].semilogy(K, errors, "-s", color=COLORS[1], ms=4)
        axes[0, j].set(title=NAMES[j], ylabel="SW2")
        axes[1, j].set(ylabel="Relative exp-action error", xlabel="Taylor order K")
        for ax in axes[:, j]:
            ax.set_xscale("log", base=2); ax.set_xticks(K, K); ax.grid(alpha=.15)
        orders.append(dict(dataset=d, orders=K, sw2=scores, exponential_action_error=errors))
    save(fig, "04-order")

    fig, axes = plt.subplots(2, 3, figsize=(6.8, 3.6), layout="constrained")
    trajectories = []
    for j, d in enumerate(DS):
        for label, k, color, marker in zip(LABELS, K, COLORS, ["o", "s", "^", "x"]):
            records = curves[d, label]
            x = [r["step"] for r in records]
            for ax, key in zip(axes[:, j], ["sigma_max", "stable_rank"]):
                vals = [mean(l["effective"][key] for l in r["ode_layers"]) for r in records]
                ax.plot(x, vals, color=color, label=f"K={k}", marker=marker, ms=3, markevery=2, lw=1)
                trajectories.append(dict(dataset=d, order=k, metric=key, steps=x, values=vals))
                ax.grid(alpha=.15)
        axes[0, j].set(title=NAMES[j], ylabel=r"Mean $\sigma_{\max}(P)$")
        axes[1, j].set(ylabel="Mean stable rank", xlabel="Training updates")
        axes[1, j].set_xticks([0, 2500, 5000], ["0", "2,500", "5,000"])
    axes[0, 0].legend(ncol=2, fontsize=7)
    save(fig, "05-p-trajectories")

    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.6), layout="constrained")
    for kind, order, label, color in [("ffn", None, "SiLU", COLORS[0]),
                                     ("swiglu", None, "SwiGLU", COLORS[1]),
                                     ("ode_original", 2, "ODE K2", COLORS[2])]:
        r = old(DS[1], kind, order)
        bins = r["time_bins"]
        axes[0].plot([(b["t_left"]+b["t_right"])/2 for b in bins],
                     [b["oracle_mse"] for b in bins], "-o", ms=2, label=label, color=color)
        n = [8, 16, 32, 64, 128, 256]
        axes[2].plot([2*x for x in n], [r["sampling"][str(x)]["sw2"] for x in n],
                     "-o", ms=2, color=color, label=label)
    for label, color in [("silu_budget", COLORS[0]), ("swiglu_budget", COLORS[1])]:
        records = [r for r in curves[DS[1], label] if r.get("sample") and r["step"] >= 4000]
        axes[1].plot([r["step"] for r in records], [r["sample"]["sliced_w2"] for r in records],
                     "-o", ms=2, color=color)
    axes[1].axhline(sw(DS[1], "gate_k2_input"), ls=":", color=COLORS[2], label="ODE K2 at 5,000")
    axes[0].set(title="Ring8: fitting over time", xlabel="Flow time t", ylabel="Oracle velocity MSE")
    axes[1].set(title="Ring8: endpoint sensitivity", xlabel="Baseline updates", ylabel="SW2")
    axes[1].set_xticks([5000, 10000, 15000], ["5k", "10k", "15k"])
    axes[2].set(title="Ring8: solver check", xlabel="Network evaluations", ylabel="SW2", xscale="log")
    axes[2].set_xticks([16, 64, 256], ["16", "64", "256"])
    axes[0].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=.15)
    save(fig, "06-ring8-case")

    # Machine-readable values and a LaTeX table share the same source as the figures.
    (HERE / "data/report_values.json").write_text(json.dumps(dict(
        endpoints=endpoints, spectral_cap=cap_records, gate=gates,
        orders=orders, trajectories=trajectories), indent=2)+"\n")
    table = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Model / updates & Gaussian & Ring8 & Moons \\", r"\midrule"]
    for label, name, step in [("silu_5000", "SiLU / 5,000", 5000),
                              ("swiglu_5000", "SwiGLU / 5,000", 5000),
                              ("gate_k2_input", "ODE K2 / 5,000", 5000),
                              ("swiglu_budget", "SwiGLU / 6,572", 6572)]:
        table.append(name + " & " + " & ".join(f"{sw(d,label,step):.4f}" for d in DS) + r" \\")
    table += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "comparison_table.tex").write_text("\n".join(table)+"\n")
    sources = [STAGE / "stage-summary.json", STAGE / "completion-audit.json",
               STAGE / "protocol.md", STAGE / "experiment-manifest.json",
               ROOT / "synthetic/models.py", ROOT / "synthetic/train.py",
               ROOT / "synthetic/data.py", ROOT / "synthetic/geometry.py",
               ROOT / "configs/synthetic/data_v1.json", ROOT / "configs/synthetic/train_v1.json",
               HERE / "data/target_examples.csv",
               *[OLD / "diagnostics" / f"{d}.json" for d in DS]]
    provenance = {
        "report": "Report 1", "evidence_revision": "84ff784",
        "review_status": "Assistant checked against repository evidence; author review pending.",
        "training_seed": 0, "new_training_performed_for_report": False,
        "target_examples": "First 1,000 x1 rows of each actual torch v1 test split; no resampling.",
        "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    }
    (HERE / "evidence_manifest.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print("Generated six figures, comparison table, report values, and evidence hashes.")


if __name__ == "__main__":
    main()
