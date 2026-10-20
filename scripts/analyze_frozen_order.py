"""Rebuild descriptive frozen-order figures and tables from curated evidence."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/research-2026-10-20-followup"
REPORT = ROOT / "reports/report1"
DATASETS = ["gaussian_aniso_2d", "gmm_ring8_2d", "two_moons_2d"]
NAMES = ["Gaussian", "Ring8", "Moons"]
ORDERS = [1, 2, 4, 8]
LABELS = ["gate_k1_input", "gate_k2_input", "order_k4_input", "order_k8_input"]
COLORS = ["#4477AA", "#EE7733", "#228833", "#AA3377"]


def read(path):
    return json.loads(path.read_text())


def main():
    summary = read(DOC / "frozen-order-summary.json")
    audit = read(DOC / "collection-audit.json")
    assert summary["complete"] and audit["sample_hashes_verified"] == 66
    assert summary["manifest_sha256"] == hashlib.sha256((DOC/"manifest-145.json").read_bytes()).hexdigest()
    rows = summary["rows"]
    idx = {(r["dataset"], r["label"], str(r["eval_order"])): r for r in rows}
    assert len(idx) == 66 and all(r["status"] == "complete" and r["seed"] == 0 for r in rows)
    probes = read(DOC / "probes-145.json")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "savefig.facecolor": "white",
    })
    figs = DOC / "figures"
    figs.mkdir(exist_ok=True)

    def save(fig, name):
        for folder in [figs, REPORT / "figures"]:
            fig.savefig(folder / f"{name}.pdf", bbox_inches="tight", pad_inches=.04)
        plt.close(fig)

    effects = []
    for ds in DATASETS:
        for label, k in zip(LABELS, ORDERS):
            native, exact = (idx[ds, label, mode] for mode in [str(k), "exact"])
            local = probes[f"{ds}/{label}.json"]
            effects.append(dict(
                dataset=ds, trained_order=k, native_sw2=native["sw2"], exact_sw2=exact["sw2"],
                delta_sw2=exact["sw2"]-native["sw2"],
                percent_change=100*(exact["sw2"]/native["sw2"]-1),
                native_oracle_mse=native["oracle_mse"], exact_oracle_mse=exact["oracle_mse"],
                endpoint_rms=exact["endpoint_rms_from_native"],
                velocity_rms=exact["velocity_rms_from_native"],
                local_native_action_error=mean(
                    p["mean_relative_action_error"] for p in local if p["eval_order"] == k),
                local_k8_action_error=mean(
                    p["mean_relative_action_error"] for p in local if p["eval_order"] == 8),
            ))
    (DOC / "exact-effects.json").write_text(json.dumps(effects, indent=2)+"\n")
    table = [
        "| Target | Trained K | Native SW2 | Exact SW2 | Change (%) | Endpoint RMS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in effects:
        table.append(f"| {NAMES[DATASETS.index(r['dataset'])]} | {r['trained_order']} | "
                     f"{r['native_sw2']:.8f} | {r['exact_sw2']:.8f} | "
                     f"{r['percent_change']:+.4f} | {r['endpoint_rms']:.8f} |")
    (DOC / "exact-effects-table.md").write_text("\n".join(table)+"\n")

    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.05), layout="constrained")
    for ax, ds, title in zip(axes, DATASETS, NAMES):
        values = np.array([[idx[ds, label, str(k)]["sw2"] for k in [1, 2, 4, 8, "exact"]]
                           for label in LABELS])
        # Identical normalization across datasets; numbers preserve exact comparisons.
        ax.imshow(values, norm=LogNorm(vmin=.025, vmax=.75), cmap="YlOrRd", aspect="auto")
        for i in range(4):
            for j in range(5):
                ax.text(j, i, f"{values[i,j]:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if values[i,j] > .25 else "black")
            ax.add_patch(Rectangle((i-.48, i-.48), .96, .96, fill=False, lw=1.3, edgecolor="#0072B2"))
        ax.set(xticks=range(5), xticklabels=["1", "2", "4", "8", "exp"],
               yticks=range(4), yticklabels=ORDERS, title=title, xlabel="Evaluation order")
    axes[0].set_ylabel("Trained order")
    save(fig, "09-frozen-order")

    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.2), layout="constrained")
    for ax, ds, name in zip(axes, DATASETS, NAMES):
        for label, k, color, marker in zip(LABELS, ORDERS, COLORS, ["o", "s", "^", "x"]):
            local = probes[f"{ds}/{label}.json"]
            values = [mean(p["mean_relative_action_error"] for p in local
                           if p["eval_order"] == n) for n in ORDERS]
            ax.semilogy(ORDERS, values, "-"+marker, ms=3, lw=1, color=color, label=f"Trained K{k}")
        ax.set(xscale="log", xticks=ORDERS, xticklabels=ORDERS, title=name,
               xlabel="Evaluation order", ylim=(1e-9, 1))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Mean relative action error")
    axes[0].legend(fontsize=6)
    save(fig, "10-fixed-input-error")

    examples = list(csv.DictReader((REPORT / "data/ring8-frozen-examples.csv").open()))
    panels = list(dict.fromkeys(r["panel"] for r in examples))
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.5), layout="constrained")
    for ax, panel in zip(axes.flat, panels):
        xy = np.array([[float(r["x"]), float(r["y"])] for r in examples if r["panel"] == panel])
        assert xy.shape == (1000, 2)
        ax.scatter(*xy.T, s=2, alpha=.4, color=COLORS[0], rasterized=True)
        ax.set(title=panel, xlim=(-4.5, 4.5), ylim=(-4.5, 4.5), aspect="equal",
               xticks=[-3, 0, 3], yticks=[-3, 0, 3], xlabel="x", ylabel="y")
    save(fig, "11-ring8-examples")

    operators = read(DOC / "controlled-operators.json")
    matched = [r for r in operators["records"] if r["family"] == "matched_eigenvalues_and_norm"]
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.25), layout="constrained")
    for k, color, marker in zip([1, 2, 4, 8, "exact"], COLORS+["black"], ["o", "s", "^", "x", "+"]):
        values = [next(e["operator_norm"] for e in r["evaluations"]
                       if e["step"] == 1 and e["order"] == k) for r in matched]
        axes[0].plot([r["parameter"] for r in matched], values, marker=marker, ms=4,
                     color=color, label=f"K{k}" if k != "exact" else "exp", lw=1)
    axes[0].set(title="Same eigenvalues and norm", xlabel=r"Chain shape $\theta$",
                ylabel="Gain at s = 1", xticks=[0, np.pi/8, np.pi/4],
                xticklabels=["0", r"$\pi/8$", r"$\pi/4$"])
    axes[0].legend(ncol=3, fontsize=6)
    values = read(REPORT / "data/report_values.json")
    # Existing report exports descriptive six-layer averages for all checkpoints.
    trajectories = values["trajectories"]
    for ds, name, color in zip(DATASETS, NAMES, COLORS):
        for ax, key, ylabel in zip(axes[1:], ["sigma_max", "stable_rank"],
                                   [r"Mean $\sigma_{\max}(P)$", "Mean stable rank"]):
            r = next(r for r in trajectories if r["dataset"] == ds and r["order"] == 2 and r["metric"] == key)
            ax.plot(r["steps"], r["values"], "-o", ms=2, color=color, label=name)
            ax.set(xlabel="Training updates", ylabel=ylabel, xticks=[0, 2500, 5000],
                   xticklabels=["0", "2,500", "5,000"])
    axes[1].set_title("Learned operator scale")
    axes[2].set_title("Learned concentration")
    axes[2].legend(fontsize=6)
    for ax in axes:
        ax.grid(alpha=.15)
    save(fig, "12-geometry-and-training")

    solver_path = DOC / "solver-check-summary.json"
    if solver_path.exists():
        solver = read(solver_path)
        assert solver["complete"] and len(solver["rows"]) == 48
        si = {(r["label"], str(r["eval_order"]), r["heun_steps"]): r for r in solver["rows"]}
        fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.3), layout="constrained")
        steps = [32, 64, 128, 256]
        for label, k, color in zip(LABELS[:2], ORDERS[:2], COLORS[:2]):
            for mode, style in [(str(k), "-o"), ("exact", "--s")]:
                axes[0].plot(steps, [si[label, mode, n]["sw2"] for n in steps], style,
                             color=color, ms=3, label=f"Train K{k}, {'exp' if mode=='exact' else 'native'}")
        for label, k, color in zip(LABELS[2:], ORDERS[2:], COLORS[2:]):
            for mode, style in [(str(k), "-o"), (str(12-k), "--s"), ("exact", ":x")]:
                axes[1].plot(steps, [si[label, mode, n]["sw2"] for n in steps], style,
                             color=color, ms=3, label=f"K{k} → {mode}")
        for label, title, color, marker in [
            ("silu_5000", "SiLU", COLORS[0], "o"), ("swiglu_5000", "SwiGLU", COLORS[1], "s")]:
            axes[2].plot(steps, [si[label, "native", n]["sw2"] for n in steps],
                         "-"+marker, color=color, ms=3, label=title)
        for ax, title in zip(axes, ["Low orders: failure persists", "K4/K8: residual differences", "Unchanged controls"]):
            ax.set(xscale="log", xticks=steps, xticklabels=steps, xlabel="Heun steps (NFE = 2×steps)",
                   ylabel="SW2", title=title)
            ax.xaxis.set_minor_locator(NullLocator())
            ax.grid(alpha=.15)
            ax.legend(fontsize=6.3)
        save(fig, "13-solver-check")
    print(json.dumps({"cells": len(rows), "exact_effects": effects}, indent=2))


if __name__ == "__main__":
    main()
