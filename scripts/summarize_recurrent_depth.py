"""Descriptive figures and paired evaluation summaries for the audited screen."""
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


def read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def summarize(root: Path) -> None:
    audit = json.loads((root/"audit.json").read_text())
    if audit["status"] != "PASS" or audit["cells"] != 42:
        raise ValueError("complete audited screen required")
    output = root/"figures"
    output.mkdir(exist_ok=False)
    rows = read_csv(root/"comparison.csv")
    curves = read_csv(root/"training-curves.csv")
    depths = read_csv(root/"depth-curves.csv")
    spectra = read_csv(root/"p-trajectories.csv")
    datasets = ("gaussian_aniso_2d", "gmm_ring8_2d", "rotated_product_bimodal_16d")
    names = ("Gaussian 2D", "Ring8 2D", "Product16")
    arms = sorted({r["arm"] for r in rows})
    index = {(r["dataset"], r["arm"]): r for r in rows}
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 160})

    def save(fig, name: str) -> None:
        fig.savefig(output/(name+".png"), bbox_inches="tight")
        fig.savefig(output/(name+".pdf"), bbox_inches="tight")
        plt.close(fig)

    ratios = np.zeros((len(arms), 3))
    for j, dataset in enumerate(datasets):
        best = min(float(r["primary_mse"]) for r in rows
                   if r["dataset"] == dataset and r["family"].startswith("swiglu"))
        ratios[:, j] = [float(index[dataset, arm]["primary_mse"])/best for arm in arms]
    fig, ax = plt.subplots(figsize=(8, 6))
    graphic = ax.imshow(np.log2(ratios), cmap="YlOrRd", aspect="auto",
                        vmin=0, vmax=np.log2(ratios.max()))
    ax.set_xticks(range(3), ("Gaussian 2D", "Ring8 2D", "Product16"))
    ax.set_yticks(range(len(arms)), [s.replace("_", " ") for s in arms])
    for i in range(len(arms)):
        for j in range(3):
            ax.text(j, i, f"{ratios[i,j]:.2f}x", ha="center", va="center",
                    color="white" if ratios[i,j]>4 else "black")
    ax.set_title("Field MSE / best generic control on the same task (lower is better)")
    fig.colorbar(graphic, ax=ax, label="log2 error ratio", shrink=.7)
    save(fig, "field-comparison")

    selected = ("ode_k2_native_final", "ode_k2_shared_final", "ode_k2_shared_deep",
                "swiglu192_native_final", "swiglu192_shared_final", "swiglu192_shared_deep")
    colors = ("#D55E00", "#E69F00", "#AA4499", "#0072B2", "#56B4E9", "#009E73")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, dataset, title in zip(axes, datasets, names):
        for arm, color in zip(selected, colors):
            data = [r for r in curves if r["dataset"]==dataset and r["arm"]==arm and r["oracle_mse"]]
            ax.plot([int(r["step"]) for r in data], [float(r["oracle_mse"]) for r in data],
                    marker="o", markersize=3, color=color, label=arm)
        ax.set_yscale("log")
        ax.set(title=title, xlabel="Training update", ylabel="Validation field MSE")
    axes[-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    save(fig, "learning-curves")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, dataset, title in zip(axes, datasets, names):
        for arm, color in zip(selected[1:3]+selected[4:6], (colors[1], colors[2], colors[4], colors[5])):
            for convention, style in (("extended_horizon", "-"), ("fixed_horizon", "--")):
                data = [r for r in depths if r["dataset"]==dataset and r["arm"]==arm
                        and r["convention"]==convention]
                if convention == "fixed_horizon":
                    data = [r for r in depths if r["dataset"]==dataset and r["arm"]==arm
                            and r["updates"]=="3"]+data
                ax.plot([int(r["updates"]) for r in data], [float(r["oracle_mse"]) for r in data],
                        style, marker="o", markersize=3, color=color,
                        label=arm if convention=="extended_horizon" else None)
        ax.set_yscale("log")
        ax.set(title=title, xlabel="Executed updates", ylabel="Frozen field MSE", xticks=(1, 3, 6, 12))
    axes[-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    fig.suptitle("Solid: longer horizon; dashed: same horizon with smaller steps", fontsize=10)
    fig.tight_layout()
    save(fig, "depth-behavior")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, dataset, title in zip(axes, datasets, names):
        for arm, color in zip(selected[:3], colors[:3]):
            values = [r for r in spectra if r["dataset"]==dataset and r["arm"]==arm]
            steps = sorted({int(r["step"]) for r in values})
            ax.plot(steps, [max(float(r["sigma_max"]) for r in values if int(r["step"])==s)
                           for s in steps], marker="o", color=color, label=arm)
        ax.set(title=title, xlabel="Training update", ylabel="Max singular value across unique P")
    axes[-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    save(fig, "p-spectra")

    # The unit here is a saved paired evaluation input; weights remain fixed.
    contrasts = (("ode_k2_shared_deep", "ode_k2_shared_final"),
                 ("swiglu192_shared_deep", "swiglu192_shared_final"),
                 ("ode_k2_shared_deep", "swiglu192_shared_deep"))
    statistics = []
    rng = np.random.default_rng(0)
    for dataset in datasets:
        group = Path(index[dataset, arms[0]]["group"])
        with np.load(group/"banks"/dataset/"field.npz") as data:
            truth = data["oracle"].astype(np.float64)
        errors = {}
        for arm in {a for pair in contrasts for a in pair}:
            with np.load(group/"runs"/dataset/arm/"diagnostics/field.npz") as data:
                errors[arm] = np.mean((data["prediction"].astype(np.float64)-truth)**2, axis=1)
        differences = np.stack([errors[a]-errors[b] for a, b in contrasts], axis=1)
        bootstrap = []
        for _ in range(20):
            selected_indices = rng.integers(len(truth), size=(50, len(truth)))
            bootstrap.append(differences[selected_indices].mean(axis=1))
        intervals = np.quantile(np.concatenate(bootstrap), (.025, .975), axis=0)
        for j, (a, b) in enumerate(contrasts):
            statistics.append(dict(dataset=dataset, candidate=a, comparator=b, evaluation_inputs=len(truth),
                candidate_mse=float(errors[a].mean()), comparator_mse=float(errors[b].mean()),
                mse_ratio=float(errors[a].mean()/errors[b].mean()),
                paired_mse_difference=float(differences[:, j].mean()),
                conditional_ci95_low=float(intervals[0,j]), conditional_ci95_high=float(intervals[1,j])))
    with (root/"paired-evaluation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(statistics[0]))
        writer.writeheader()
        writer.writerows(statistics)
    (root/"stats-appendix.md").write_text(
        "# Statistical scope\n\nTraining seed is 0 throughout the registered design. The architecture comparison is descriptive. "
        "Checkpoints, internal visits, coordinates and time panels are not independent training repetitions. "
        "The nine prespecified-for-analysis paired contrasts in paired-evaluation.csv use per-input squared errors averaged over coordinates. "
        "Intervals are pointwise 95% percentile intervals from 1000 paired resamples of evaluation rows, bootstrap seed 0, conditional on the fixed trained weights. "
        "They describe evaluation-sample variation, not training-run variation or familywise significance. "
        "No p-values, significance decisions, run-level standard deviations or multiplicity-adjusted discovery claims are made. "
        "Nomination uses the original engineering thresholds and the full generic frontier, not these intervals. "
        "SW2 uses 2048 samples and 128 fixed directions; slice directions are not independent model replications.\n")
    (root/"figure-catalog.md").write_text(
        "# Figures\n\nAll figures are exported as PDF and PNG and have no error bars.\n\n"
        "- field-comparison: all 42 endpoints, normalized against the best generic field MSE for each task. Notice that the ODE gap is largest on Product16. This rejects a practical quality claim for the screened setting. Values are exact descriptive endpoint ratios.\n"
        "- learning-curves: fixed checkpoint errors for the native and recurrent primary families. Notice whether supervision changes convergence rather than selecting a favorable checkpoint. Missing between-checkpoint behavior is not interpolated evidence.\n"
        "- depth-behavior: frozen shared models with extended and fixed internal horizons. Notice that extra internal work does not reliably improve the learned predictor. This separates numerical refinement from longer computation; neither curve is a retrained architecture.\n"
        "- p-spectra: maximum singular value across unique P matrices at each checkpoint. Notice the actual bounded numerical trajectories. This is not the Jacobian spectrum of the conditioned whole network, and cannot by itself explain the quality gap.\n")
    files = [p for p in root.rglob("*") if p.is_file() and p.name != "summary-provenance.json"]
    provenance = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "files": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (root/"summary-provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(json.dumps({"figures": 4, "paired_contrasts": len(statistics)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    summarize(parser.parse_args().results.resolve())
