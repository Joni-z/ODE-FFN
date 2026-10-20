"""Plot audited gate-study evidence without modifying the submitted report."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.collect_gate_study import audit

KINDS = ["ode_original", "ode_fixed", "ode_scalar", "ffn", "swiglu"]
LABELS = ["Input gate", "Fixed 0.6", "Scalar", "SiLU", "SwiGLU"]
COLORS = ["#0072B2", "#D55E00", "#009E73", "#666666", "#CC79A7"]
DATASETS = ["gaussian_aniso_2d", "gmm_ring8_2d", "two_moons_2d"]
NAMES = ["Anisotropic Gaussian", "Ring8", "Two Moons"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = audit(args.input)
    output = args.input / "output"
    records = json.loads((output / "summary.json").read_text())["results"]
    destination = args.output
    destination.mkdir(parents=True, exist_ok=True)
    figures = destination / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})

    def save(fig: plt.Figure, name: str) -> None:
        fig.savefig(figures / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(figures / f"{name}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    def selected(dataset: str, kind: str, primary: bool = True) -> dict:
        return next(r for r in records if r["dataset"] == dataset and r["model"] == kind
                    and r["primary" if primary else "selected"])

    table = []
    for dataset in DATASETS:
        for kind in KINDS:
            primary, chosen = selected(dataset, kind), selected(dataset, kind, False)
            table.append(dict(dataset=dataset, model=kind, primary_test_sw2=primary["test_sw2"],
                              validation_selected_step=chosen["step"],
                              selected_test_sw2=chosen["test_sw2"],
                              primary_oracle_mse=primary["validation_oracle_mse"],
                              parameters=primary["parameters"], forward_macs=primary["forward_macs"]))
    with (destination / "results.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    for ax, dataset, title in zip(axes, DATASETS, NAMES):
        for i, (kind, color) in enumerate(zip(KINDS, COLORS)):
            p, s = selected(dataset, kind), selected(dataset, kind, False)
            ax.plot([i, i], [p["test_sw2"], s["test_sw2"]], color=color, alpha=.6)
            ax.scatter(i, p["test_sw2"], marker="o", color=color, s=45,
                       label="Step 5,000" if i == 0 else None)
            ax.scatter(i, s["test_sw2"], marker="x", color=color, s=55,
                       label="Validation-selected" if i == 0 else None)
        ax.set(title=title, ylabel="Test SW2 (lower is better)", xticks=range(5),
               xticklabels=LABELS, ylim=(0, None))
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=8)
    save(fig, "test-endpoints")

    trajectories = {}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    for ax, dataset, title in zip(axes, DATASETS, NAMES):
        for kind, label, color in zip(KINDS, LABELS, COLORS):
            path = output / "runs" / f"gates_20261021_{kind}" / dataset / kind / "seed000/metrics.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            trajectories[(dataset, kind)] = rows
            samples = [r for r in rows if "sample" in r and r["step"] > 0]
            ax.plot([r["step"] for r in samples], [r["sample"]["sliced_w2"] for r in samples],
                    color=color, marker="o", markersize=3, label=label)
        ax.set(title=title, xlabel="Training update", ylabel="Validation SW2",
               ylim=(0, None))
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    save(fig, "validation-trajectories")

    fig, axes = plt.subplots(3, 6, figsize=(13, 7.5), layout="constrained")
    for row, (dataset, title) in enumerate(zip(DATASETS, NAMES)):
        arrays = []
        for kind in KINDS:
            with np.load(output / "samples" / f"{dataset}-{kind}-5000.npz") as data:
                arrays.append(data["generated"].copy())
                target = data["target"].copy()
        arrays.insert(0, target)
        all_points = np.concatenate(arrays)
        lo, hi = all_points.min(axis=0), all_points.max(axis=0)
        pad = (hi-lo) * .06
        for column, (points, label, color) in enumerate(zip(arrays, ["Target", *LABELS], ["#222222", *COLORS])):
            ax = axes[row, column]
            ax.scatter(points[:, 0], points[:, 1], s=2, alpha=.3, color=color, rasterized=True)
            ax.set(xlim=(lo[0]-pad[0], hi[0]+pad[0]), ylim=(lo[1]-pad[1], hi[1]+pad[1]))
            ax.set_aspect("equal", adjustable="box")
            if row == 0:
                ax.set_title(label)
            if column == 0:
                ax.set_ylabel(title)
    save(fig, "samples-step5000")

    fig, axes = plt.subplots(3, 3, figsize=(12, 8), layout="constrained")
    for column, (dataset, title) in enumerate(zip(DATASETS, NAMES)):
        for kind, label, color in zip(KINDS[:3], LABELS[:3], COLORS[:3]):
            rows = trajectories[(dataset, kind)]
            steps = [r["step"] for r in rows]
            raw = [np.mean([layer["effective"]["sigma_max"] for layer in r["ode_layers"]])
                   for r in rows]
            effective = [np.mean([layer["activation"]["s_mean"] * layer["effective"]["sigma_max"]
                                  for layer in r["ode_layers"]]) for r in rows]
            duration = [np.mean([layer["activation"]["s_mean"] for layer in r["ode_layers"]]) for r in rows]
            axes[0, column].plot(steps, raw, color=color, label=label)
            axes[1, column].plot(steps, effective, color=color, label=label)
            axes[2, column].plot(steps, duration, color=color, label=label)
        axes[0, column].set(title=title, ylabel="Mean ||P|| over layers")
        axes[1, column].set(ylabel="Mean of s ||P|| over probes/layers")
        axes[2, column].set(xlabel="Training update", ylabel="Mean duration s", ylim=(.2, 1.))
        for ax in axes[:, column]:
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    save(fig, "effective-operator-trajectories")
    provenance = json.loads((output / "runs/gates_20261021_ode_original/gmm_ring8_2d/"
                             "ode_original/seed000/provenance.json").read_text())
    distribution = provenance["dataset_specification"]["distribution"]
    means = np.asarray(distribution["means"])
    assert np.allclose(distribution["variances"], .04)
    case = []
    for index, kind in enumerate(KINDS):
        with np.load(output / "samples" / f"gmm_ring8_2d-{kind}-5000.npz") as samples:
            arrays = [(kind, samples["generated"])]
            if index == 0:
                arrays.insert(0, ("target", samples["target"]))
            for name, points in arrays:
                distances = np.linalg.norm(points[:, None, :] - means[None, :, :], axis=-1)
                nearest = distances.argmin(axis=1)
                frequencies = np.bincount(nearest, minlength=8) / len(points)
                case.append(dict(model=name, sample_count=len(points), radius=.6,
                                 off_mode_fraction=float((distances.min(axis=1) > .6).mean()),
                                 nearest_mode_frequencies=frequencies.tolist(),
                                 frequency_tv_to_uniform=float(abs(frequencies-.125).sum()/2)))
    (destination / "ring8-case-study.json").write_text(json.dumps(case, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(7.5, 3.3), layout="constrained")
    bars = ax.bar(["Target", *LABELS], [100*r["off_mode_fraction"] for r in case],
                  color=["#222222", *COLORS])
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set(ylabel="Points outside all mode neighborhoods (%)",
           title="Ring8 at step 5,000; neighborhood radius 0.6")
    ax.set_ylim(0, max(100*r["off_mode_fraction"] for r in case)*1.2)
    save(fig, "ring8-off-mode-points")
    resolution_path = destination / "sample-count-check.json"
    if resolution_path.exists():
        resolution = json.loads(resolution_path.read_text())["results"]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
        for ax, dataset, title in zip(axes, DATASETS, NAMES):
            for index, (kind, color) in enumerate(zip(KINDS, COLORS)):
                record = next(r for r in resolution if r["dataset"] == dataset
                              and r["model"] == kind and r["primary"])
                ax.scatter(index-.16, record["test_sw2_2048"], marker="x", color=color,
                           label="2,048 / test" if index == 0 else None)
                ax.scatter(index, record["test_sw2_8192"], marker="o", color=color,
                           label="8,192 / test" if index == 0 else None)
                ax.scatter(index+.16, record["validation_target_sw2_8192"], marker="^",
                           facecolors="none", edgecolors=color,
                           label="8,192 / validation targets" if index == 0 else None)
            ax.set(title=title, ylabel="SW2 at fixed step 5,000", xticks=range(5),
                   xticklabels=LABELS, ylim=(0, None))
            ax.tick_params(axis="x", rotation=25)
            ax.grid(axis="y", alpha=.2)
        axes[0].legend(fontsize=7)
        save(fig, "evaluation-sample-count")
    (destination / "audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (destination / "summary.json").write_text(json.dumps(dict(table=table), indent=2) + "\n")
    (destination / "endpoint-diagnostics.json").write_bytes((output / "summary.json").read_bytes())
    (destination / "frozen-inputs.json").write_bytes((output / "frozen-inputs.json").read_bytes())
    (destination / "benchmark.json").write_bytes((output / "benchmark.json").read_bytes())
    (destination / "training-trajectories.json").write_text(json.dumps(
        {f"{dataset}/{kind}": rows for (dataset, kind), rows in trajectories.items()},
        indent=2) + "\n")
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
