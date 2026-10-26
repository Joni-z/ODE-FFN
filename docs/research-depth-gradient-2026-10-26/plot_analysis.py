"""Make descriptive figures from independently verified frozen gradients."""
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main(runtime: Path, report: Path) -> None:
    raw = runtime / "output"
    audit = json.loads((runtime / "independent-analysis/independent-audit.json").read_text())
    assert audit["status"] == "PASS"
    for name, digest in audit["frozen_output_sha256"].items():
        assert hashlib.sha256((runtime / name).read_bytes()).hexdigest() == digest
    figure_dir = report / "figures"
    figure_dir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180, "pdf.fonttype": 42})
    families = ("ode_k2", "swiglu128", "swiglu192")
    labels = ("ODE K2", "SwiGLU 128", "SwiGLU 192")
    steps = (1000, 2500, 5000)
    means = read_csv(raw / "mean-modes.csv")
    batches = read_csv(raw / "batch-modes.csv")
    mean_map = {(r["arm"], int(r["step"]), r["bank"], r["group"]): r for r in means}
    colors = ("#0072B2", "#E69F00", "#CC79A7")
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.6), sharex=True, sharey=True)
    for row, readout in enumerate(("direct", "gradient")):
        for col, (family, label) in enumerate(zip(families, labels)):
            arm = family + "_shared_" + readout
            values = [mean_map[(arm, s, "A", "all")] for s in steps]
            ax = axes[row, col]
            bottom = np.zeros(3)
            for mode, color, name in zip(("E0", "E1", "E2"), colors,
                                          ("Common (tied)", "Linear depth", "Quadratic depth")):
                height = np.array([float(r[mode]) / float(r["total"]) for r in values])
                ax.bar(np.arange(3), height, bottom=bottom, color=color, label=name, width=0.66)
                bottom += height
            for i, value in enumerate(values):
                ax.text(i, float(value["retained"]) / 2, f'{float(value["retained"]):.2f}',
                        ha="center", va="center", color="white", fontsize=10)
            ax.set_title(label + (" / direct" if readout == "direct" else " / potential gradient"))
            ax.set_xticks(np.arange(3), ("1k", "2.5k", "5k"))
            ax.set_ylim(0, 1.03)
            if col == 0:
                ax.set_ylabel("Fraction of block-gradient energy")
            if row == 1:
                ax.set_xlabel("Saved training update")
    handles, legends = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legends, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.96), frameon=False)
    fig.suptitle("Tying omits depth-dependent gradients in every model family", y=1.0, fontsize=14)
    fig.text(0.5, 0.01, "Bank A mean gradient, complete block coordinates. Initialization has zero block gradient and is omitted.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 0.91))
    for ext in ("png", "pdf"):
        fig.savefig(figure_dir / ("all-family-depth-modes." + ext), bbox_inches="tight")
    plt.close(fig)

    primary = "ode_k2_shared_gradient"
    late = (2500, 5000)
    selection = {int(r["step"]): r for r in read_csv(raw / "P-selection.csv") if r["arm"] == primary}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    ax = axes[0, 0]
    for bank, color, marker in (("A", colors[0], "o"), ("B", "#009E73", "s")):
        values = [float(mean_map[(primary, s, bank, "P")]["retained"]) for s in late]
        ax.plot((0, 1), values, marker=marker, color=color, lw=2, label=f"Bank {bank} mean")
    for i, step in enumerate(late):
        values = [float(r["retained"]) for r in batches if r["arm"] == primary and int(r["step"]) == step
                  and r["bank"] == "A" and r["group"] == "P"]
        ax.scatter(i + np.linspace(-0.065, 0.065, 8), values, s=18, color=colors[0], alpha=0.45)
    ax.axhline(0.5, color="black", ls=":", label="Selection requires <= 0.50")
    ax.set_title("Primary condition fails at both snapshots")
    ax.set_ylabel("P energy retained under tying")
    ax.set_xticks((0, 1), ("2.5k", "5k")); ax.set_ylim(0.35, 1.0)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    ax = axes[0, 1]
    x = np.arange(2)
    for shift, key, color, label, threshold in (
        (-0.17, "P_share_of_blocked", colors[0], "P share of blocked energy", 0.5),
        (0.17, "linear_share_of_lost_P", colors[1], "Linear share of lost P energy", 0.7),
    ):
        vals = [float(selection[s][key]) for s in late]
        bars = ax.bar(x + shift, vals, width=0.31, color=color, label=label)
        ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=3)
        ax.axhline(threshold, color=color, ls=":", lw=1)
    ax.set_xticks(x, ("2.5k", "5k")); ax.set_ylim(0, 1.30)
    ax.set_ylabel("Fraction"); ax.set_title("Secondary energy conditions pass")
    ax.legend(frameon=False, fontsize=8, loc="upper center")

    ax = axes[1, 0]
    for shift, key, color, label in ((-0.17, "common_transfer", colors[0], "Common only"),
                                    (0.17, "affine_transfer", colors[1], "Common + linear depth")):
        vals = [float(selection[s][key]) for s in late]
        bars = ax.bar(x + shift, vals, width=0.31, color=color, label=label)
        ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=3)
    ax.set_xticks(x, ("2.5k", "5k")); ax.set_ylim(0, 1.1)
    ax.set_ylabel("Bank A direction alignment with bank B")
    ax.set_title("Depth-dependent directions transfer locally")
    ax.legend(frameon=False, fontsize=8, loc="upper center")

    ax = axes[1, 1]
    for i, (key, label, color) in enumerate(zip(("cosine_01", "cosine_02", "cosine_12"),
                                              ("Visits 1,2", "Visits 1,3", "Visits 2,3"), colors)):
        vals = [float(mean_map[(primary, s, "A", "P")][key]) for s in late]
        ax.bar(x + (i - 1) * 0.23, vals, width=0.21, color=color, label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, ("2.5k", "5k")); ax.set_ylim(-0.05, 1.2)
    ax.set_ylabel("Pairwise P-gradient cosine")
    ax.set_title("Bank-mean gradients are not opposing")
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=3)
    fig.suptitle("Shared-potential ODE: positive secondary evidence does not pass the declared rule", fontsize=13)
    fig.text(0.5, 0.01, "Dots: eight individual bank-A batches, not independent training repeats. No confidence intervals or significance claim.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(figure_dir / ("P-selection-and-transfer." + ext), bbox_inches="tight")
    plt.close(fig)

    # Post-result descriptive factorization, not a new selection rule or direction.
    factor_rows = []
    for file in sorted((raw / "gradients").glob("*/step-*.npz")):
        arm = file.parent.name
        step = int(file.stem.split("-")[1])
        family = arm.split("_shared_")[0]
        groups = json.loads((raw / (family + "-coordinates.json")).read_text())["groups"]
        with np.load(file) as arrays:
            means = arrays["independent"].mean(axis=1)
        for bank, g in zip("AB", means):
            for group, index in groups.items():
                native = g[:, index]
                radii = np.sqrt(np.sum(native ** 2, axis=1))
                total = float(np.sum(radii ** 2))
                radius_sum = float(np.sum(radii))
                norm_sum_squared = float(np.sum(native.sum(axis=0) ** 2))
                amplitude = radius_sum ** 2 / (3 * total) if total > 0 else None
                directional = norm_sum_squared / radius_sum ** 2 if radius_sum > 0 else None
                retained = norm_sum_squared / (3 * total) if total > 0 else None
                if retained is not None:
                    assert np.isclose(amplitude * directional, retained, atol=1e-13)
                factor_rows.append(dict(arm=arm, step=step, bank=bank, group=group,
                    norm_visit1=radii[0], norm_visit2=radii[1], norm_visit3=radii[2],
                    amplitude_factor=amplitude, directional_factor=directional, retained=retained))
    with (report / "amplitude-direction-factors.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(factor_rows[0]))
        writer.writeheader(); writer.writerows(factor_rows)
    print(json.dumps({"figures": 2, "factor_rows": len(factor_rows), "audit": audit["status"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    main(args.runtime.resolve(), args.report.resolve())
