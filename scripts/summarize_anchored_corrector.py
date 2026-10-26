"""Supporting figures and fixed-weight paired statistics for Phase A."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--fixed-point", type=Path, required=True)
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()
    group, results, fixed = args.group.resolve(), args.results.resolve(), args.fixed_point.resolve()
    assert json.loads((results/"audit.json").read_text())["status"]=="PASS"
    dataset = "rotated_product_bimodal_16d"
    config = json.loads((group/"study.json").read_text())
    with (results/"iteration-curves.csv").open() as handle:
        curves = list(csv.DictReader(handle))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    colors = {"ode_k2": "#D55E00", "swiglu192": "#0072B2", "swiglu128": "#009E73"}
    for arm, setting in config["arms"].items():
        selected = [r for r in curves if r["arm"]==arm]
        style = "-" if setting["mode"]=="anchored" else "--"
        axes[0].plot([int(r["updates"]) for r in selected], [float(r["oracle_mse"]) for r in selected],
                     style, color=colors[setting["family"]], marker="o", markersize=3,
                     label=setting["family"]+" "+setting["mode"])
        metrics = json.loads((group/"runs"/dataset/arm/"result.json").read_text())
        if setting["mode"]=="anchored":
            changes = metrics["hidden_changes"]
            axes[1].plot([r["iteration"] for r in changes], [r["step_rms"] for r in changes],
                         color=colors[setting["family"]], marker="o", markersize=3, label=setting["family"])
    axes[0].set(title="Full validation prediction error", xlabel="Executed updates", ylabel="Field MSE")
    axes[0].set_yscale("log")
    axes[0].set_xticks((1, 2, 3, 6, 12))
    axes[1].set(title="Anchored fixed-equation residual on probe", xlabel="Current iteration", ylabel="Residual RMS")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc="lower center", ncol=3)
    fig.suptitle("Solid: anchored; dashed: accumulated. The two panels use different banks.", fontsize=9)
    fig.tight_layout(rect=(0, .13, 1, 1))
    fig.savefig(results/"iteration-behavior.png", dpi=160, bbox_inches="tight")
    fig.savefig(results/"iteration-behavior.pdf", bbox_inches="tight")
    plt.close(fig)
    if args.figures_only:
        return

    # Bootstrap is conditional on the fixed models and common exposed validation inputs.
    with np.load(group/"banks"/dataset/"field.npz") as data:
        truth = data["oracle"].astype(np.float64)
    rng = np.random.default_rng(0)
    paired = []
    for family in colors:
        errors = []
        for rule in ("anchored", "accumulating"):
            with np.load(group/"runs"/dataset/f"{family}_{rule}_final/diagnostics/field.npz") as data:
                errors.append(np.mean((data["prediction"].astype(np.float64)-truth)**2, axis=1))
        difference = errors[0]-errors[1]
        bootstrap = np.concatenate([difference[rng.integers(len(truth), size=(50,len(truth)))].mean(axis=1)
                                    for _ in range(20)])
        interval = np.quantile(bootstrap, (.025,.975))
        paired.append(dict(family=family, evaluation_inputs=len(truth),
            anchored_mse=float(errors[0].mean()), accumulated_mse=float(errors[1].mean()),
            ratio=float(errors[0].mean()/errors[1].mean()), difference=float(difference.mean()),
            conditional_ci95_low=float(interval[0]), conditional_ci95_high=float(interval[1])))
    with (results/"paired-evaluation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader(); writer.writerows(paired)
    (results/"stats-appendix.md").write_text(
        "# Statistical scope\n\nAll formal training uses seed0 and the registered5000-update endpoint. "
        "The three within-family update-rule contrasts are paired at each common validation input. "
        "The displayed95% percentile intervals use1000 paired evaluation-row resamples, bootstrap seed0, conditional on fixed weights. "
        "They are pointwise evaluation intervals; no p-values, training-repetition significance, familywise discovery claim, or run-level standard deviation is computed. "
        "The original nomination thresholds determine continuation. Previously exposed validation inputs are not presented as a fresh test. "
        "The probe diagnostic is post-screen, preregistered before its paired outcomes, uses256 fixed inputs, and selects no architecture.\n")
    (results/"figure-catalog.md").write_text(
        "# Figure catalog\n\nNo figure has error bars; raw pointwise paired evaluation intervals are in paired-evaluation.csv.\n\n"
        "- comparison.pdf/png: all six new endpoints in field, Jacobian and generation metrics. The dashed line in each panel is the best combined generic value, potentially a different arm for each metric. Notice that anchoring worsens every primary family. This stops the setting, without claiming an impossibility result.\n"
        "- iteration-behavior.pdf/png: left, full validation error versus internal update count; right, anchored residual on the256-input probe. Notice the ODE residual decrease alongside worsening validation prediction. The panels use different banks; the separate fixed-point-results files establish a paired same-input comparison. This blocks interpreting numerical convergence as task correctness.\n")
    with (fixed/"summary.csv").open() as handle:
        summary = list(csv.DictReader(handle))
    values = [r for r in summary if r["arm"]=="ode_k2_anchored_final"]
    provenance = json.loads((fixed/"provenance.json").read_text())
    fraction = provenance["checkpoints"]["ode_k2_anchored_final"]["fraction_residual_down_and_error_up"]
    with (results/"analysis-report.md").open("a") as handle:
        handle.write("\nAll three within-family anchored endpoints are worse than accumulated controls; exact effects and conditional intervals are in paired-evaluation.csv. No Ring8/Gaussian extension or longer training is triggered.\n\n![Iteration behavior](iteration-behavior.png)\n\n")
        handle.write("The paired probe diagnostic uses the same256 inputs for residual and task error:\n\n| ODE anchored iteration | Probe field MSE | Hidden residual RMS |\n|---|---:|---:|\n")
        for row in values:
            handle.write(f"| {row['iteration']} | {float(row['field_mse']):.7f} | {float(row['hidden_next_change_rms']):.7f} |\n")
        handle.write(f"\nFrom iteration3 to12, residual falls while field error rises on{fraction:.3%} of those same inputs. This is a finite probe observation, not a global contraction certificate. The effect is also present in the anchored SwiGLU192 control, so it is not ODE-specific.\n\n")
        handle.write("For a hypothetical contraction G with factor q<1 and an L-Lipschitz readout D, the residual only controls distance to its own fixed point:\n\n```text\n||h-h*|| <= ||G(h)-h||/(1-q)\n||D(h)-target|| <= L*||G(h)-h||/(1-q) + ||D(h*)-target||\n```\n\nThe last term is task bias and is not bounded by numerical convergence. No q or global Lipschitz certificate is inferred for the actual network. The practical implication is to specify what hidden evolution should accomplish before imposing additional solver or spectral properties; this screen does not select a new loss, constraint or architecture.\n\n")
        handle.write("Claim candidates: keep the failure of the registered input-anchoring intervention; keep the observed separation of equation residual and prediction accuracy; reject a claim that the original P is now useful, that all implicit architectures fail, or that finite probes prove global stability.\n")
    files = [p for p in results.rglob("*") if p.is_file() and p.name!="artifact-check.json"]
    (results/"artifact-check.json").write_text(json.dumps(dict(
        figure_status="pending visual inspection", script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        files={str(p.relative_to(results)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}), indent=2)+"\n")
    print(json.dumps({"paired_contrasts":3,"ode_same_input_fraction":fraction}))


if __name__ == "__main__":
    main()
