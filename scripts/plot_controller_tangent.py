"""Audit and report frozen controller tangents; never update model parameters."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

DATASETS = ("gmm_ring8_2d", "rotated_product_bimodal_16d")
TASK_NAMES = {DATASETS[0]: "Ring8", DATASETS[1]: "Product16"}
ARMS = ("scalar_mul", "scalar_add", "vector_mul", "vector_add", "swiglu_budget", "swiglu")
ARM_NAMES = ("Scalar mul", "Scalar add", "Vector mul", "Vector add", "SwiGLU 96", "SwiGLU 128")
COMPONENTS = ("controller", "controller_value_rms_equalized", "projection_P", "full_ffn")
COMPONENT_NAMES = ("Controller", "RMS-equalized counterfactual", "P", "Full FFN")
COLORS = ("#0072B2", "#D55E00", "#CC79A7", "#222222")
PENALTIES = (1e-4, 1e-3, 1e-2, .1, 1., 10.)
TIMES = (.1, .3, .5, .7, .9)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    if value is None:
        return "—"
    return f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value)


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                       "| " + " | ".join(["---"] * len(headers)) + " |"] +
                      ["| " + " | ".join(map(fmt, row)) + " |" for row in rows]) + "\n"


def penalty_row(component, penalty=.1):
    return next(row for row in component["rows"] if row["penalty"] == penalty)


def audit_and_export(summary_path, output):
    summary = json.loads(summary_path.read_text())
    assert summary["state"] == "COMPLETE"
    rows = summary["rows"]
    expected = {(d, a, s) for d in DATASETS for a in ARMS for s in (1000, 5000)}
    assert len(rows) == 24 and {(r["dataset"], r["label"], r["step"]) for r in rows} == expected
    inputs = {str(summary_path): sha(summary_path)}
    for filename, digest in summary["files_sha256"].items():
        path = Path(filename)
        assert sha(path) == digest, filename
        inputs[str(path)] = digest
    root = summary_path.parent.parent
    source_manifest = root / "source-manifest.json"
    source = json.loads(source_manifest.read_text())
    inputs[str(source_manifest)] = sha(source_manifest)
    for filename, digest in source["files_sha256"].items():
        path = root / "analysis-source" / filename
        assert sha(path) == digest, filename
        inputs[str(path)] = digest
    assert source["unit_tests_passed"] == 6 and source["training"] is False
    assert summary["script_sha256"] == source["files_sha256"]["scripts/diagnose_controller_tangent.py"]
    assert summary["helper_sha256"] == source["files_sha256"]["scripts/controller_tangent.py"]
    repository = Path(__file__).resolve().parents[1]
    for filename in ("protocol-controller-tangent.md", "math-controller-response.md"):
        path = repository / "docs/research-architecture-search" / filename
        inputs[str(path)] = sha(path)

    cases, components, ridges, layers, time_rows = [], [], [], [], []
    max_mse_error = 0.
    for row in rows:
        identity = {k: row[k] for k in ("dataset", "label", "step")}
        full = row["components"]["full_ffn"]
        control = row["components"].get("controller")
        case = dict(identity, original_calibration_mse=full["original_calibration_mse"],
                    original_evaluation_mse=full["original_evaluation_mse"],
                    original_all_anchor_mse=row["raw_oracle_mse"],
                    free_gate_rank_min=row.get("free_gate_rank_min"),
                    free_gate_rank_max=row.get("free_gate_rank_max"),
                    free_gate_residual_floor_share=row.get("free_gate_residual_floor_share"),
                    controller_trace_share=control["trace_scale"] / full["trace_scale"] if control else None,
                    checkpoint_sha256=row["checkpoint_sha256"], config_sha256=row["config_sha256"],
                    dataset_manifest_sha256=row["dataset_manifest_sha256"])
        cases.append(case)
        for layer in row["layers"]:
            layers.append(dict(identity, **layer))
        path = summary_path.parent / row["dataset"] / row["label"] / f"step-{row['step']}" / "response.npz"
        assert inputs[str(path)] == sha(path)
        with np.load(path, allow_pickle=False) as data:
            fit, evaluation = data["fit_indices"], data["evaluation_indices"]
            times = data["t"].ravel()
            residual = data["oracle"] - data["prediction"]
            dimension = residual.shape[1]
            assert residual.shape == (160, 2 if row["dataset"] == DATASETS[0] else 16)
            assert len(fit) == len(evaluation) == 80 and not np.intersect1d(fit, evaluation).size
            assert np.array_equal(fit, np.concatenate([np.arange(i * 32, i * 32 + 16) for i in range(5)]))
            assert np.array_equal(evaluation, fit + 16)
            assert np.allclose(np.unique(times), TIMES, rtol=0, atol=0)
            np.testing.assert_allclose(np.mean(residual ** 2), row["raw_oracle_mse"], rtol=2e-12, atol=1e-14)
            if control:
                p_trace = row["components"]["projection_P"]["trace_scale"]
                np.testing.assert_allclose(control["trace_scale"] + p_trace, full["trace_scale"], rtol=2e-12)
            for component_name, component in row["components"].items():
                assert not component["zero_response"]
                assert tuple(r["penalty"] for r in component["rows"]) == PENALTIES
                np.testing.assert_allclose(np.mean(residual[fit] ** 2), component["original_calibration_mse"], rtol=2e-12)
                np.testing.assert_allclose(np.mean(residual[evaluation] ** 2), component["original_evaluation_mse"], rtol=2e-12)
                eigenvalues = data[component_name + "__calibration_eigenvalues"]
                np.testing.assert_allclose(eigenvalues.sum() ** 2 / (eigenvalues @ eigenvalues), component["effective_rank"], rtol=2e-12)
                component_row = dict(identity, component=component_name, calibration_coordinates=80 * dimension,
                                     **{k: v for k, v in component.items() if k != "rows"})
                components.append(component_row)
                for entry in component["rows"]:
                    correction = data[component_name + f"__evaluation_correction_{entry['penalty']}"]
                    after = residual[evaluation] - correction
                    recomputed = float(np.mean(after ** 2))
                    max_mse_error = max(max_mse_error, abs(recomputed - entry["evaluation_mse"]))
                    np.testing.assert_allclose(recomputed, entry["evaluation_mse"], rtol=2e-12, atol=1e-14)
                    np.testing.assert_allclose(entry["calibration_mse"] / component["original_calibration_mse"], entry["calibration_ratio"], rtol=2e-12)
                    np.testing.assert_allclose(recomputed / component["original_evaluation_mse"], entry["evaluation_ratio"], rtol=2e-12)
                    ridges.append(dict(identity, component=component_name,
                                       original_calibration_mse=component["original_calibration_mse"],
                                       original_evaluation_mse=component["original_evaluation_mse"], **entry,
                                       evaluation_correction_rms=float(np.sqrt(np.mean(correction ** 2)))))
                    for time in TIMES:
                        mask = times[evaluation] == time
                        assert mask.sum() == 16
                        original = float(np.mean(residual[evaluation][mask] ** 2))
                        corrected = float(np.mean(after[mask] ** 2))
                        time_rows.append(dict(identity, component=component_name, penalty=entry["penalty"], time=time,
                                              original_evaluation_mse=original, evaluation_mse=corrected,
                                              evaluation_ratio=corrected / original,
                                              evaluation_correction_rms=float(np.sqrt(np.mean(correction[mask] ** 2)))))
    assert (len(cases), len(components), len(ridges), len(layers), len(time_rows)) == (24, 72, 432, 96, 2160)
    for name, values in (("cases", cases), ("components", components), ("ridge-curves", ridges),
                         ("layers", layers), ("evaluation-by-time", time_rows)):
        write_csv(output / f"{name}.csv", values)
    audit = dict(state="PASS", cases=24, kernels=72, ridge_rows=432, layer_rows=96,
                 evaluation_time_rows=2160, response_archives=24, upstream_hash_entries_verified=50,
                 frozen_source_files_verified=4, upstream_kernel_unit_tests=6,
                 fit_paths=16, evaluation_paths=16, times=list(TIMES), penalties=list(PENALTIES),
                 max_evaluation_mse_reconstruction_error=max_mse_error,
                 training=False, weights_updated=False, input_sha256=inputs)
    (output / "analysis-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return rows, cases, components, ridges, layers, time_rows, audit


def make_plots(rows, output):
    lookup = {(r["dataset"], r["label"], r["step"]): r for r in rows}
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    columns = [(d, s) for d in DATASETS for s in (1000, 5000)]
    fig, axes = plt.subplots(4, 6, figsize=(19, 12.5), sharex=True, sharey=True)
    for row_index, (dataset, step) in enumerate(columns):
        for arm_index, arm in enumerate(ARMS):
            ax = axes[row_index, arm_index]
            record = lookup[dataset, arm, step]
            for component_name, color in zip(COMPONENTS, COLORS):
                if component_name not in record["components"]:
                    continue
                entries = record["components"][component_name]["rows"]
                for split, style in (("calibration", "--"), ("evaluation", "-")):
                    ax.plot(PENALTIES, [e[split + "_ratio"] for e in entries], color=color,
                            ls=style, marker="o" if split == "evaluation" else None,
                            markersize=2.3, lw=1.25, alpha=.9)
            ax.axhline(1, color=".5", lw=.8, ls=":")
            ax.set(xscale="log", yscale="symlog", ylim=(0, 10), xlim=(7e-5, 14))
            ax.set_yscale("symlog", linthresh=.05, linscale=.5)
            ax.set_yticks([0, .05, .1, .5, 1, 2, 5, 10], labels=["0", ".05", ".1", ".5", "1", "2", "5", "10"])
            ax.set_xticks([1e-4, 1e-2, 1, 10], labels=["1e−4", ".01", "1", "10"])
            ax.grid(axis="y", alpha=.13)
            if row_index == 0:
                ax.set_title(ARM_NAMES[arm_index], fontsize=11)
            if arm_index == 0:
                ax.set_ylabel(f"{TASK_NAMES[dataset]} · step {step}\nResidual MSE / original MSE")
            if row_index == 3:
                ax.set_xlabel("Fixed ridge penalty λ")
    handles = [Line2D([], [], color=c, label=n, lw=2) for c, n in zip(COLORS, COMPONENT_NAMES)]
    handles.extend([Line2D([], [], color=".4", ls="--", label="Calibration anchors"),
                    Line2D([], [], color=".4", marker="o", label="Exposed evaluation anchors")])
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, .948), ncol=3, frameon=False)
    fig.suptitle("Frozen tangent predictions across every declared ridge penalty", fontsize=15, y=.991)
    fig.text(.5, .957, "24 checkpoints; no weight update. Each ratio uses that checkpoint and split's original MSE. Dotted line: no correction.", ha="center", fontsize=10)
    fig.text(.5, .017, "Y scale: linear below .05, logarithmic above. Full FFN is the only SwiGLU component. Stage15 cross-family FiLM initialization was not paired.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.063, right=.99, bottom=.07, top=.875, wspace=.13, hspace=.22)
    for extension in ("png", "pdf"):
        fig.savefig(figures / f"all-ridge-curves.{extension}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 4, figsize=(15.5, 9.5))
    x = np.arange(6)
    for column, (dataset, step) in enumerate(columns):
        records = [lookup[dataset, arm, step] for arm in ARMS]
        axes[0, column].bar(x[:4], [r["free_gate_residual_floor_share"] for r in records[:4]],
                            color=["#56B4E9", "#E69F00", "#56B4E9", "#E69F00"], width=.7)
        for i, record in enumerate(records[:4]):
            floor = record["free_gate_residual_floor_share"]
            axes[0, column].text(i, floor + .025, "≈0" if floor < 1e-12 else f"{floor:.3f}", ha="center", fontsize=8)
        for i in (4, 5):
            axes[0, column].text(i, .04, "n/a", ha="center", fontsize=8, color=".4")
        axes[0, column].set(ylim=(0, .85), title=f"{TASK_NAMES[dataset]} · step {step}")
        for offset, component_name, color in ((-.23, "controller", COLORS[0]), (0, "projection_P", COLORS[2]), (.23, "full_ffn", COLORS[3])):
            positions = [i for i, r in enumerate(records) if component_name in r["components"]]
            axes[1, column].bar(np.array(positions) + offset, [records[i]["components"][component_name]["trace_scale"] for i in positions], width=.22, color=color)
            axes[2, column].bar(np.array(positions) + offset, [records[i]["components"][component_name]["effective_rank"] for i in positions], width=.22, color=color)
        axes[1, column].set(yscale="log", ylim=(1e-3, 1e4))
        axes[2, column].set(ylim=(0, 100))
        for ax in axes[:, column]:
            ax.set_xticks(x, ["S mul", "S add", "V mul", "V add", "Swi96", "Swi128"], rotation=40, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=.12)
        if column == 0:
            axes[0, column].set_ylabel("Pointwise free-gate\nresidual energy share")
            axes[1, column].set_ylabel("Native calibration kernel\nmean diagonal (log scale)")
            axes[2, column].set_ylabel("Kernel participation rank\n(tr K)² / tr(K²)")
    handles = [Line2D([], [], color=COLORS[i], lw=5, label=COMPONENT_NAMES[i]) for i in (0, 2, 3)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, .945), ncol=3, frameon=False)
    fig.suptitle("Free pointwise control and shared parameter responses answer different questions", fontsize=14, y=.99)
    fig.text(.5, .952, "Free gate floors use all 160 anchors; kernels use 80 calibration anchors. Calibration output dimensions: Ring8 160; Product16 1280.", ha="center", fontsize=9)
    fig.text(.5, .02, "Native coordinates only; the RMS-equalized counterfactual is excluded here. Effective rank is spectral concentration, not algebraic rank or a training guarantee.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.08, right=.99, bottom=.09, top=.875, hspace=.49, wspace=.22)
    for extension in ("png", "pdf"):
        fig.savefig(figures / f"native-response-geometry.{extension}", dpi=180)
    plt.close(fig)


def make_report(rows, cases, components, ridges, layers, time_rows, audit, output):
    lookup = {(r["dataset"], r["label"], r["step"]): r for r in rows}
    report = """# What the frozen controller tangents show

The vector models have enough independent gate directions to remove the measured velocity residual **at each anchor separately**. On Product16, that pointwise capacity does not establish a useful shared-controller correction: fitting a shared tangent to 16 exposed paths usually worsens error on the other 16 paths, including for both SwiGLU references. This measurement rejects a missing-pointwise-direction explanation for the vector models; it does not identify a new architecture or explain the training gap by receiver amplitude alone.

## What was measured

The completed Stage15 networks were frozen at 1,000 and 5,000 updates: four ODE variants and two SwiGLU budgets, on Ring8 and Product16. All 24 cases are included. At each of five times (.1, .3, .5, .7, .9), path indices 0:16 supply calibration anchors and 256:272 supply evaluation anchors. These are existing, exposed validation paths with population-oracle velocity labels; neither split is a fresh holdout. The same paths recur across time.

Exact kernels describe the response to the six shared controller parameter blocks, the six projection matrices P, and the complete FFN parameters. All downstream gates recompute natively. The complete-FFN kernel excludes the input/output heads, time conditioning and other non-FFN parameters. SwiGLU is compared through its complete FFN, not a fictitious controller. The original Stage15 ODE/SwiGLU FiLM initial tensors were not paired; cross-family comparisons here remain external references. The [separate FiLM pairing control](../analysis-backbone-pairing/analysis-report.md) is now complete and audited: the better paired SwiGLU baseline has Product16 MSE .092326 versus .090831 previously, and Ring8 MSE .015296 versus .015184. Those training endpoints use the full validation bank, not this diagnostic's 16 evaluation paths. The tangent kernels here still use the original unpaired checkpoints; they were not recomputed for the new paired SwiGLU models.

Each kernel was divided by its calibration mean diagonal and evaluated at all six declared penalties, λ = 1e−4, 1e−3, .01, .1, 1, 10. No penalty was selected as a winner. Numerical results below at λ=.1 are one fixed illustration, not a tuned operating point. No weights were changed. A predicted tangent correction is not an achieved nonlinear-network error.

![Every ridge curve](figures/all-ridge-curves.png)

## 1. The scalar and vector restrictions differ

For one vector-gated solve, the direct output response is `G diag(q)`, where q is Ph for multiplication or h for addition, and G is the complete downstream Jacobian. Nonzero q coordinates preserve G's column span. Changing the receiver can change magnitudes and conditioning without creating new pointwise directions under the same G. Across trained networks G itself differs.

On Product16, all scalar checkpoints have free-gate rank six, the number of scalar controllers. Their unexplained residual energy is 60.2–61.4% at 1,000 updates and 70.0–70.4% at 5,000. Both vector models have rank 16 and numerical-zero floors (about 1e−30). On Ring8 every ODE case has rank two and a numerical-zero floor. These are unbounded, independent gate changes at each anchor; finite gate ranges and the shared MLP are relaxed. They are not approximation lower bounds after retraining.

## 2. Native controller sensitivity is small, but amplitude is not a diagnosis

The controller accounts for 0.0237–12.42% of native complete-FFN kernel trace across these ODE cases; P accounts for the remainder. This is a statement in the actual Euclidean parameter coordinates. It is not invariant to reparameterization, and the parameter blocks need not have equal sizes. The participation rank measures spectral concentration, not algebraic dimension. Both traces and ranks are shown without discarding the early checkpoints.

![Native response geometry](figures/native-response-geometry.png)

Receiver-RMS equalization changes each layer's response separately at each input. It therefore changes the shared fitting problem and is not an equivalent parameter coordinate system. It retains channel imbalance, gate slopes, controller features and downstream anisotropy. On Product16 both native and equalized controllers worsen evaluation in 46 of their 48 fixed-penalty cells. This supplies no clean amplitude-causality result and does not justify another amplitude tweak.

## 3. Calibration fit is not evaluation improvement

Product16's full-FFN tangent worsens evaluation in 67 of 72 fixed-penalty cells. The five exceptions are weak corrections at λ=10, with a best residual ratio of .965644; none constitutes a trained improvement. At λ=.1, all 12 Product16 checkpoints worsen evaluation, including both SwiGLUs. At the smallest penalty, the final full-FFN calibration residual shares are 0.0000118–0.000458, while evaluation shares are 2.09–4.36. Near-interpolation of this small calibration bank is therefore poor evidence of a useful repair.

Ring8 provides an important counterexample. At 1,000 updates, every full-FFN penalty worsens evaluation (36/36 cells). At 5,000, every one improves its linearized evaluation residual (36/36), with ratios .388477–.766472. This diagnosis depends on task and trained representation; it does not show that tangent fitting always fails. The per-time CSV preserves all five time panels without treating them as independent repetitions.

The tables retain each split's original MSE. Ratios across different starting models do not establish a performance ranking. The fixed λ=.1 columns are illustrative only; the figure and statistical appendix retain the complete ridge grid.

"""
    for dataset in DATASETS:
        report += f"### {TASK_NAMES[dataset]}: complete-FFN illustration at λ=.1\n\n"
        records = []
        for step in (1000, 5000):
            for arm, name in zip(ARMS, ARM_NAMES):
                r = lookup[dataset, arm, step]
                c = r["components"]["full_ffn"]
                p = penalty_row(c)
                records.append([step, name, c["original_calibration_mse"], c["original_evaluation_mse"],
                                p["calibration_ratio"], p["evaluation_ratio"], p["feature_update_norm"]])
        report += table(["Update", "Model", "Original calibration MSE", "Original evaluation MSE", "Calibration share", "Evaluation share", "Implied update norm"], records) + "\n"
    report += """## Case study: Product16 vector addition

At 5,000 updates the vector additive network has original evaluation MSE .170235, compared with .274190 for vector multiplication on these anchors. Its free-gate floor is already numerical zero, as is the parent's. At λ=.1 its shared controller gives evaluation ratio 1.257, and RMS equalization gives 1.242: neither is a repair. Its full FFN gives ratio 1.353 despite calibration ratio .352. Small SwiGLU starts at .093969 and also overfits this correction, giving evaluation ratio 1.237. Thus this diagnostic cannot turn the additive model's observed gain into evidence for a new pointwise direction or a universally bad ODE tangent.

Native controller update norms at this penalty are 4.854 for vector multiplication and 1.831 for vector addition; full-FFN norms are .896 and .670. A ridge penalty does not ensure a local, feasible nonlinear change. These are absolute Euclidean parameter norms without a relative parameter-norm or finite-step validation; no nonlinear update was applied. Counterfactual feature norms have no direct network-parameter meaning. All 432 norms and original MSEs are exported, including the much larger norms at small penalties.

## What follows, and what does not

The vector models' pointwise span is sufficient here, so a design justified only by adding those same directions lacks support. The weak transfer of tiny-bank corrections is shared by ODE and SwiGLU on Product16. It cannot isolate an ODE-specific optimization failure, an architecture lower bound, or a receiver-amplitude cause. The native controller/P contrast motivates separating learned projection features from controller accessibility, but does not choose feature sharing, a new generator, a temperature or a higher order.

A proposed architecture still needs an independent mechanism argument and matched control registration. The [completed FiLM pairing control](../analysis-backbone-pairing/analysis-report.md) leaves the main ODE deficit in place, but does not retroactively pair the original tangent checkpoints. This bundle remains a posthoc diagnostic of those existing checkpoints, not an additional training result.

## Files and verification

The analysis audited 24 cases, 72 kernels, 432 ridge rows, 96 layer summaries and 2,160 evaluation-by-time rows. It checked all 50 upstream file hashes and four frozen diagnostic-source hashes. The frozen manifest records six independently passed kernel tests. Stored evaluation corrections reproduce every evaluation MSE; kernel spectra reproduce every participation rank, and controller + P traces reproduce the ODE full-FFN trace. These are analysis consistency checks, not another training run.

- [Statistical appendix with every numeric ridge row](stats-appendix.md)
- [Exact case values](cases.csv), [kernel values](components.csv), [ridge values and norms](ridge-curves.csv), [layer values](layers.csv), [time panels](evaluation-by-time.csv)
- [Figure catalog](figure-catalog.md), [analysis audit](analysis-audit.json), [file hashes](SHA256SUMS.json)

CSV floats preserve Python's round-trip representation of the JSON/NPZ values. Markdown displays six significant digits; it does not replace the exact exports.
"""
    (output / "analysis-report.md").write_text(report)

    appendix = """# Statistical appendix: frozen controller tangent diagnosis

All outputs are descriptive, conditional on the saved weights, anchors and parameter coordinates. There are no significance tests, confidence intervals or independent time replicates. This is an exposed-anchor diagnostic using privileged population-oracle labels, not a fresh generalization estimate or a finite nonlinear update. Every predeclared checkpoint and ridge value is retained.

## Estimands and normalization

For residual e = oracle − prediction, the native parameter kernel is K = J Jᵀ. Fit anchors comprise 16 paths at five times (80 anchors); output-coordinate counts are 160 for Ring8 and 1,280 for Product16. Let c = trace(K_fit,fit)/n. The normalized blocks are K/c. For each fixed λ, the dual coefficient is `(K_fit,fit/c + λI)^−1 e_fit`; multiplying it by the corresponding normalized cross-kernel gives the predicted evaluation correction. MSE averages over anchors and output coordinates. Each residual share divides by the original MSE on the same split.

The recorded native update norm is `sqrt(alphaᵀ (K_fit,fit/c) alpha / c)`, the Euclidean norm of the implied native parameter change. It is not a fractional parameter change or proof that linearization is valid at that finite displacement. The receiver-RMS counterfactual has only a feature-space norm. It is not a fixed parameter-coordinate change. Participation rank is `(sum eigenvalues)^2 / sum(eigenvalues^2)`, not the number of nonzero eigenvalues.

The free-gate floor uses all 160 anchors and unbounded independent controls. Numerical rank keeps singular values above `max(1e−12, largest singular value × 1e−12)`. Its residual share is the ratio of summed squared projected residuals to summed squared original residuals. Tiny negative eigenvalues are checked against a relative numerical tolerance and clipped to zero in the frozen ridge routine; no case has zero trace. Per-layer controller traces use all 160 anchors, whereas component mean diagonals use calibration anchors only; do not sum the former to reproduce the latter.

Complete FFN excludes non-FFN parameters. Original ODE/SwiGLU FiLM initializations were unpaired, so cross-family references are not an isolated FFN causal comparison. Time rows reuse the same 16 paths. All λ=.1 examples in the report were fixed illustrations, not selected minima.

## Case and kernel tables

"""
    for dataset in DATASETS:
        appendix += f"### {TASK_NAMES[dataset]}\n\n"
        selected = [r for r in cases if r["dataset"] == dataset]
        appendix += table(["Model", "Update", "All-anchor MSE", "Gate rank min", "Gate rank max", "Free-gate floor", "Controller trace share"],
                          [[r["label"], r["step"], r["original_all_anchor_mse"], r["free_gate_rank_min"], r["free_gate_rank_max"],
                            r["free_gate_residual_floor_share"], r["controller_trace_share"]] for r in selected]) + "\n"
    appendix += table(["Task", "Model", "Update", "Component", "Native mean diagonal", "Participation rank", "Min normalized eigenvalue", "Max normalized eigenvalue"],
                      [[TASK_NAMES[r["dataset"]], r["label"], r["step"], r["component"], r["trace_scale"], r["effective_rank"],
                        r["normalized_eigen_min"], r["normalized_eigen_max"]] for r in components]) + "\n"
    appendix += "## Every ridge row\n\nEach table states the original MSE for both splits. Norms are native parameter norms except for the explicitly named RMS-equalized feature counterfactual. Exact values and per-time MSEs are in the CSV exports.\n\n"
    for dataset in DATASETS:
        for step in (1000, 5000):
            for arm in ARMS:
                r = lookup[dataset, arm, step]
                c = r["components"]["full_ffn"]
                appendix += f"### {TASK_NAMES[dataset]} / {arm} / update {step}\n\nOriginal calibration MSE: {c['original_calibration_mse']:.9g}; original evaluation MSE: {c['original_evaluation_mse']:.9g}.\n\n"
                selected = [a for a in ridges if (a["dataset"], a["label"], a["step"]) == (dataset, arm, step)]
                appendix += table(["Component", "λ", "Calibration MSE", "Evaluation MSE", "Calibration share", "Evaluation share", "Update norm", "Evaluation correction RMS"],
                                  [[a[k] for k in ("component", "penalty", "calibration_mse", "evaluation_mse", "calibration_ratio", "evaluation_ratio", "feature_update_norm", "evaluation_correction_rms")] for a in selected]) + "\n"
    appendix += f"## Audit coverage\n\n24 cases; 72 kernels; 432 ridge rows; 96 layer rows; 2,160 time rows. Maximum absolute discrepancy when reconstructing stored evaluation MSEs: {audit['max_evaluation_mse_reconstruction_error']:.17g}. No case was removed. See `analysis-audit.json` for input hashes and `SHA256SUMS.json` for generated files.\n"
    (output / "stats-appendix.md").write_text(appendix)
    catalog = """# Figure catalog

| Figure | Coverage and encoding | Interpretation and limits |
| --- | --- | --- |
| [Every ridge curve, PNG](figures/all-ridge-curves.png) / [PDF](figures/all-ridge-curves.pdf) | 24 panels: both tasks × both steps × six models. All 72 kernels and 432 fixed-penalty rows. Color identifies component; solid lines show evaluation and dashed lines calibration. Y scale is linear below .05, logarithmic above; all panels share limits. Ratio 1 means unchanged error. | Demonstrates tiny-bank fit/evaluation divergence without selecting λ. Each ratio has its own checkpoint/split denominator. These are tangent predictions on exposed anchors, not finite trained updates. Only full FFN applies to SwiGLU. |
| [Native response geometry, PNG](figures/native-response-geometry.png) / [PDF](figures/native-response-geometry.pdf) | Four columns: Ring8/Product16 × steps 1,000/5,000. Rows show free-gate residual energy floor, native kernel mean diagonal, and participation rank. Every ODE checkpoint and both SwiGLU references are retained. Controller/P bars are absent for SwiGLU, not zero. | Contrasts independent pointwise gate capacity with shared parameter sensitivity and spectral concentration. RMS-equalized counterfactual is excluded from native statistics. Trace depends on parameter coordinates; participation rank is not algebraic rank. Free floors use 160 anchors; calibration kernels 80. |

Both figures originate from the exact exports in this folder and the immutable diagnostic summary/response archives. Original Stage15 ODE/SwiGLU FiLM initialization was unpaired. All five evaluation times are preserved in `evaluation-by-time.csv`; they reuse paths and are not independent experiments. PDF exports contain vector graphics and embedded fonts. See `plot-provenance.json` for the generation command, versions and script hash.
"""
    (output / "figure-catalog.md").write_text(catalog)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, cases, components, ridges, layers, time_rows, audit = audit_and_export(args.summary.resolve(), args.output)
    make_plots(rows, args.output)
    make_report(rows, cases, components, ridges, layers, time_rows, audit, args.output)
    provenance = dict(script_sha256=sha(Path(__file__)), numpy_version=np.__version__, matplotlib_version=matplotlib.__version__,
                      command=f"python3 scripts/plot_controller_tangent.py --summary {args.summary} --output {args.output}",
                      scope="Posthoc frozen tangent predictions only; no training, no weight updates, no chosen winning ridge penalty.",
                      input_sha256=audit["input_sha256"])
    (args.output / "plot-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    hashes = {str(path.relative_to(args.output)): sha(path) for path in sorted(args.output.rglob("*"))
              if path.is_file() and path.name != "SHA256SUMS.json"}
    hashes["../../../scripts/plot_controller_tangent.py"] = sha(Path(__file__))
    (args.output / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k != "input_sha256"}, indent=2))


if __name__ == "__main__":
    main()
