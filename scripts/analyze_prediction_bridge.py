"""Recompute the registered prediction-bridge screen from saved native arrays."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def table(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(study: Path, output: Path) -> None:
    state = load(study/"output/status.json")
    if state["state"] != "COMPLETE" or state["completed"] != 24:
        raise ValueError("all 24 cells must be terminal before selection")
    output.mkdir(parents=True, exist_ok=False)
    source = study/"source"
    for name, expected in load(source/"source-sha256.json").items():
        if digest(source/name) != expected:
            raise ValueError("frozen source mismatch: "+name)
    rows, curves, spectra, receipts = [], [], [], []
    rng_by_dataset: dict[str, dict[int, torch.Tensor]] = {}
    chain_by_dataset: dict[str, dict[int, str]] = {}
    common_by_dataset: dict[str, dict] = {}
    raw_by_dataset_backbone: dict[tuple, dict] = {}
    run_root = study/"output/runs"
    for run in sorted(run_root.glob("*/*")):
        native = load(run/"result.json")
        if load(run/"status.json")["state"] != "COMPLETE" or native["step"] != 5000:
            raise ValueError("incomplete native cell")
        dataset = native["dataset"]
        label = native["backbone"]+"_"+native["prediction"]
        banks = study/"output/banks"/dataset
        actual = {"dataset": dataset, "arm": label, "backbone": native["backbone"],
                  "prediction": native["prediction"], "parameters": native["parameters"],
                  "dense_forward_macs": native["dense_forward_macs"]}
        for name in ("interior", "native", "uniform_clipped"):
            with np.load(banks/(name+".npz")) as data:
                oracle = data["oracle"]
            with np.load(run/"diagnostics"/(name+".npz")) as data:
                prediction = data["prediction"]
            if prediction.shape != oracle.shape or prediction.shape[0] != 20000:
                raise ValueError("unexpected field array shape")
            if not np.isfinite(prediction).all():
                raise ValueError("nonfinite predictions")
            error = float(np.mean((prediction-oracle)**2))
            if error != native[name+"_oracle_mse"]:
                raise ValueError("native field metric mismatch")
            actual[name+"_oracle_mse"] = error
        jac_errors = []
        for time in (.1, .3, .5, .7, .9):
            with np.load(banks/f"panel_{time:.1f}.npz") as data:
                truth = data["oracle_jacobian"]
            with np.load(run/"diagnostics"/f"panel_{time:.1f}.npz") as data:
                prediction = data["jacobian"]
            if prediction.shape != truth.shape or prediction.shape[0] != 512:
                raise ValueError("unexpected derivative array shape")
            jac_errors.append(float(np.mean(np.sum((prediction-truth)**2, axis=(1, 2)))))
        actual["jacobian_error"] = float(np.mean(jac_errors))
        if actual["jacobian_error"] != native["jacobian_error"]:
            raise ValueError("native derivative metric mismatch")
        with np.load(banks/"sampling.npz") as data:
            target, directions = data["target"].astype(np.float64), data["directions"].astype(np.float64)
        with np.load(run/"diagnostics/samples.npz") as data:
            generated = data["generated"].astype(np.float64)
        if generated.shape != target.shape or len(generated) != 2048:
            raise ValueError("unexpected sample array shape")
        actual["sw2"] = float(np.sqrt(np.mean((np.sort(generated@directions.T, axis=0)
                                               - np.sort(target@directions.T, axis=0))**2)))
        if abs(actual["sw2"]-native["sw2"]) > 2e-6:
            raise ValueError("independent SW2 recomputation mismatch")
        actual["sw2_native"] = native["sw2"]
        for step in (0, 1000, 2500, 5000):
            checkpoint = torch.load(run/f"step_{step:07d}.pt", map_location="cpu", weights_only=False)
            if checkpoint["step"] != step:
                raise ValueError("checkpoint step differs")
            if not all(torch.isfinite(tensor).all() for tensor in checkpoint["model"].values()):
                raise ValueError("nonfinite checkpoint")
            prior_rng = rng_by_dataset.setdefault(dataset, {}).setdefault(step, checkpoint["data_rng"])
            if not torch.equal(prior_rng, checkpoint["data_rng"]):
                raise ValueError("cross-arm data RNG mismatch")
            prior_chain = chain_by_dataset.setdefault(dataset, {}).setdefault(step, checkpoint["input_stream_digest"])
            if prior_chain != checkpoint["input_stream_digest"]:
                raise ValueError("actual training input hash chain differs")
            if step == 0:
                key = (dataset, native["backbone"])
                raw_reference = raw_by_dataset_backbone.setdefault(key, checkpoint["model"])
                if any(not torch.equal(value, raw_reference[name])
                       for name, value in checkpoint["model"].items()):
                    raise ValueError("initial raw weights differ between prediction spaces")
                surrounding = {name: value for name, value in checkpoint["model"].items() if ".ffn." not in name}
                surrounding.update({"buffer:"+name: value for name, value in checkpoint["runtime_buffers"].items()
                                    if ".ffn." not in name})
                reference = common_by_dataset.setdefault(dataset, surrounding)
                if set(reference) != set(surrounding) or any(not torch.equal(value, reference[name])
                                                           for name, value in surrounding.items()):
                    raise ValueError("actual shared-backbone initialization mismatch")
            for name, values in load(run/f"spectrum_{step:07d}.json").items():
                singular = values["singular_values"]
                spectra.append({"dataset": dataset, "arm": label, "step": step, "layer": name,
                                "sigma_max": max(singular), "sigma_min": min(singular),
                                "sigma_mean": float(np.mean(singular)),
                                "eigen_real_max": max(values["eigenvalue_real"]),
                                "eigen_imag_abs_max": max(map(abs, values["eigenvalue_imag"]))})
        metrics = [json.loads(line) for line in (run/"metrics.jsonl").read_text().splitlines()]
        actual["recorded_training_seconds"] = metrics[-1]["wall_seconds"]
        for metric in metrics:
            curves.append({"dataset": dataset, "arm": label, "step": metric["step"],
                           **{key: metric.get(key) for key in ("train_loss", "grad_norm", "clip_fraction",
                              "interior_oracle_mse", "native_oracle_mse", "uniform_clipped_oracle_mse", "wall_seconds")}})
        rows.append(actual)
        receipts.append({"run": str(run), "result_sha256": digest(run/"result.json"),
                         "final_checkpoint_sha256": digest(run/"step_0005000.pt")})
    if len(rows) != 24:
        raise ValueError("expected exactly24 results")
    by_dataset = {dataset: {row["arm"]: row for row in rows if row["dataset"] == dataset}
                  for dataset in sorted({row["dataset"] for row in rows})}
    expected_arms = {label+"_"+prediction for label in ("ode_k1", "ode_k2", "swiglu128", "swiglu192")
                     for prediction in ("v", "x")}
    if set(by_dataset) != {"gaussian_aniso_2d", "gmm_ring8_2d", "rotated_product_bimodal_16d"}:
        raise ValueError("unexpected dataset set")
    if any(set(arms) != expected_arms for arms in by_dataset.values()):
        raise ValueError("unexpected arm set")
    nonlinear = ["gmm_ring8_2d", "rotated_product_bimodal_16d"]
    decisions = []
    for candidate in ("ode_k1_v", "ode_k1_x", "ode_k2_v", "ode_k2_x"):
        conditions = []
        qualifying = []
        comparisons = {}
        for dataset, arms in by_dataset.items():
            controls = [r for r in arms.values() if r["backbone"].startswith("swiglu")]
            baseline = min(controls, key=lambda r: r["interior_oracle_mse"])
            own = arms[candidate]
            ratio = own["interior_oracle_mse"]/baseline["interior_oracle_mse"]
            comparisons[dataset] = {"field_baseline": baseline["arm"], "primary_mse_ratio": ratio,
                "jacobian_ratio": own["jacobian_error"]/min(r["jacobian_error"] for r in controls),
                "sw2_ratio": own["sw2"]/min(r["sw2"] for r in controls),
                "mac_ratio": own["dense_forward_macs"]/baseline["dense_forward_macs"]}
            conditions.append(ratio <= 1.05)
            if dataset in nonlinear:
                conditions.extend([comparisons[dataset]["jacobian_ratio"] <= 1.1,
                                   comparisons[dataset]["sw2_ratio"] <= 1.1])
                if ratio <= .9 and comparisons[dataset]["mac_ratio"] <= 1.05:
                    qualifying.append(dataset)
        accepted = bool(qualifying) and all(conditions)
        decisions.append({"candidate": candidate, "qualifies_for_continuation": accepted,
                          "qualifying_tasks": qualifying, "comparisons": comparisons})
    nominees = [row["candidate"] for row in decisions if row["qualifies_for_continuation"]]
    table(output/"comparison.csv", rows)
    table(output/"training-curves.csv", curves)
    table(output/"p-trajectories.csv", spectra)
    write(output/"decision.json", {"nominees": nominees, "decision": "CONTINUE" if nominees else "STOP",
                                    "candidates": decisions})
    inputs = {str(path.relative_to(study)): digest(path) for path in (study/"output").rglob("*")
              if path.is_file() and path.suffix in (".pt", ".npz")}
    write(output/"audit.json", {"status": "PASS", "cells": len(rows), "checkpoints": 96,
        "source_sha256": load(source/"source-sha256.json"), "receipts": receipts,
        "analysis_source_sha256": digest(Path(__file__)), "input_file_sha256": inputs,
        "actual_input_hash_chains_match": True, "actual_data_rng_states_match": True,
        "actual_raw_initial_weights_paired": True, "actual_non_ffn_initialization_paired": True,
        "all_three_field_metrics_recomputed": True, "jacobian_metrics_recomputed": True,
        "sw2_recomputed_independently_in_float64": True,
        "scope": "Saved native arrays and states audited; positive nominees still need continuation and fresh tests."})
    report = ["# Prediction-space bridge: results", "",
              "All24 registered cells completed. The saved field, derivative and sampling metrics were recomputed; common input streams and initialization controls passed.", "",
              "| Dataset | Arm | Primary field MSE | Native-time MSE | SW2 |", "|---|---|---:|---:|---:|"]
    for row in rows:
        report.append(f"| {row['dataset']} | {row['arm']} | {row['interior_oracle_mse']:.6g} | {row['native_oracle_mse']:.6g} | {row['sw2']:.6g} |")
    report.extend(["", "Decision: "+("continue the declared nominees: "+", ".join(nominees) if nominees else
                   "no ODE arm passes the predeclared quality/transfer/cost screen; stop this fixed setting."), "",
        "This comparison concerns tiny synthetic backbones under the declared common JiT-style prediction/time recipe. It does not establish full JiT image-generation performance or attribute any generic prediction-space effect specifically to an ODE block. Concurrent training wall intervals are retained as records, not isolated latency measurements."])
    (output/"results.md").write_text("\n".join(report)+"\n")
    print(json.dumps({"audit": "PASS", "cells": 24, "nominees": nominees, "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.study.resolve(), args.output.resolve())
