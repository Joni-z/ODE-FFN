"""Audit and compare complete same-host recurrent-depth experiment groups."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def load(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def csv_write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare(name: str, actual: float, expected: float, *, rtol=5e-6, atol=5e-8):
    if not np.isfinite(actual) or abs(actual-expected)>atol+rtol*abs(expected):
        raise AssertionError(f"{name}: {actual} versus {expected}")


def analyze(inputs: list[Path], source: Path, output: Path) -> None:
    source_hashes = load(source/"source-sha256.json")
    for name, expected in source_hashes.items():
        if sha(source/name) != expected:
            raise AssertionError("frozen source changed")
    all_rows, curves, p_rows, visits, depth_rows, receipts = [], [], [], [], [], []
    seen_datasets = set()
    for group in inputs:
        status, config, provenance = [load(group/name) for name in ("status.json", "study.json", "provenance.json")]
        if status["state"] != "COMPLETE" or status["completed"] != status["total"]:
            raise ValueError("experiment group is not terminal: "+str(group))
        if provenance["source_sha256"] != source_hashes:
            raise AssertionError("hosts did not use the same frozen source")
        for dataset in config["datasets"]:
            if dataset in seen_datasets:
                raise ValueError("duplicate dataset assignment")
            seen_datasets.add(dataset)
            rng_reference, chain_reference, global_reference, native_condition_reference = {}, {}, None, None
            initial_states, initial_buffers = {}, {}
            banks = group/"banks"/dataset
            with np.load(banks/"field.npz") as data:
                oracle = data["oracle"].astype(np.float64)
            with np.load(banks/"sampling.npz") as data:
                target, target2, directions = [data[key].astype(np.float64) for key in ("target", "target2", "directions")]
            for arm, setting in config["arms"].items():
                run = group/"runs"/dataset/arm
                native = load(run/"result.json")
                if native["step"] != 5000 or load(run/"status.json")["state"] != "COMPLETE":
                    raise ValueError("nonterminal training cell")
                with np.load(run/"diagnostics/field.npz") as data:
                    prediction = data["prediction"].astype(np.float64)
                if prediction.shape != oracle.shape:
                    raise AssertionError("field shape mismatch")
                primary = float(np.mean((prediction-oracle)**2))
                compare("field", primary, native["oracle_mse"])
                derivatives = []
                for time in (.1, .3, .5, .7, .9):
                    with np.load(banks/f"panel_{time:.1f}.npz") as data:
                        truth = data["oracle_jacobian"]
                    with np.load(run/"diagnostics"/f"panel_{time:.1f}.npz") as data:
                        actual = data["jacobian"]
                    if actual.shape != truth.shape or actual.shape[0] != 512:
                        raise AssertionError("derivative shape mismatch")
                    derivatives.append(float(np.mean(np.sum((actual-truth)**2, axis=(1, 2)))))
                jac_error = float(np.mean(derivatives))
                compare("jacobian", jac_error, native["jacobian_error"], rtol=1e-10, atol=1e-12)
                with np.load(run/"diagnostics/samples.npz") as data:
                    generated = data["generated"].astype(np.float64)
                if generated.shape != target.shape:
                    raise AssertionError("sample shape mismatch")
                sw2 = float(np.sqrt(np.mean((np.sort(generated@directions.T, axis=0)
                                            -np.sort(target@directions.T, axis=0))**2)))
                compare("SW2", sw2, native["sw2"], rtol=0., atol=2e-6)
                for step in (0, 1000, 2500, 5000):
                    path = run/f"step_{step:07d}.pt"
                    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                    if checkpoint["step"] != step or not all(torch.isfinite(v).all() for v in checkpoint["model"].values()):
                        raise AssertionError("invalid checkpoint")
                    rng = rng_reference.setdefault(step, checkpoint["data_rng"])
                    chain = chain_reference.setdefault(step, checkpoint["input_stream_digest"])
                    if not torch.equal(rng, checkpoint["data_rng"]) or chain != checkpoint["input_stream_digest"]:
                        raise AssertionError("same-host actual training streams differ")
                    if step == 0:
                        initial_states[arm], initial_buffers[arm] = checkpoint["model"], checkpoint["runtime_buffers"]
                        global_state = {k: v for k, v in checkpoint["model"].items() if not k.startswith("raw.blocks.")}
                        global_state.update({"buffer:"+k: v for k, v in checkpoint["runtime_buffers"].items()})
                        if global_reference is None:
                            global_reference = global_state
                        elif set(global_reference) != set(global_state) or any(not torch.equal(v, global_reference[k]) for k, v in global_state.items()):
                            raise AssertionError("global surrounding initialization differs")
                        if setting["mode"] == "native":
                            condition_state = {k: v for k, v in checkpoint["model"].items()
                                               if k.startswith("raw.blocks.") and ".ffn." not in k}
                            if native_condition_reference is None:
                                native_condition_reference = condition_state
                            elif any(not torch.equal(v, native_condition_reference[k]) for k, v in condition_state.items()):
                                raise AssertionError("native block surrounding initialization differs")
                    for layer, values in load(run/f"spectrum_{step:07d}.json").items():
                        singular = np.asarray(values["singular_values"])
                        p_rows.append(dict(dataset=dataset, arm=arm, step=step, layer=layer,
                            sigma_max=float(singular.max()), sigma_min=float(singular.min()),
                            sigma_mean=float(singular.mean()), stable_rank=float(np.sum(singular**2)/singular.max()**2),
                            eigen_real_max=max(values["eigenvalue_real"]),
                            eigen_imag_max=max(map(abs, values["eigenvalue_imag"]))))
                    captured = load(run/f"visits_{step:07d}.json")
                    expected_visits = 6 if setting["family"] == "ode_k2" else 0
                    if len(captured) != expected_visits:
                        raise AssertionError("missing per-visit activation records")
                    visits.extend(dict(dataset=dataset, arm=arm, step=step, **row) for row in captured)
                metrics = [json.loads(line) for line in (run/"metrics.jsonl").read_text().splitlines()]
                for item in metrics:
                    curves.append(dict(dataset=dataset, arm=arm, step=item["step"],
                        **{key: item.get(key) for key in ("train_loss", "grad_norm", "clip_fraction", "oracle_mse", "wall_seconds")}))
                all_rows.append(dict(dataset=dataset, arm=arm, **setting, primary_mse=primary,
                    jacobian_error=jac_error, sw2=sw2, parameters=native["parameters"],
                    dense_forward_macs=native["dense_forward_macs"],
                    concurrent_training_seconds=metrics[-1]["wall_seconds"], group=str(group)))
                depth_rows.extend(dict(dataset=dataset, arm=arm, **item) for item in native.get("depth_curve", []))
                receipts.append(dict(dataset=dataset, arm=arm, result_sha256=sha(run/"result.json"),
                                     final_checkpoint_sha256=sha(run/"step_0005000.pt")))
            # Final/deep loss arms have the same actual initialization.
            for arm in initial_states:
                if arm.endswith("_deep"):
                    paired = arm[:-5]+"_final"
                    if any(not torch.equal(v, initial_states[paired][k]) for k, v in initial_states[arm].items()):
                        raise AssertionError("loss arms start from different tensors")
            # Every cloned block starts from the same complete core as the shared model.
            for family in ("ode_k2", "swiglu192"):
                cloned = initial_states[family+"_cloned_final"]
                shared = initial_states[family+"_shared_final"]
                for key, value in shared.items():
                    if key.startswith("raw.blocks.0."):
                        suffix = key[len("raw.blocks.0."):]
                        for index in range(3):
                            if not torch.equal(value, cloned[f"raw.blocks.{index}."+suffix]):
                                raise AssertionError("cloned/shared initial cores differ")
    expected = {"gaussian_aniso_2d", "gmm_ring8_2d", "rotated_product_bimodal_16d"}
    if seen_datasets != expected or len(all_rows) != 42:
        raise ValueError("the complete registered three-task screen is required")
    output.mkdir(parents=True, exist_ok=False)
    by_dataset = {dataset: {r["arm"]: r for r in all_rows if r["dataset"] == dataset} for dataset in expected}
    candidates, extended_eligible = [], []
    nonlinear = expected-{"gaussian_aniso_2d"}
    ode_arms = [name for name in next(iter(by_dataset.values())) if name.startswith("ode_k2")]
    for arm in ode_arms:
        comparisons = {}
        for dataset, arms in by_dataset.items():
            generic = [row for row in arms.values() if row["family"].startswith("swiglu")]
            best = min(generic, key=lambda row: row["primary_mse"])
            dense = min((row for row in generic if row["mode"] != "shared"), key=lambda row: row["primary_mse"])
            own = arms[arm]
            comparisons[dataset] = dict(primary_ratio=own["primary_mse"]/best["primary_mse"],
                primary_baseline=best["arm"], mac_ratio=own["dense_forward_macs"]/best["dense_forward_macs"],
                jacobian_ratio=own["jacobian_error"]/min(row["jacobian_error"] for row in generic),
                sw2_ratio=own["sw2"]/min(row["sw2"] for row in generic),
                parameter_ratio_to_dense=own["parameters"]/dense["parameters"],
                shared_generic_ratio=own["primary_mse"]/arms["swiglu192_shared_"+own["supervision"]]["primary_mse"])
        guards = all(v["primary_ratio"] <= 1.05 for v in comparisons.values()) and all(
            comparisons[d]["jacobian_ratio"] <= 1.1 and comparisons[d]["sw2_ratio"] <= 1.1 for d in nonlinear)
        quality = guards and any(comparisons[d]["primary_ratio"] <= .9 and comparisons[d]["mac_ratio"] <= 1.05 for d in nonlinear)
        efficient = arm.endswith(("shared_final", "shared_deep")) and guards and all(
            v["parameter_ratio_to_dense"] <= .75 and v["shared_generic_ratio"] <= 1.05 for v in comparisons.values())
        eligible = "_shared_" in arm and all(comparisons[d]["primary_ratio"] <= 1.1 for d in nonlinear)
        if eligible:
            extended_eligible.append(arm)
        candidates.append(dict(arm=arm, quality_nominee=quality, parameter_efficiency_nominee=bool(efficient),
                               extended_sampling_eligible=eligible, comparisons=comparisons))
    csv_write(output/"comparison.csv", all_rows)
    csv_write(output/"training-curves.csv", curves)
    csv_write(output/"p-trajectories.csv", p_rows)
    csv_write(output/"depth-curves.csv", depth_rows)
    write(output/"visit-activations.json", visits)
    nominees = [r["arm"] for r in candidates if r["quality_nominee"] or r["parameter_efficiency_nominee"]]
    write(output/"decision.json", dict(nominees=nominees, extended_sampling_eligible=extended_eligible,
        candidates=candidates, additional_acceptance_control="shared SwiGLU128 required before accepting any nominee"))
    write(output/"audit.json", dict(status="PASS", cells=42, checkpoint_count=168,
        shared_initialization_checks="PASS", training_stream_checks="PASS", per_visit_captures="PASS",
        metric_recomputation="float64 array recomputation with declared float32 tolerances",
        source_sha256=source_hashes, receipts=receipts, analysis_source_sha256=sha(Path(__file__))))
    lines = ["# Recurrent-depth screen", "", "All42 registered cells are terminal. Metrics below are independently recomputed from saved arrays. No new test targets were used.", "",
             "| Dataset | Arm | Field MSE | Jacobian error | SW2 | Params | MACs |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in all_rows:
        lines.append(f"| {row['dataset']} | {row['arm']} | {row['primary_mse']:.6g} | {row['jacobian_error']:.6g} | {row['sw2']:.6g} | {row['parameters']} | {row['dense_forward_macs']} |")
    lines.extend(["", "Training-depth nominees: "+(", ".join(nominees) or "none"),
                  "Extended-generation eligibility: "+(", ".join(extended_eligible) or "none"), "",
                  "Sharing and intermediate supervision are distinct factors. Generic recurrence gains are not original-ODELayer gains. Recorded training seconds include concurrent evaluation and are not isolated latency measurements."])
    (output/"results.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(audit="PASS", cells=42, nominees=nominees,
                         extended_sampling_eligible=extended_eligible)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze([p.resolve() for p in args.inputs], args.source.resolve(), args.output.resolve())
