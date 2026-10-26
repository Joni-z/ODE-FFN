"""Frozen raw-affine error diagnosis; no neural-network training or evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import psutil


TIMES = (0.1, 0.3, 0.5, 0.7, 0.9)


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def mse(a: np.ndarray) -> float:
    return float(np.mean(a * a))


def design(x: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones(len(x)), x))


def solve_svd(q: np.ndarray, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u, s, vt = np.linalg.svd(q, full_matrices=False)
    assert s[-1] > np.finfo(np.float64).eps * max(q.shape) * s[0]
    coefficient = vt.T @ ((u.T @ e) / s[:, None])
    return q @ coefficient, coefficient


def cross_fit(q: np.ndarray, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert len(q) % 2 == 0
    half = len(q) // 2
    predicted = np.empty_like(e)
    coefficients = []
    for train, test in ((slice(0, half), slice(half, None)),
                        (slice(half, None), slice(0, half))):
        _, coefficient = solve_svd(q[train], e[train])
        predicted[test] = q[test] @ coefficient
        coefficients.append(coefficient)
    return predicted, np.stack(coefficients)


def self_check() -> dict[str, Any]:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 3)); q = design(x)
    coefficient = rng.normal(size=(4, 3))
    affine = q @ coefficient
    arbitrary = rng.normal(size=(64, 3))
    residual = arbitrary - solve_svd(q, arbitrary)[0]
    predicted, recovered = solve_svd(q, affine + residual)
    np.testing.assert_allclose(predicted, affine, atol=1e-13, rtol=1e-12)
    np.testing.assert_allclose(recovered, coefficient, atol=1e-13, rtol=1e-12)
    oof, _ = cross_fit(q, affine)
    np.testing.assert_allclose(oof, affine, atol=1e-13, rtol=1e-12)
    # A coefficient for the first test half depends only on the other half.
    changed = affine.copy(); changed[32:] += 100
    _, changed_coefficients = cross_fit(q, changed)
    np.testing.assert_array_equal(changed_coefficients[0], cross_fit(q, affine)[1][0])
    return dict(status="PASS", known_affine_plus_orthogonal_residual=True,
                exact_out_of_fold_affine_recovery=True, test_labels_excluded_from_fit=True)


def main(raw: Path, output: Path) -> None:
    assert read(raw / "status.json")["state"] == "COMPLETE"
    source = Path(__file__).resolve().parent
    manifest = read(source / "source-sha256.json")
    for name, digest in manifest.items():
        assert sha(source / name) == digest
    config = read(raw / "study.json")
    assert len(config["datasets"]) == 3 and len(config["arms"]) == 7
    assert config["seed"] == 0 and config["steps"] == 5000
    provenance = read(raw / "provenance.json")
    old_source = Path(provenance["source_root"])
    for name, digest in provenance["source_sha256"].items():
        assert sha(old_source / name) == digest
    output.mkdir(parents=True, exist_ok=False)
    (output / "arrays").mkdir()
    process = psutil.Process(); start = time.monotonic()
    inputs: dict[str, str] = {}
    for path in (raw / "study.json", raw / "provenance.json", old_source / "source-sha256.json"):
        inputs[str(path)] = sha(path)
    rows = []; telemetry = []; maxima = dict(energy=0.0, qr_projection=0.0, qr_crossfit=0.0, coordinate_invariance=0.0)
    rotation_rng = np.random.default_rng(0)
    for dataset in config["datasets"]:
        for arm, setting in config["arms"].items():
            folder = raw / "runs" / dataset / arm
            assert read(folder / "status.json")["state"] == "COMPLETE"
            result = read(folder / "result.json")
            for name in ("result.json", "step_0005000.pt"):
                path = folder / name; inputs[str(path)] = sha(path)
            assert result["step"] == 5000 and result["arm"] == arm
            for clock in TIMES:
                bank_path = raw / "banks" / dataset / f"panel_{clock:.1f}.npz"
                pred_path = folder / "diagnostics" / f"panel_{clock:.1f}.npz"
                for path in (bank_path, pred_path): inputs[str(path)] = sha(path)
                with np.load(bank_path) as bank, np.load(pred_path) as pred:
                    x = bank["x"].astype(np.float64)
                    error = pred["prediction"].astype(np.float64) - bank["oracle"]
                    assert len(x) == 512 and np.allclose(bank["t"], clock)
                q = design(x)
                affine, coefficient = solve_svd(q, error)
                nonlinear = error - affine
                oof, fold_coefficients = cross_fit(q, error)
                # Distinct QR regression implementation on every real case.
                qq, rr = np.linalg.qr(q, mode="reduced")
                qr_coeff = np.linalg.solve(rr, qq.T @ error)
                qr_affine = q @ qr_coeff
                qr_oof = np.empty_like(error)
                for train, test in ((slice(0, 256), slice(256, None)), (slice(256, None), slice(0, 256))):
                    uq, rq = np.linalg.qr(q[train], mode="reduced")
                    qr_oof[test] = q[test] @ np.linalg.solve(rq, uq.T @ error[train])
                np.testing.assert_allclose(qr_affine, affine, rtol=1e-9, atol=2e-11)
                np.testing.assert_allclose(qr_oof, oof, rtol=1e-9, atol=2e-11)
                np.testing.assert_allclose(q.T @ nonlinear, 0, atol=1e-8, rtol=0)
                rotation, _ = np.linalg.qr(rotation_rng.normal(size=(x.shape[1], x.shape[1])))
                translated = x @ rotation + np.linspace(-0.2, 0.2, x.shape[1])
                transformed = solve_svd(design(translated), error)[0]
                np.testing.assert_allclose(transformed, affine, rtol=1e-9, atol=2e-11)
                total, linear_mse, nonlinear_mse, corrected = mse(error), mse(affine), mse(nonlinear), mse(error - oof)
                energy_error = abs(total - linear_mse - nonlinear_mse)
                assert energy_error < 1e-11 * max(1.0, total)
                recorded = next(p["field_mse"] for p in result["panels"] if abs(p["time"] - clock) < 1e-6)
                assert np.isclose(total, recorded, rtol=1e-10, atol=1e-12)
                for key, value in (("energy", energy_error), ("qr_projection", np.max(np.abs(qr_affine - affine))),
                                   ("qr_crossfit", np.max(np.abs(qr_oof - oof))),
                                   ("coordinate_invariance", np.max(np.abs(transformed - affine)))):
                    maxima[key] = max(maxima[key], float(value))
                row = dict(dataset=dataset, arm=arm, family=setting["family"], readout=setting["readout"], time=clock,
                    total=total, affine=linear_mse, nonlinear=nonlinear_mse,
                    crossfit_corrected=corrected, crossfit_reduction=total-corrected,
                    affine_fraction=linear_mse/total if total>0 else None,
                    crossfit_reduction_fraction=(total-corrected)/total if total>0 else None,
                    design_condition=float(np.linalg.cond(q)))
                rows.append(row)
                np.savez_compressed(output / "arrays" / f"{dataset}-{arm}-{clock:.1f}.npz",
                    error=error, affine=affine, nonlinear=nonlinear, coefficient=coefficient,
                    out_of_fold_affine=oof, fold_coefficients=fold_coefficients)
            rec = dict(dataset=dataset, arm=arm, elapsed_seconds=time.monotonic()-start,
                       rss_MiB=process.memory_info().rss/2**20, available_GiB=psutil.virtual_memory().available/2**30)
            telemetry.append(rec)
            with (output / "telemetry.jsonl").open("a") as handle: handle.write(json.dumps(rec)+"\n")
            write(output / "status.json", dict(state="RUNNING", completed=len(rows), total=105, **rec))
    aggregate = []
    for dataset in config["datasets"]:
        for arm in config["arms"]:
            matched = [r for r in rows if r["dataset"] == dataset and r["arm"] == arm]
            assert len(matched) == 5
            aggregate.append(dict(dataset=dataset, arm=arm, **{key:float(np.mean([r[key] for r in matched]))
                for key in ("total", "affine", "nonlinear", "crossfit_corrected", "crossfit_reduction")}))
    gaps = []
    amap = {(r["dataset"], r["arm"]):r for r in aggregate}
    for dataset in config["datasets"]:
        for mode in ("direct", "gradient"):
            parent = amap[(dataset, "ode_k2_"+mode)]
            for ref in ("swiglu128_"+mode, "swiglu192_"+mode, "swiglu384_direct"):
                baseline = amap[(dataset, ref)]
                delta = {key:parent[key]-baseline[key] for key in ("total", "affine", "nonlinear", "crossfit_reduction")}
                share = delta["affine"] / delta["total"] if delta["total"] > 0 else None
                transfer_share = delta["crossfit_reduction"] / delta["total"] if delta["total"] > 0 else None
                gaps.append(dict(dataset=dataset, parent="ode_k2_"+mode, reference=ref, **delta,
                    affine_excess_share=share, crossfit_excess_share=transfer_share,
                    majority_affine=share is not None and transfer_share is not None and share>=0.5 and transfer_share>=0.5))
    primary = [r for r in gaps if r["dataset"]=="rotated_product_bimodal_16d" and r["parent"]=="ode_k2_direct"]
    assert len(primary)==3
    survives = all(r["majority_affine"] for r in primary)
    csv_write(output/"panels.csv", rows); csv_write(output/"aggregates.csv", aggregate); csv_write(output/"gaps.csv", gaps)
    write(output/"decision.json", dict(mostly_affine_explanation_survives=survives,primary_comparisons=primary,
        architecture_selected=False,action="localize only; no training selected"))
    for name,digest in inputs.items(): assert sha(Path(name))==digest
    for name,digest in manifest.items(): assert sha(source/name)==digest
    write(output/"input-bindings.json", inputs)
    audit=dict(status="PASS",panels=len(rows),models=len(aggregate),gaps=len(gaps),input_bindings=len(inputs),
        maximum_errors=maxima,source_manifest_sha256=sha(source/"source-sha256.json"),
        peak_sampled_RSS_MiB=max(r["rss_MiB"] for r in telemetry),elapsed_seconds=time.monotonic()-start)
    write(output/"audit.json",audit);write(output/"status.json",dict(state="COMPLETE",**audit))
    print(json.dumps(dict(audit=audit,decision=read(output/"decision.json")),indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check",action="store_true")
    parser.add_argument("--raw",type=Path);parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check()))
    else:
        if args.raw is None or args.output is None: parser.error("--raw and --output required")
        main(args.raw.resolve(),args.output.resolve())
