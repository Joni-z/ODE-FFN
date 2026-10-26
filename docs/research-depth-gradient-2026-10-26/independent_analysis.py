"""Recompute the frozen depth diagnostic with explicit NumPy identities.

No synthetic model/helper imports and no optimizer steps. Torch is used only to
read checkpoint provenance, not to compute the gradient decomposition.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


FIELDS = ("E0", "E1", "E2", "total", "retained", "linear_retained",
          "linear_share_of_lost", "cross_energy", "cosine_01", "cosine_02", "cosine_12")


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_table(path: Path, keys: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {tuple(row[k] for k in keys): row for row in rows}
    assert len(result) == len(rows), (path, "duplicate row keys")
    return result


def norm2(value: np.ndarray) -> float:
    return float(np.sum(value * value))


def ratio(a: float, b: float) -> float | None:
    return float(a / b) if b > 0 else None


def describe(g: np.ndarray) -> dict[str, float | None]:
    # Deliberately do not use a depth-basis matrix or the production helper.
    assert g.ndim == 2 and g.shape[0] == 3
    e0 = norm2(g[0] + g[1] + g[2]) / 3
    e1 = norm2(g[2] - g[0]) / 2
    e2 = norm2(g[0] - 2 * g[1] + g[2]) / 6
    total = norm2(g)
    assert np.isclose(e0 + e1 + e2, total, rtol=1e-11, atol=1e-14)
    dots = [float(np.sum(g[i] * g[j])) for i, j in ((0, 1), (0, 2), (1, 2))]
    cosines = [ratio(dot, np.sqrt(norm2(g[i]) * norm2(g[j])))
               for dot, (i, j) in zip(dots, ((0, 1), (0, 2), (1, 2)))]
    return dict(zip(FIELDS, (e0, e1, e2, total, ratio(e0, total),
                ratio(e0 + e1, total), ratio(e1, e1 + e2),
                2 * sum(dots), *cosines)))


def projection(g: np.ndarray, affine: bool) -> np.ndarray:
    result = np.repeat(g.mean(axis=0, keepdims=True), 3, axis=0)
    if affine:
        slope = (g[2] - g[0]) / 2
        result[0] -= slope
        result[2] += slope
    return result


def alignment(ga: np.ndarray, gb: np.ndarray, affine: bool) -> float | None:
    direction = projection(ga, affine)
    return ratio(float(np.sum(direction * gb)), np.sqrt(norm2(direction) * norm2(gb)))


def select(gp_a: np.ndarray, gp_b: np.ndarray, all_a: np.ndarray) -> dict[str, Any]:
    p, block = describe(gp_a), describe(all_a)
    share = ratio(p["E1"] + p["E2"], block["E1"] + block["E2"])
    common, affine = alignment(gp_a, gp_b, False), alignment(gp_a, gp_b, True)
    conditions = {
        "P_retained_at_most_half": p["retained"] is not None and p["retained"] <= 0.5,
        "P_at_least_half_blocked": share is not None and share >= 0.5,
        "linear_recovers_at_least_70_percent": p["linear_share_of_lost"] is not None and p["linear_share_of_lost"] >= 0.7,
        "affine_transfer_at_least_point2": affine is not None and affine >= 0.2,
        "transfer_gain_at_least_point05": common is not None and affine is not None and affine - common >= 0.05,
    }
    return dict(retained_P=p["retained"], P_share_of_blocked=share,
                linear_share_of_lost_P=p["linear_share_of_lost"],
                common_transfer=common, affine_transfer=affine,
                passes=all(conditions.values()), conditions=conditions)


def fingerprint(previous: str, target: np.ndarray, source: np.ndarray, time: np.ndarray) -> str:
    digest = hashlib.sha256(bytes.fromhex(previous))
    for tag, value in ((b"target", target), (b"source", source), (b"time", time)):
        digest.update(tag)
        digest.update(str((value.shape, str(value.dtype))).encode("ascii"))
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def run(runtime: Path, output: Path) -> dict[str, Any]:
    original = runtime / "output"
    assert load(original / "status.json")["state"] == "COMPLETE"
    output.mkdir(parents=True, exist_ok=False)
    bindings = load(original / "input-bindings.json")
    for name, value in bindings.items():
        assert sha(Path(name)) == value, name
    manifest = runtime / "source/source-sha256.json"
    for name, value in load(manifest).items():
        assert sha(runtime / "source" / name) == value, name
    assert sha(manifest) == load(original / "audit.json")["source_manifest_sha256"]
    frozen_outputs = {str(p.relative_to(runtime)): sha(p) for p in original.rglob("*") if p.is_file()}
    batch_rows = read_table(original / "batch-modes.csv", ("arm", "step", "bank", "batch", "group"))
    mean_rows = read_table(original / "mean-modes.csv", ("arm", "step", "bank", "group"))
    transfer_rows = read_table(original / "transfers.csv", ("arm", "step", "group"))
    selection_rows = read_table(original / "P-selection.csv", ("arm", "step"))
    identity_rows = read_table(original / "identities.csv", ("arm", "step", "bank", "batch"))
    finite_rows = read_table(original / "finite-directions.csv", ("arm", "mode"))
    counts = dict(batch_rows=0, mean_rows=0, transfer_rows=0, selections=0, gradient_sums=0, finite_directions=0)
    max_abs = 0.0
    late = []
    display = []

    def check(actual: float | None, value: str | float | None) -> None:
        nonlocal max_abs
        expected = None if value in ("", None) else float(value)
        if actual is None or expected is None:
            assert actual is expected, (actual, expected)
            return
        assert np.isclose(actual, expected, rtol=2e-9, atol=2e-13), (actual, expected)
        max_abs = max(max_abs, abs(actual - expected))

    for file in sorted((original / "gradients").glob("*/step-*.npz")):
        arm = file.parent.name
        step = file.stem.split("-")[1]
        family = arm.split("_shared_")[0]
        coordinates = load(original / (family + "-coordinates.json"))
        groups = coordinates["groups"]
        width = sum(coordinates["sizes"])
        disjoint = [v for k, v in groups.items() if k not in ("FFN", "all")]
        assert sorted(i for values in disjoint for i in values) == list(range(width))
        with np.load(file) as arrays:
            gradient = arrays["independent"]
            shared = arrays["shared"]
            loss = arrays["loss"]
        assert gradient.shape == (2, 8, 3, width)
        assert shared.shape == (2, 8, width)
        assert np.all(np.isfinite(gradient)) and gradient.dtype == np.float64
        np.testing.assert_allclose(shared, gradient.sum(axis=2), rtol=1e-7, atol=1e-9)
        ga, gb = gradient.mean(axis=1)
        for bank, label in enumerate("AB"):
            for batch in range(8):
                row = identity_rows[(arm, step, label, str(batch))]
                check(float(loss[bank, batch]), row["loss_shared"])
                check(float(loss[bank, batch]), row["loss_cloned"])
                error = float(np.max(np.abs(shared[bank, batch] - gradient[bank, batch].sum(axis=0))))
                check(error, row["gradient_sum_error"])
                assert float(row["common_gradient_error"]) <= 1e-9
                counts["gradient_sums"] += 1
        for group, index in groups.items():
            for bank, label in enumerate("AB"):
                for batch in range(8):
                    metrics = describe(gradient[bank, batch][:, index])
                    row = batch_rows[(arm, step, label, str(batch), group)]
                    for key in FIELDS:
                        check(metrics[key], row[key])
                    counts["batch_rows"] += 1
                metrics = describe((ga if bank == 0 else gb)[:, index])
                row = mean_rows[(arm, step, label, group)]
                for key in FIELDS:
                    check(metrics[key], row[key])
                counts["mean_rows"] += 1
                if group == ("P" if family == "ode_k2" else "FFN"):
                    display.append(dict(arm=arm, step=int(step), bank=label, group=group, **metrics))
            tr = transfer_rows[(arm, step, group)]
            check(alignment(ga[:, index], gb[:, index], False), tr["common_alignment"])
            check(alignment(ga[:, index], gb[:, index], True), tr["affine_alignment"])
            counts["transfer_rows"] += 1
        if family == "ode_k2":
            index = groups["P"]
            result = select(ga[:, index], gb[:, index], ga)
            passed = sum(select(g[:, index], gb[:, index], g)["passes"] for g in gradient[0])
            expected = selection_rows[(arm, step)]
            for key, value in result.items():
                if key not in ("passes", "conditions"):
                    check(value, expected[key])
            assert result["passes"] == (expected["passes"] == "True")
            assert passed == int(expected["bankA_batches_passing"])
            counts["selections"] += 1
            if arm == "ode_k2_shared_gradient" and int(step) in (2500, 5000):
                late.append(dict(arm=arm, step=int(step), bankA_batches_passing=passed, **result))
        if int(step) == 5000:
            index = groups["P"] if family == "ode_k2" else groups["FFN"]
            for label, affine in (("common", False), ("affine", True)):
                projected = projection(ga[:, index], affine)
                expected_derivative = -float(np.sum(gb[:, index] * projected)) / np.sqrt(norm2(projected))
                row = finite_rows[(arm, label)]
                check(expected_derivative, row["bankB_analytic"])
                error = abs(expected_derivative - float(row["bankB_finite"])) / max(1.0, abs(expected_derivative))
                assert error < 1e-6
                counts["finite_directions"] += 1
    assert counts == dict(batch_rows=len(batch_rows), mean_rows=len(mean_rows), transfer_rows=len(transfer_rows),
                          selections=len(selection_rows), gradient_sums=len(identity_rows), finite_directions=len(finite_rows))
    assert len(late) == 2
    selected = all(case["passes"] for case in late)
    assert selected == load(original / "decision.json")["eligible_for_separate_P_depth_protocol"]

    # Read metadata only, without restoring a predictor or re-running the helper.
    import torch
    digests: dict[int, str] = {}
    checkpoint_count = 0
    for name in bindings:
        if Path(name).suffix != ".pt":
            continue
        state = torch.load(name, map_location="cpu", weights_only=False)
        step = state["step"]
        if step in digests:
            assert digests[step] == state["input_stream_digest"]
        digests[step] = state["input_stream_digest"]
        checkpoint_count += 1
    input_batches = 0
    for step, previous in digests.items():
        with np.load(original / "banks" / f"step-{step}.npz") as bank:
            x = (1 - bank["time"]) * bank["source"] + bank["time"] * bank["target"]
            np.testing.assert_array_equal(x, bank["state"])
            np.testing.assert_array_equal(bank["target"] - bank["source"], bank["velocity"])
            expected = load(original / "banks" / f"step-{step}-hashes.json")
            for i in range(16):
                previous = fingerprint(previous, bank["target"][i], bank["source"][i], bank["time"][i])
                assert previous == expected[i]
                input_batches += 1
    for name, value in frozen_outputs.items():
        assert sha(runtime / name) == value, name
    result = dict(status="PASS", counts=counts, checkpoints_bound=checkpoint_count,
                  input_batches_verified=input_batches, source_files=len(load(manifest)), input_bindings=len(bindings),
                  maximum_csv_absolute_difference=max_abs, eligible_for_P_depth=selected, late_P_cases=late,
                  scope="Independent NumPy gradient algebra, stream hashes, interpolation and provenance. Saved finite-difference values checked against independently reconstructed analytic directions; no new predictor evaluation.",
                  frozen_output_sha256=frozen_outputs, analysis_source_sha256=sha(Path(__file__)))
    (output / "independent-audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output / "display-summary.json").write_text(json.dumps(display, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.runtime.resolve(), args.output.resolve())
    print(json.dumps({k: v for k, v in result.items() if k != "frozen_output_sha256"}, indent=2))
