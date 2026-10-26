"""Audit the fixed six-arm Product16 screen and apply its registered stop rule."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_anchored_corrector as trainer
from synthetic.recurrent_depth import build as recurrent_build, compute_budget
from synthetic.train import restore_model, generated_samples


def load(path: Path):
    return json.loads(path.read_text())


def same_tensors(a: dict, b: dict) -> None:
    assert set(a)==set(b)
    assert all(torch.equal(a[k], b[k]) for k in a)


def recompute(run: Path, banks: Path, model) -> dict:
    with np.load(banks/"field.npz") as data:
        x, t, truth = [data[k] for k in ("x", "t", "oracle")]
    with torch.no_grad():
        prediction = np.concatenate([model(torch.from_numpy(x[start:start+1024]),
            torch.from_numpy(t[start:start+1024])).numpy() for start in range(0, len(x), 1024)])
    with np.load(run/"diagnostics/field.npz") as data:
        np.testing.assert_array_equal(prediction, data["prediction"])
    errors = np.mean((prediction.astype(np.float64)-truth.astype(np.float64))**2, axis=1)
    jacobians = []
    for value in (.1, .3, .5, .7, .9):
        with np.load(banks/f"panel_{value:.1f}.npz") as data:
            px, pt, oracle = [data[k] for k in ("x", "t", "oracle_jacobian")]
        _, derivative = trainer.base.jacobian(model, torch.from_numpy(px), torch.from_numpy(pt))
        actual = derivative.numpy()
        with np.load(run/"diagnostics"/f"panel_{value:.1f}.npz") as data:
            np.testing.assert_array_equal(actual, data["jacobian"])
        jacobians.append(float(np.mean(np.sum((actual-oracle)**2, axis=(1, 2)))))
    with np.load(banks/"sampling.npz") as data:
        source, target, directions = [data[k] for k in ("source", "target", "directions")]
    samples = generated_samples(model, torch.from_numpy(source), 32).numpy()
    with np.load(run/"diagnostics/samples.npz") as data:
        np.testing.assert_array_equal(samples, data["generated"])
    projected = samples.astype(np.float64)@directions.astype(np.float64).T
    expected = target.astype(np.float64)@directions.astype(np.float64).T
    sw2 = float(np.sqrt(np.mean((np.sort(projected, axis=0)-np.sort(expected, axis=0))**2)))
    native = load(run/"result.json")
    primary, jac = float(errors.mean()), float(np.mean(jacobians))
    np.testing.assert_allclose(primary, native["oracle_mse"], rtol=5e-6, atol=5e-8)
    np.testing.assert_allclose(jac, native["jacobian_error"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(sw2, native["sw2"], rtol=0, atol=2e-6)
    budget = compute_budget(model)
    assert budget["parameters"]==native["parameters"]
    assert budget["dense_forward_macs"]==native["dense_forward_macs"]
    return dict(primary_mse=primary, jacobian_error=jac, sw2=sw2,
                parameters=budget["parameters"], dense_forward_macs=budget["dense_forward_macs"])


def analyze(group: Path, prior: Path, prior_audit: Path, source: Path, output: Path) -> None:
    torch.set_num_threads(1)
    dataset = "rotated_product_bimodal_16d"
    config, status = load(group/"study.json"), load(group/"status.json")
    assert status["state"]=="COMPLETE" and status["completed"]==status["total"]==6
    assert config["arms"]==trainer.ARMS and config["datasets"]==[dataset]
    hashes = load(source/"source-sha256.json")
    assert load(group/"provenance.json")["source_sha256"]==hashes
    for name, expected in hashes.items():
        assert trainer.base.sha256(source/name)==expected
        assert trainer.base.sha256(ROOT/name)==expected
    old_audit = load(prior_audit)
    assert old_audit["status"]=="PASS" and old_audit["cells"]==42
    old_source = load(prior/"provenance.json")["source_sha256"]
    assert old_source==old_audit["source_sha256"]
    for name in ("synthetic/models.py", "synthetic/recurrent_depth.py", "synthetic/train.py"):
        assert hashes[name]==old_source[name]
    banks, old_banks = group/"banks"/dataset, prior/"banks"/dataset
    for name in ("field.npz", "probe.npz", "sampling.npz", *[f"panel_{v:.1f}.npz" for v in (.1,.3,.5,.7,.9)]):
        with np.load(banks/name) as a, np.load(old_banks/name) as b:
            assert a.files==b.files
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
    rows, initial, checkpoints, curves = [], {}, {}, []
    rng_reference, chain_reference = {}, {}
    for arm, setting in trainer.ARMS.items():
        run = group/"runs"/dataset/arm
        native = load(run/"result.json")
        assert native["step"]==5000 and load(run/"status.json")["state"]=="COMPLETE"
        assert all(native[k]==v for k,v in setting.items())
        for step in (0, 1000, 2500, 5000):
            checkpoint = torch.load(run/f"step_{step:07d}.pt", map_location="cpu", weights_only=False)
            assert checkpoint["step"]==step
            assert all(torch.isfinite(value).all() for value in checkpoint["model"].values())
            if step in rng_reference:
                assert torch.equal(rng_reference[step], checkpoint["data_rng"])
                assert chain_reference[step]==checkpoint["input_stream_digest"]
            else:
                rng_reference[step], chain_reference[step] = checkpoint["data_rng"], checkpoint["input_stream_digest"]
            assert len(load(run/f"visits_{step:07d}.json"))==(6 if setting["family"]=="ode_k2" else 0)
            checkpoints[arm,step] = checkpoint
        initial[arm] = checkpoints[arm,0]
        model = trainer.build(16, setting["family"], setting["mode"])
        restore_model(model, checkpoints[arm,5000])
        row = dict(arm=arm, family=setting["family"], mode=setting["mode"], origin="new six-arm screen",
                   **recompute(run, banks, model))
        rows.append(row)
        curves.extend(dict(arm=arm, **value) for value in native["iteration_curve"])
    for family in ("ode_k2", "swiglu192", "swiglu128"):
        a,b = [initial[f"{family}_{rule}_final"] for rule in ("accumulating", "anchored")]
        same_tensors(a["model"], b["model"])
        same_tensors(a["runtime_buffers"], b["runtime_buffers"])
    old_config = load(prior/"study.json")
    exact_controls = []
    for family in ("ode_k2", "swiglu192"):
        old = prior/"runs"/dataset/(family+"_shared_final")
        new = group/"runs"/dataset/(family+"_accumulating_final")
        assert load(old/"provenance.json")["manifest_sha256"]==load(new/"provenance.json")["manifest_sha256"]
        for step in (0,1000,2500,5000):
            a = torch.load(old/f"step_{step:07d}.pt", map_location="cpu", weights_only=False)
            b = checkpoints[family+"_accumulating_final",step]
            same_tensors(a["model"], b["model"])
            same_tensors(a["runtime_buffers"], b["runtime_buffers"])
            assert a["input_stream_digest"]==b["input_stream_digest"]
            assert torch.equal(a["data_rng"],b["data_rng"])
        exact_controls.append(family)
    # Replay every previous practical dense control used by the nomination frontier.
    for arm, setting in old_config["arms"].items():
        if setting["family"].startswith("swiglu") and setting["mode"]!="shared":
            run = prior/"runs"/dataset/arm
            checkpoint = torch.load(run/"step_0005000.pt", map_location="cpu", weights_only=False)
            model = recurrent_build(16, setting["family"], setting["mode"])
            restore_model(model, checkpoint)
            rows.append(dict(arm=arm, family=setting["family"], mode=setting["mode"],
                             origin="previous dense control", **recompute(run, old_banks, model)))
    generic = [r for r in rows if r["family"].startswith("swiglu")]
    candidate = next(r for r in rows if r["arm"]=="ode_k2_anchored_final")
    best = min(generic, key=lambda r:r["primary_mse"])
    active = next(r for r in rows if r["arm"]=="swiglu192_anchored_final")
    ratios = dict(primary=candidate["primary_mse"]/best["primary_mse"],
        jacobian=candidate["jacobian_error"]/min(r["jacobian_error"] for r in generic),
        sw2=candidate["sw2"]/min(r["sw2"] for r in generic),
        macs=candidate["dense_forward_macs"]/active["dense_forward_macs"])
    advance = ratios["primary"]<=.9 and ratios["jacobian"]<=1.1 and ratios["sw2"]<=1.1 and ratios["macs"]<=1.05
    output.mkdir(parents=True, exist_ok=False)
    for name, values in (("comparison.csv", rows), ("iteration-curves.csv", curves)):
        with (output/name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)
    trainer.base.write(output/"decision.json", dict(advance=bool(advance), candidate=candidate["arm"],
        primary_baseline=best["arm"], ratios=ratios, action="run Ring8/Gaussian six-arm groups" if advance else "stop this setting"))
    trainer.base.write(output/"audit.json", dict(status="PASS", new_cells=6, checkpoints=24,
        exact_previous_controls=exact_controls, replayed_previous_dense_controls=6,
        complete_field_jacobian_sampling_replay="bitwise array equality for all12 compared endpoints",
        same_initialization_and_streams="PASS", source_sha256=hashes,
        analysis_source_sha256=trainer.base.sha256(Path(__file__)),
        final_checkpoints={arm:trainer.base.sha256(group/"runs"/dataset/arm/"step_0005000.pt") for arm in trainer.ARMS}))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    short = [r["arm"].replace("_final", "").replace("_", " ") for r in rows[:6]]
    for ax, metric in zip(axes, ("primary_mse", "jacobian_error", "sw2")):
        ax.barh(short, [r[metric] for r in rows[:6]], color=["#D55E00" if r["family"]=="ode_k2" else "#0072B2" for r in rows[:6]])
        ax.axvline(min(r[metric] for r in generic), color="black", linestyle="--", linewidth=1)
        ax.set_title(metric)
        ax.invert_yaxis()
        if ax is not axes[0]:
            ax.set_yticklabels([])
    fig.suptitle("Product16: six fixed endpoints; dashed = combined generic frontier", fontsize=10)
    fig.tight_layout()
    fig.savefig(output/"comparison.png", dpi=160, bbox_inches="tight")
    fig.savefig(output/"comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    lines = ["# Anchored-corrector Phase A", "", "| Arm | Origin | Field MSE | Jacobian error | SW2 | Params | MACs |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['arm']} | {r['origin']} | {r['primary_mse']:.7f} | {r['jacobian_error']:.5f} | {r['sw2']:.5f} | {r['parameters']} | {r['dense_forward_macs']} |")
    lines.extend(["", "Decision: "+("advance" if advance else "stop this setting"), "",
                  "All six endpoints and all six reused dense controls were replayed from actual weights, including full field arrays, five Jacobian panels and Heun32 samples. The two repeated accumulated controls match their previous checkpoints bitwise at all four saved steps. Ratios are descriptive; no inference across training repetitions is made.", "",
                  "![Comparison](comparison.png)"])
    (output/"analysis-report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(audit="PASS", advance=bool(advance), ratios=ratios)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("group", "prior", "prior-audit", "source", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    analyze(args.group.resolve(), args.prior.resolve(), args.prior_audit.resolve(), args.source.resolve(), args.output.resolve())
