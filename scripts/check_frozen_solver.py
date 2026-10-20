"""Post hoc Ring8 solver-resolution check against an immutable frozen-order run."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    sys.path.insert(0, str(args.runtime.resolve()))
    import numpy as np
    import torch
    from scripts.run_frozen_order import (
        digest, load_model, read, telemetry, verify, weights_digest,
    )
    from scripts.monitor_stage import now, write_json
    from synthetic.interventions import evaluation_order
    from synthetic.train import generated_samples, sliced_w2

    manifest_path = args.runtime / "manifest.json"
    manifest = read(manifest_path)
    verify(manifest)
    previous = read(args.previous / "summary.json")
    assert previous["complete"] and len(previous["rows"]) == 66
    old = {(r["label"], str(r["eval_order"])): r for r in previous["rows"]
           if r["dataset"] == "gmm_ring8_2d"}
    args.output.mkdir(parents=True, exist_ok=True)
    started = now()
    with (args.output / "launch.json").open("x") as handle:
        json.dump(dict(
            started_at=started, pid=os.getpid(), seed=0, new_training=False,
            script_sha256=digest(__file__), manifest_sha256=digest(manifest_path),
            previous_summary_sha256=digest(args.previous / "summary.json"),
            steps=[32, 64, 128, 256],
            note="Post hoc solver check; fixed checkpoints, samples, directions and gate weights.",
        ), handle, indent=2)
    stop = threading.Event()
    monitor = threading.Thread(
        target=telemetry, args=(args.output / "gpu_usage.csv", stop), daemon=True)
    if args.device == "cuda":
        monitor.start()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    mode_map = {
        "gate_k1_input": [1, "exact"], "gate_k2_input": [2, "exact"],
        "order_k4_input": [4, 8, "exact"], "order_k8_input": [8, 4, "exact"],
        "silu_5000": ["native"], "swiglu_5000": ["native"],
    }
    try:
        for entry in manifest["checkpoints"]:
            if entry["dataset"] != "gmm_ring8_2d":
                continue
            model, fixed = load_model(entry, manifest, args.device)
            before = weights_digest(model)
            if not (args.output / "fixed.npz").exists():
                np.savez_compressed(args.output / "fixed.npz", **{
                    k: fixed[k].cpu().numpy()
                    for k in ["sample_source", "sample_target", "directions"]})
            native = {}
            for mode in mode_map[entry["label"]]:
                last = None
                for steps in [32, 64, 128, 256]:
                    tick = time.monotonic()
                    context = (nullcontext() if mode == "native"
                               else evaluation_order(model, mode, chunk=512))
                    with torch.no_grad(), context:
                        samples = generated_samples(model, fixed["sample_source"], steps)
                        assert torch.isfinite(samples).all()
                        sw2 = sliced_w2(samples, fixed["sample_target"], fixed["directions"])
                    if mode == mode_map[entry["label"]][0]:
                        native[steps] = samples.clone()
                    row = dict(
                        dataset=entry["dataset"], label=entry["label"],
                        trained_order=entry["orders"], eval_order=mode,
                        heun_steps=steps, velocity_evaluations=2*steps, sw2=sw2,
                        checkpoint_sha256=entry["checkpoint_sha256"],
                        endpoint_rms_from_native=(samples-native[steps]).square().mean().sqrt().item(),
                        endpoint_rms_from_previous_resolution=(
                            (samples-last).square().mean().sqrt().item() if last is not None else None),
                        elapsed_seconds=time.monotonic()-tick,
                    )
                    if steps == 32:
                        prior = old[entry["label"], str(mode)]
                        source = args.previous / prior["samples_relative_path"]
                        assert digest(source) == prior["samples_sha256"]
                        with np.load(source, allow_pickle=False) as data:
                            previous_samples = torch.from_numpy(data["generated"]).to(args.device)
                        row["previous_sw2_abs_difference"] = abs(sw2-prior["sw2"])
                        row["previous_endpoint_rms_difference"] = (
                            (samples-previous_samples).square().mean().sqrt().item())
                        assert row["previous_sw2_abs_difference"] < 5e-4
                        assert row["previous_endpoint_rms_difference"] < 5e-4
                    target = args.output / "samples" / f"{entry['label']}--{mode}--{steps}.npz"
                    target.parent.mkdir(exist_ok=True)
                    np.savez_compressed(target, generated=samples.cpu().numpy())
                    row["samples_relative_path"] = str(target.relative_to(args.output))
                    row["samples_sha256"] = digest(target)
                    rows.append(row)
                    write_json(args.output / "progress.json", dict(
                        checked_at=now(), complete=False, cells=len(rows), expected_cells=48, rows=rows))
                    print(json.dumps(row), flush=True)
                    last = samples
            assert before == weights_digest(model)
        verify(manifest)
        assert len(rows) == 48
        write_json(args.output / "summary.json", dict(
            complete=True, seed=0, new_training=False, started_at=started,
            completed_at=now(), cells=48, rows=rows,
            fixed_sha256=digest(args.output / "fixed.npz"),
            launch=read(args.output / "launch.json"),
        ))
    except BaseException as error:
        write_json(args.output / "failure.json", dict(time=now(), error=repr(error), cells=len(rows)))
        raise
    finally:
        stop.set()
        if monitor.is_alive():
            monitor.join(timeout=12)


if __name__ == "__main__":
    main()
