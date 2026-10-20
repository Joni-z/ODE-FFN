"""Evaluate the already-selected gate-study checkpoints at 8,192 samples."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.runtime))
    from synthetic.models import VelocityNet
    from synthetic.train import generated_samples, load_targets, sha256, sliced_w2
    from scripts.monitor_stage import now, write_json
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    assert json.loads((args.study / "status.json").read_text())["state"] == "COMPLETE"
    frozen = json.loads((args.study / "frozen-inputs.json").read_text())
    for path, digest in frozen["source_and_data_sha256"].items():
        assert sha256(args.runtime / path) == digest, path
    rows = json.loads((args.study / "summary.json").read_text())["results"]
    source = np.random.default_rng(4427).standard_normal((8192, 2)).astype(np.float32)
    source = torch.from_numpy(source)
    results = []
    started = time.monotonic()
    write_json(args.output / "status.json", dict(state="RUNNING", time=now(), cells=len(rows)))
    for row in rows:
        dataset, model_name, step = row["dataset"], row["model"], row["step"]
        folder = args.study / "runs" / f"gates_20261021_{model_name}" / dataset / model_name / "seed000"
        cfg = json.loads((folder / "config.json").read_text())
        checkpoint = folder / "checkpoints" / f"step_{step:07d}.pt"
        assert sha256(checkpoint) == row["checkpoint_sha256"]
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert state["step"] == step
        model = VelocityNet(2, model_name, **cfg["model_options"])
        model.load_state_dict(state["model"], strict=True)
        model.eval()
        original_path = args.study / "samples" / f"{dataset}-{model_name}-{step}.npz"
        assert sha256(original_path) == row["samples_sha256"]
        with np.load(original_path) as original:
            assert np.array_equal(source[:2048].numpy(), original["source"])
            directions = torch.from_numpy(original["directions"].copy())
            old_samples = torch.from_numpy(original["generated"].copy())
            old_target = torch.from_numpy(original["target"].copy())
        samples = generated_samples(model, source, 32)
        coordinate_difference = float((samples[:2048]-old_samples).abs().max())
        prefix_score = sliced_w2(samples[:2048], old_target, directions)
        assert coordinate_difference < 1e-4, coordinate_difference
        assert abs(prefix_score-row["test_sw2"]) < 1e-5
        folder_data = args.runtime / "data/synthetic/v1" / dataset
        validation = torch.from_numpy(load_targets(folder_data / "validation.npz", 2)[:8192])
        test = torch.from_numpy(load_targets(folder_data / "test.npz", 2)[:8192])
        assert len(validation) == len(test) == 8192
        result = dict(dataset=dataset, model=model_name, step=step,
                      primary=row["primary"], selected=row["selected"],
                      checkpoint_sha256=row["checkpoint_sha256"],
                      test_sw2_2048=row["test_sw2"],
                      test_sw2_8192=sliced_w2(samples, test, directions),
                      validation_target_sw2_8192=sliced_w2(samples, validation, directions),
                      target_reference_sw2_8192=sliced_w2(validation, test, directions),
                      native_prefix_coordinate_max_difference=coordinate_difference,
                      native_prefix_sw2_difference=abs(prefix_score-row["test_sw2"]))
        sample_path = args.output / f"{dataset}-{model_name}-{step}.npz"
        np.savez_compressed(sample_path, generated=samples.numpy())
        result["samples_sha256"] = sha256(sample_path)
        results.append(result)
        write_json(args.output / "summary.json",
                   dict(results=results, script_sha256=sha256(__file__),
                        original_summary_sha256=sha256(args.study / "summary.json"),
                        sample_count=8192, heun_steps=32, directions=128))
        write_json(args.output / "status.json",
                   dict(state="RUNNING", time=now(), complete=len(results), cells=len(rows)))
    write_json(args.output / "status.json",
               dict(state="COMPLETE", time=now(), complete=len(results), seconds=time.monotonic()-started))


if __name__ == "__main__":
    main()
