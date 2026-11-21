"""Collect candidate and reference endpoints on identical samples for Report 4."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(candidate_root, reference_root):
    results = []
    for candidate in read(Path(__file__).with_name("grid.json"))["cells"]:
        dataset = candidate["dataset"]
        case_id = candidate["id"]
        families = [(name, reference_root / "analysis" / "confirmation")
                    for name in ("free_isru", "swiglu64" if dataset == "gmm_ring8_2d" else "swiglu192")]
        families.extend([("state_midpoint4", reference_root / "state-analysis" / "confirmation"),
                         ("tied_euler8", candidate_root / "confirmation")])
        task_rows = []
        for family, parent in families:
            folder, = parent.glob(f"{dataset}__{family}__*")
            binding = read(folder / "binding.json")
            assert read(folder / "COMPLETE.json")["passed"]
            assert binding["split"] == "confirmation" and binding["assets"]["stream"] == 2026111800
            assert (binding["heun_steps"], binding["banks"], binding["samples_per_bank"]) == (128, 8, 4096)
            cfg = binding["source_config"]
            assert cfg["steps"] == 20000 and cfg["seeds"] == [0] and cfg["cell"]["dataset"] == dataset
            assert cfg["cell"]["family"] == family
            row, = [r for r in read(folder / "summary.json") if r["step"] == 20000]
            assert len(row["bank_sw2"]) == 8
            reference = Path(binding["source_folder"])
            if family == "tied_euler8":
                assert folder.name == case_id
            assert row["checkpoint_sha256"] == digest(reference / "checkpoints/step_0020000.pt")
            native = read(reference / "final_test_sampling.json")
            assert native["sampling_split"] == "test" and native["sample"]["heun_steps"] == 32
            task_rows.append(dict(dataset=dataset, family=family, native_test_sw2_heun32=native["sample"]["sliced_w2"],
                                  assets_sha256=binding["assets"]["sha256"], summary_sha256=digest(folder / "summary.json"),
                                  **{k: v for k, v in row.items() if k in
                                     ("pooled_sw2", "field_mse", "sector_l1", "gap_error_max", "bank_sw2",
                                      "checkpoint_sha256", "config_sha256", "manifest_sha256")}))
        assert len({r["assets_sha256"] for r in task_rows}) == 1
        assert len({r["manifest_sha256"] for r in task_rows}) == 1
        results.extend(task_rows)
    return dict(steps=20000, seed=0, heun_steps=128, stream=2026111800, banks=8,
                samples_per_bank=4096, metric="pooled sliced W2", models=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_root", type=Path)
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = collect(args.candidate_root, args.reference_root)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for row in result["models"]:
        print(row["dataset"], row["family"], format(row["pooled_sw2"], ".3f"))
