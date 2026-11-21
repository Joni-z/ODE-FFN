"""Verify endpoint evidence and collect the paired, equal-budget comparison."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DOC = REPO / "docs/report4/equal-compute"
PRIOR = REPO / "docs/report4/tied-residual"


def read(file):
    return json.loads(file.read_text())


def sha(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def scores(row):
    assert len(row["bank_sw2"]) == 8
    assert all(math.isfinite(x) for x in [row["field_mse"], row["pooled_sw2"], *row["bank_sw2"]])
    return {key: row[key] for key in ("field_mse", "pooled_sw2", "bank_sw2")}


def collect(work):
    budget = read(DOC / "budget.json")
    prior = read(PRIOR / "trained-results.json")
    assert prior["steps"] == budget["reference_steps"] == 20000
    assert prior["seed"] == budget["seed"] == 0
    assert (prior["stream"], prior["banks"], prior["samples_per_bank"], prior["heun_steps"]) == (2026111800, 8, 4096, 128)
    progress = read(work / "progress.json")["rows"]
    results = []
    for plan in budget["rows"]:
        ds, family, key = plan["dataset"], plan["family"], plan["id"]
        old, = [r for r in prior["models"] if r["dataset"] == ds and r["family"] == family]
        assert old["checkpoint_sha256"] == plan["checkpoint_sha256"]
        rec = dict(dataset=ds, family=family, optimizer=plan["optimizer"], steps=plan["total_steps"],
                   training_flops=plan["actual_flops"], midpoint_budget_flops=plan["target_flops"],
                   original_20k_sw2=old["pooled_sw2"], assets_sha256=old["assets_sha256"])
        assert abs(plan["actual_flops"] - plan["target_flops"]) <= plan["step_flops"]/2
        if family == "state_midpoint4":
            results.append(dict(**rec, checkpoint_sha256=old["checkpoint_sha256"], **scores(old)))
            short = read(PRIOR / (ds + "-midpoint2.json"))
            assert short["checkpoint_sha256"] == old["checkpoint_sha256"]
            assert short["assets_sha256"] == old["assets_sha256"]
            results.append(dict(rec, family="midpoint4_train_midpoint2_sample",
                                checkpoint_sha256=old["checkpoint_sha256"], original_20k_sw2=short["pooled_sw2"], **scores(short)))
            continue
        assert progress[key]["phase"] == "complete"
        end = read(work / "trained-endpoints" / (key + ".json"))
        assert end["parent_sha256"] == old["checkpoint_sha256"]
        assert end["target"] == plan["total_steps"] and sha(Path(end["endpoint"])) == end["checkpoint_sha256"]
        folder = Path(progress[key]["output"])
        done, binding = read(folder/"COMPLETE.json"), read(folder/"binding.json")
        assert done["passed"] and not done["witness"] and done["steps"] == [plan["total_steps"]]
        assert binding["source_config"]["seeds"] == [0] and binding["source_config"]["cell"]["id"] == key
        assert binding["assets"]["sha256"] == old["assets_sha256"] and binding["assets"]["stream"] == prior["stream"]
        assert binding["split"] == "confirmation" and (binding["banks"], binding["samples_per_bank"], binding["heun_steps"]) == (8, 4096, 128)
        row, = read(folder / "summary.json")
        assert row["step"] == plan["total_steps"] and row["checkpoint_sha256"] == end["checkpoint_sha256"]
        assert row["manifest_sha256"] == plan["manifest_sha256"] and row["config_sha256"] == plan["config_sha256"]
        results.append(dict(**rec, checkpoint_sha256=row["checkpoint_sha256"], summary_sha256=sha(folder/"summary.json"),
                            parent_sha256=end["parent_sha256"], **scores(row)))
    assert len(results) == 8
    for row in results:
        midpoint, = [r for r in results if r["dataset"] == row["dataset"] and r["family"] == "state_midpoint4"]
        assert row["assets_sha256"] == midpoint["assets_sha256"]
        delta = [a-b for a,b in zip(row["bank_sw2"], midpoint["bank_sw2"])]
        row.update(pooled_delta_vs_native=row["pooled_sw2"]-midpoint["pooled_sw2"],
                   lower_banks_vs_native=sum(d < 0 for d in delta), mean_bank_delta_vs_native=statistics.mean(delta))
    return dict(seed=0, budget_sha256=sha(DOC/"budget.json"), prior_sha256=sha(PRIOR/"trained-results.json"),
                stream=prior["stream"], banks=8, samples_per_bank=4096, projections=256, outer_heun_steps=128, rows=results)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    a = p.parse_args()
    result = collect(a.work)
    (DOC / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for r in result["rows"]:
        print(r["dataset"], r["family"], r["steps"], f'{r["pooled_sw2"]:.5f}', r["lower_banks_vs_native"])
