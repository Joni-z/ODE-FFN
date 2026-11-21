"""Bind the new checkpoints and both held-out streams to the original midpoint results."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[3]
DOC = REPO / "docs/report4/reduced-state"
EQUAL = REPO / "docs/report4/equal-compute"


def read(file):
    return json.loads(file.read_text())


def sha(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def verified(folder, identity, total):
    assert read(folder/"COMPLETE.json")["passed"]
    assert (Path(identity)/"ADAPTER_COMPLETE").exists()
    row, = [r for r in read(folder/"summary.json") if r["step"] == total]
    binding = read(folder/"binding.json")
    cfg = binding["source_config"]
    assert Path(binding["source_folder"]) == Path(identity) and cfg["seeds"] == [0]
    assert not read(folder/"COMPLETE.json")["witness"]
    assert row["checkpoint_sha256"] == sha(Path(identity)/"checkpoints"/f"step_{total:07d}.pt")
    assert row["config_sha256"] == sha(Path(identity)/"config.json")
    assert len(row["bank_sw2"]) == 8
    assert (binding["heun_steps"], binding["banks"], binding["samples_per_bank"]) == (128, 8, 4096)
    result = dict(split=binding["split"], assets_sha256=binding["assets"]["sha256"],
                  summary_sha256=sha(folder/"summary.json"), stream=binding["assets"]["stream"])
    keys = ("step","field_mse","pooled_sw2","bank_sw2","sector_l1","gap_mass",
            "gap_error_max","checkpoint_sha256","manifest_sha256")
    return {**result, **{k:row[k] for k in keys if k in row}}


def collect(work):
    refs = read(EQUAL/"results.json")["rows"]
    rows = []
    for case in read(DOC/"budget.json"):
        case_id, ds = case["id"], case["dataset"]
        frozen_ref, = [r for r in refs if r["dataset"] == ds and r["family"] == "state_midpoint4"]
        end = read(work/"matched/trained-endpoints"/(case_id+".json"))
        source = work/"train/report4_reduced_state"/ds/case_id/"seed000"
        computed = {}
        for split in ("selection", "confirmation"):
            ref_folder, = (work.parent/"coefficient-20261116/state-analysis"/split).glob(f"{ds}__state_midpoint4__*")
            prior = verified(ref_folder, read(ref_folder/"binding.json")["source_folder"], 20000)
            assert prior["checkpoint_sha256"] == frozen_ref["checkpoint_sha256"]
            at20 = verified(work/split/case_id, source, 20000)
            matched = verified(work/"matched"/split/case_id, end["folder"], case["total_steps"])
            assert at20["checkpoint_sha256"] == end["parent_sha256"]
            assert matched["checkpoint_sha256"] == end["checkpoint_sha256"]
            assert len({row["assets_sha256"] for row in (prior, at20, matched)}) == 1
            if split == "confirmation":
                assert prior["assets_sha256"] == frozen_ref["assets_sha256"]
                assert prior["pooled_sw2"] == frozen_ref["pooled_sw2"]
            for row in (at20, matched):
                delta = [a-b for a,b in zip(row["bank_sw2"],prior["bank_sw2"])]
                row.update(delta_vs_midpoint= row["pooled_sw2"]-prior["pooled_sw2"],
                           banks_better_than_midpoint=sum(d<0 for d in delta),
                           mean_bank_delta_vs_midpoint=mean(delta))
            computed[split] = dict(reference=prior, at_20k=at20, matched=matched)
        rows.append(dict(dataset=ds, family=case["family"], step_flops=case["step_flops"],
                         midpoint_budget_flops=case["target_flops"], matched_steps=case["total_steps"],
                         matched_flops=case["actual_flops"], scores=computed))
    return dict(seed=0, outer_heun_steps=128, banks=8, samples_per_bank=4096, rows=rows)


if __name__ == "__main__":
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("work", type=Path)
    work = arg.parse_args().work
    result = collect(work)
    (DOC/"results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    for row in result["rows"]:
        print(row["dataset"], row["family"], row["matched_steps"],
              *[f'{row["scores"][s][k]["pooled_sw2"]:.5f}' for s in ("selection","confirmation")
                for k in ("at_20k","matched")])
