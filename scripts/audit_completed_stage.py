"""Rehash all selected checkpoints and actual remote data; retain manifest discrepancy."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    args = parser.parse_args()
    root, report = args.runtime, args.report_root
    data = json.loads((report/"stage-summary.json").read_text())
    manifest = json.loads((report/"experiment-manifest.json").read_text())
    sources = json.loads((root/"source-manifest.json").read_text())
    assert data["complete"]
    for name, sha in sources["files"].items():
        assert digest(root/name) == sha, name
    data_root = root/"data/synthetic/v1"
    actual_data_sha = digest(data_root/"manifest.json")
    data_manifest = json.loads((data_root/"manifest.json").read_text())
    data_files = []
    for name, value in data_manifest["files"].items():
        if name.split("/")[0] in manifest["datasets"]:
            assert digest(data_root/name) == value["sha256"], name
            data_files.append(name)
    checkpoints = []
    for d in data["diagnostics"]:
        folder = Path(d["run_dir"])
        snap = folder/"checkpoints"/f"step_{d['step']:07d}.pt"
        assert digest(snap) == d["checkpoint_sha256"], str(snap)
        assert digest(folder/"config.json") == d["config_sha256"]
        assert digest(root/"scripts/run_stage_waves.py") == d["evaluation_script_sha256"]
        assert digest(root/"scripts/diagnose_synthetic.py") == d["helper_sha256"]
        assert d["seed"] == 0
        provenance = json.loads((folder/"provenance.json").read_text())
        assert provenance["dataset_manifest_sha256"] == actual_data_sha
        for name, sha in d["source_sha256"].items():
            assert digest(root/"synthetic"/name) == sha
        checkpoints.append({key:d[key] for key in ("dataset","label","step","checkpoint_sha256")})
    assert len(checkpoints) == 54 and len({(d["dataset"],d["label"],d["step"]) for d in checkpoints}) == 54
    assert len(data["progress"]) == 30
    for p in data["progress"]:
        folder = Path(p["run_dir"])
        assert p["complete"] and (folder/"COMPLETE").read_text().strip() == str(p["last_step"])
    for ds in manifest["datasets"]:
        values = json.loads((report/f"prefix-check-{ds}.json").read_text())
        assert all(v["model_max_abs_difference_at5000"] == 0 for v in values.values())
    result = dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  training_runs=30, diagnostic_checkpoints=54, frozen_source_files=len(sources["files"]),
                  all_hashes_valid=True, actual_data_files_checked=len(data_files),
                  recorded_data_manifest_sha256=sources["data_manifest_sha256"],
                  actual_data_manifest_sha256=actual_data_sha,
                  frozen_data_manifest_annotation_matches=sources["data_manifest_sha256"] == actual_data_sha,
                  annotation_note="Frozen snapshot used a local-data manifest hash; actual run provenance is authoritative. See data-content-audit.json for local/remote array comparison. Do not rewrite frozen history.",
                  baseline_prefix_max_abs_difference=0,
                  summary_sha256=digest(report/"stage-summary.json"), checkpoints=checkpoints)
    (report/"completion-audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="checkpoints"}))


if __name__ == "__main__":
    main()
