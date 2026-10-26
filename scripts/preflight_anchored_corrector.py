"""Native-host pipeline smoke and original shared-control checkpoint replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_anchored_corrector as runner
from synthetic.train import restore_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-v1", type=Path, required=True)
    parser.add_argument("--data-product", type=Path, required=True)
    parser.add_argument("--prior-product", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    hashes = json.loads((ROOT/"source-sha256.json").read_text())
    for name, expected in hashes.items():
        assert runner.base.sha256(ROOT/name)==expected
    dataset = "rotated_product_bimodal_16d"
    config = dict(seed=0, steps=3, workers=1, device="cpu", datasets=[dataset], arms=runner.ARMS,
                  data_roots={dataset: str(args.data_product.resolve())})
    runner.base.write(output/"study.json", config)
    runner.base.prepare_banks(config, dataset, output)
    receipt = []
    for arm in ("ode_k2_accumulating_final", "ode_k2_anchored_final", "swiglu192_anchored_final", "swiglu128_anchored_final"):
        runner.train_cell(output/"study.json", output, dataset, arm, steps=3)
        folder = output/"runs"/dataset/arm
        state = torch.load(folder/"step_0000003.pt", map_location="cpu", weights_only=False)
        setting = runner.ARMS[arm]
        model = runner.build(16, setting["family"], setting["mode"])
        restore_model(model, state)
        replay = runner.base.field_evaluation(model, output/"banks"/dataset)
        native = json.loads((folder/"result.json").read_text())
        assert replay["oracle_mse"]==native["oracle_mse"]
        assert len(json.loads((folder/"visits_0000003.json").read_text()))==(6 if setting["family"]=="ode_k2" else 0)
        receipt.append(dict(arm=arm, input_stream_digest=state["input_stream_digest"], replay="PASS"))
    assert len({r["input_stream_digest"] for r in receipt})==1
    previous = []
    if args.prior_product:
        prior = args.prior_product.resolve()
        for family in ("ode_k2", "swiglu192"):
            folder = prior/"runs"/dataset/(family+"_shared_final")
            state = torch.load(folder/"step_0005000.pt", map_location="cpu", weights_only=False)
            model = runner.build(16, family, "accumulating")
            restore_model(model, state)
            replay = runner.base.field_evaluation(model, prior/"banks"/dataset)
            native = json.loads((folder/"result.json").read_text())
            assert replay["oracle_mse"]==native["oracle_mse"]
            previous.append(dict(family=family, final_checkpoint_sha256=runner.base.sha256(folder/"step_0005000.pt"),
                                 primary_replay="PASS"))
    for name, expected in hashes.items():
        assert runner.base.sha256(ROOT/name)==expected
    runner.base.write(output/"PASS.json", dict(status="PASS", source_sha256=hashes,
        native_platform=sys.platform, torch=torch.__version__, smoke=receipt, previous_shared_controls=previous))
    print(json.dumps(dict(status="PASS", smoke=len(receipt), previous_replays=len(previous))))


if __name__ == "__main__":
    main()
