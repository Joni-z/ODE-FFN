"""Paired probe diagnostic: fixed-equation residual versus prediction error."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_anchored_corrector as trainer
from synthetic.train import restore_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    group, output = args.group.resolve(), args.output.resolve()
    dataset = "rotated_product_bimodal_16d"
    config = json.loads((group/"study.json").read_text())
    _, _, distribution = trainer.base.load_data(config, dataset)
    banks = group/"banks"/dataset
    with np.load(banks/"probe.npz") as data:
        x, t = [data[key][:256] for key in ("x", "t")]
    assert len(x)==256
    truth = distribution.velocity(x, t)
    rows, summary, checkpoint_hashes = [], [], {}
    for arm, setting in trainer.ARMS.items():
        run = group/"runs"/dataset/arm
        checkpoint = run/"step_0005000.pt"
        model = trainer.build(16, setting["family"], setting["mode"])
        restore_model(model, torch.load(checkpoint, map_location="cpu", weights_only=False))
        with torch.no_grad():
            states = model.hidden_states(torch.from_numpy(x), torch.from_numpy(t), updates=13)
            errors, residuals = {}, {}
            for count in (3, 6, 12):
                prediction = model.raw.output(model.raw.norm(states[count])).numpy()
                errors[count] = np.mean((prediction-truth)**2, axis=1)
                residuals[count] = (states[count+1]-states[count]).double().square().mean(dim=1).sqrt().numpy()
                for index in range(len(x)):
                    rows.append(dict(arm=arm, iteration=count, input_index=index,
                        field_squared_error=float(errors[count][index]),
                        hidden_next_change_rms=float(residuals[count][index])))
                summary.append(dict(arm=arm, iteration=count, field_mse=float(errors[count].mean()),
                    hidden_next_change_rms=float(np.sqrt(np.mean(residuals[count]**2)))))
        fraction = float(np.mean((residuals[12]<residuals[3]) & (errors[12]>errors[3])))
        checkpoint_hashes[arm] = dict(sha256=trainer.base.sha256(checkpoint),
            fraction_residual_down_and_error_up=fraction,
            residual_name="fixed-point residual" if setting["mode"]=="anchored" else "next-state change")
    output.mkdir(parents=True, exist_ok=False)
    for name, values in (("per-input.csv", rows), ("summary.csv", summary)):
        with (output/name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    trainer.base.write(output/"provenance.json", dict(status="PASS", inputs=256,
        protocol_sha256=trainer.base.sha256(ROOT/trainer.DOC/"fixed-point-diagnostic.md"),
        input_sha256=trainer.base.sha256(banks/"probe.npz"), checkpoints=checkpoint_hashes,
        script_sha256=trainer.base.sha256(Path(__file__)), architecture_nomination=False))
    print(json.dumps(dict(rows=len(rows), checkpoint_fractions=checkpoint_hashes)))


if __name__ == "__main__":
    main()
