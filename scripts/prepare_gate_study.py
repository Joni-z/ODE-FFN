"""Write the predeclared gate study configs and resource-independent manifest."""
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_stage_waves import forward_macs
from synthetic.models import VelocityNet
from synthetic.train import sha256

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "configs/synthetic/gates_20261021"
KINDS = ("ode_original", "ode_fixed", "ode_scalar", "ffn", "swiglu")


def main() -> None:
    base = json.loads((ROOT / "configs/synthetic/train_swiglu_v1.json").read_text())
    FOLDER.mkdir(parents=True, exist_ok=True)
    entries = []
    for kind in KINDS:
        cfg = copy.deepcopy(base)
        cfg.update(experiment_id=f"gates_20261021_{kind}", models=[kind],
                   sampling_split="validation")
        path = FOLDER / f"{kind}.json"
        content = json.dumps(cfg, indent=2) + "\n"
        if path.exists() and path.read_text() != content:
            raise FileExistsError(path)
        path.write_text(content)
        entries.append(dict(model=kind, config=str(path.relative_to(ROOT)),
                            sha256=sha256(path),
                            parameters=VelocityNet(2, kind, **cfg["model_options"]).parameter_count(),
                            forward_macs=forward_macs(kind, 2, cfg["model_options"])))
    manifest = dict(datasets=base["datasets"], seed=0, entries=entries, runs=15,
                    primary_step=5000, selection_steps=[1000, 2000, 3000, 4000, 5000],
                    selection_rule="minimum validation SW2, earlier step breaks ties",
                    protocol_sha256=sha256(ROOT / "docs/research-2026-10-21-gates/protocol.md"))
    path = FOLDER / "manifest.json"
    content = json.dumps(manifest, indent=2) + "\n"
    if path.exists() and path.read_text() != content:
        raise FileExistsError(path)
    path.write_text(content)
    print(content)


if __name__ == "__main__":
    main()
