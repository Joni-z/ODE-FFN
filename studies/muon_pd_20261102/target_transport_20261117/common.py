"""Resolve the existing pinned builder, native sampler, restore and asset code."""
import copy
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
COEFFICIENT = HERE.parent/'coefficient_20261116'
sys.path.insert(0, str(COEFFICIENT))
from cross_solver_evaluate import base, ev, assets, substitute_solver, torch  # noqa:E402


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def dump(path: Path, value: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(path).with_suffix(Path(path).suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def restored_pair(case: dict, data_root: str) -> tuple:
    folder = Path(case['folder'])
    cfg = read(folder/'config.json')
    assert cfg['steps'] == 20000 and cfg['seeds'] == [0]
    assert cfg['cell']['id'] == case['source_id']
    assert (folder/'ADAPTER_COMPLETE').exists()
    cache = base.verify_data(data_root, cfg['cell'])
    manifest = cache/'manifest.json'
    law = base.train.load_distribution(read(manifest)['datasets'][case['dataset']])
    model, identity = ev.restore(folder, cfg, manifest, 20000, 'cpu')
    fixed = base.train.fixed_eval_data(cache/case['dataset'], law, law.dim, 0,
                                     torch.device('cpu'), 4096, 256, 'test')
    actual = base.train.generated_samples(model, fixed['sample_source'], 32).numpy()
    import numpy as np
    with np.load(folder/'final_test_examples.npz') as saved:
        assert np.array_equal(actual, saved['generated'])
    crossed = copy.deepcopy(model)
    substitute_solver(crossed, case['inference_solver'])
    return model, crossed, dict(native_anchor_exact=True, **identity)
