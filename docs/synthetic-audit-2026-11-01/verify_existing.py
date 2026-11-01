"""Read-only replay of existing synthetic evidence; never trains or saves weights.

Run with the preserved CPU environment and an explicit output JSON path.
Historical absolute paths are intentional: this verifies the actual old runs.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--workspace', type=Path, required=True)
parser.add_argument('--repository', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(1)
torch.set_grad_enabled(False)
base = args.workspace / 'architecture-search'
cubic = base / 'cubic_split_20261026'
joint = base / 'joint_dependence_20261026'
native = base / 'conservative_readout_20261026'
sys.path.insert(0, str(cubic / 'source'))
import cubic_study as study
sys.path.insert(0, str(joint / 'source'))
import joint_study
joint_study.setup(joint)

def load(path):
    return json.loads(path.read_text())

def sha(path):
    with path.open('rb') as handle:
        h = hashlib.sha256()
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-9)

def sw2(a, b, directions):
    # Independent NumPy implementation, using fixed saved projection directions.
    delta = np.sort(a @ directions.T, axis=0) - np.sort(b @ directions.T, axis=0)
    return float(np.sqrt(np.mean(delta.astype(np.float64)**2)))

receipt = {'scope': 'existing cubic and dependent-sign evidence; no optimizer updates',
           'device': 'cpu', 'torch': torch.__version__, 'bindings': {}, 'cells': [],
           'input_sha256': {}, 'report_rows': 0}
for root in (cubic, joint):
    ready = load(root / 'ready.json')
    checked = 0
    for category in ('bindings', 'prior_documents'):
        for name, digest in ready.get(category, {}).items():
            assert sha(Path(name)) == digest, name
            checked += 1
    receipt['bindings'][root.name] = checked
    receipt['input_sha256'][str(root / 'ready.json')] = sha(root / 'ready.json')

tasks = []
for dataset in study.legacy.DATASETS:
    for family in ('ode_k2', 'free_isru', 'swiglu64', 'swiglu128', 'swiglu160'):
        folder = study.reference(dataset, family) if family in study.REFERENCES else cubic / 'output/runs' / dataset / family
        tasks.append((dataset, family, folder, native / 'output/banks' / dataset, False))
for family in ('ode_k2', 'free_isru', 'swiglu64', 'swiglu128', 'swiglu160'):
    tasks.append(('sign_code', family, joint / 'output/runs' / family, joint / 'banks', True))

for dataset, family, folder, banks, is_joint in tasks:
    assert load(folder / 'status.json')['state'] == 'COMPLETE'
    with np.load(banks / 'field.npz') as data:
        x, t, oracle = [data[k] for k in ('x', 't', 'oracle')]
    with np.load(folder / 'diagnostics/field.npz') as data:
        saved = data['prediction']
    model = joint_study.build(family) if is_joint else study.build(x.shape[1], family)
    checkpoint = folder / 'step_0005000.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['step'] == 5000
    study.restore_model(model, state)
    model.eval()
    prediction = np.concatenate([model(torch.from_numpy(x[k:k+1024]), torch.from_numpy(t[k:k+1024])).numpy()
                                 for k in range(0, len(x), 1024)])
    np.testing.assert_array_equal(prediction, saved)
    result = load(folder / 'result.json')
    mse = float(np.mean((prediction - oracle)**2))
    close(mse, result['oracle_mse'])
    with np.load(banks / 'sampling.npz') as data:
        source, target, target2, directions = [data[k] for k in ('source', 'target', 'target2', 'directions')]
    with np.load(folder / 'diagnostics/samples.npz') as data:
        samples = data['generated']
    score = sw2(samples, target, directions)
    floor = sw2(target, target2, directions)
    close(score, result['sw2']); close(floor, result['sw2_reference_floor'])
    # Reintegrate only the key positive Product16 comparison, with the native
    # recorded solver. Other cells verify saved samples and the metric only.
    sampling_replayed = dataset == 'rotated_product_bimodal_16d' and family in ('free_isru', 'swiglu64', 'swiglu160')
    if sampling_replayed:
        fresh = study.legacy.generated_samples(model, torch.from_numpy(source), 32).numpy()
        np.testing.assert_array_equal(fresh, samples)
    receipt['cells'].append(dict(dataset=dataset, family=family,
        parameters=sum(p.numel() for p in model.parameters()),
        width=model.raw.width, depth=model.raw.depth, field_examples=len(x),
        max_prediction_error=float(np.max(np.abs(prediction-saved))),
        oracle_mse=mse, sw2=score, sampling_replayed=sampling_replayed,
        input_stream_digest=state['input_stream_digest'],
        has_attention=any('attention' in type(m).__name__.lower() for m in model.modules()),
        checkpoint_sha256=sha(checkpoint)))
    receipt['input_sha256'][str(folder / 'result.json')] = sha(folder / 'result.json')
    print(dataset, family, 'PASS', flush=True)

report = args.repository / 'reports/report2/data/synthetic.csv'
for row in csv.DictReader(report.open()):
    source = args.repository / row['source']
    candidates = list(csv.DictReader(source.open()))
    dataset = {'Gaussian2D':'gaussian_aniso_2d', 'Ring8':'gmm_ring8_2d',
               'Product16':'rotated_product_bimodal_16d', 'Dependent signs':'sign_code',
               'Half mixture':'half_mixture'}[row['law']]
    candidates = [r for r in candidates if r.get('family', r.get('arm')) == row['arm']]
    if row['law'] in ('Gaussian2D', 'Ring8', 'Product16'):
        candidates = [r for r in candidates if r['dataset'] == dataset]
    elif row['law'] == 'Dependent signs':
        candidates = [r for r in candidates if r['law'] == dataset]
    elif row['law'] == 'Half mixture':
        candidates = [r for r in candidates if r['law'] == dataset]
    assert len(candidates) == 1, (row, candidates)
    for key in ('oracle_mse', 'sw2'):
        close(float(row[key]), float(candidates[0][key]))
    receipt['report_rows'] += 1
    receipt['input_sha256'][row['source']] = sha(source)
receipt['status'] = 'PASS'
receipt['limitations'] = ['No optimizer trajectory was rerun.',
    'Half-mixture report values were checked against published source rows, not its checkpoints.',
    'Numerical replay does not establish convergence, causal attribution, or image transfer.']
args.output.write_text(json.dumps(receipt, indent=2) + '\n')
print('AUDIT_REPLAY_PASS', len(receipt['cells']), flush=True)
