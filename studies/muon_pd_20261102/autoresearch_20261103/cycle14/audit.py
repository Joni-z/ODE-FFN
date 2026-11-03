"""Frozen sampling validation; calls the archived native model and evaluator."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cycle13'))
from trial13 import base, build_model, torch

HERE = Path(__file__).resolve().parent
ROOT = base.HERE.parents[1]
TAG = 'professor_validation14_20261104'


def array_hash(x):
    x = np.ascontiguousarray(x)
    return hashlib.sha256(str((x.shape, x.dtype.str)).encode() + x.tobytes()).hexdigest()


def cohort():
    base.verify_sources()
    grid = json.loads((HERE / 'grid.json').read_text())
    tables = {}
    for name, digest in grid['source_csvs'].items():
        assert base.digest(ROOT / name) == digest
        with (ROOT / name).open() as f:
            tables[name] = list(csv.DictReader(f))
    assert grid['model_seed'] == 0 and len(grid['cells']) == 10
    assert len({c['id'] for c in grid['cells']}) == 10
    for case in grid['cells']:
        row = next(r for r in tables[case['source_csv']] if r['id'] == case['source_id'])
        assert row['checkpoint_sha256'] == case['final_checkpoint_sha256']
        assert all(row[k] == case[k] for k in ('host', 'dataset', 'family', 'optimizer', 'data_version'))
        assert float(row['projection_lr']) == case['projection_lr']
        assert float(row['replayed_field_mse']) == case['native_field_mse']
        assert float(row['numpy_test_sw2']) == case['native_test_sw2']
    return grid


def bank_arrays(law, directions, count, banks, stream_base):
    arrays = {'directions': directions.copy()}
    for bank in range(banks):
        rng = np.random.default_rng(stream_base + bank)
        arrays[f'source_{bank}'] = rng.standard_normal((count, law.dim)).astype(np.float32)
        arrays[f'target_{bank}'] = law.sample(rng, count)
        arrays[f'target2_{bank}'] = law.sample(rng, count)
    return arrays


def tensor_snapshot(model):
    return {k: v.detach().clone() for k, v in
            list(model.named_parameters()) + list(model.named_buffers())}


def same_tensors(model, snapshot):
    now = tensor_snapshot(model)
    assert now.keys() == snapshot.keys()
    assert all(torch.equal(now[k], v) for k, v in snapshot.items())


def inputs(case, grid, data_root, device):
    folder = (Path(grid['workspaces'][case['host']]) / case['run_dir'] / 'output' /
              case['experiment_id'] / case['dataset'] / case['source_id'] / 'seed000')
    assert (folder / 'ADAPTER_COMPLETE').exists()
    for name, digest in case['file_hashes'].items():
        assert base.digest(name) == digest, name
    cfg = json.loads((folder / 'config.json').read_text())
    assert cfg['steps'] == 20000 and cfg['seeds'] == [0]
    assert cfg['sample_count'] == 4096 and cfg['heun_steps'] == 32
    assert cfg['cell']['id'] == case['source_id']
    assert all(cfg['cell'][k] == case[k] for k in ('dataset', 'family', 'data_version', 'optimizer'))
    cache = base.verify_data(data_root, case)
    assert cache.resolve() == Path(cfg['data_root']).resolve()
    manifest = cache / 'manifest.json'
    spec = json.loads(manifest.read_text())['datasets'][case['dataset']]
    law = base.train.load_distribution(spec)
    fixed = base.train.fixed_eval_data(cache / case['dataset'], law, spec['dim'], 0,
                                       device, 4096, 256, 'test')
    return folder, cfg, manifest, law, fixed


def restored(case, folder, manifest, dim, step, device):
    checkpoint = folder / 'checkpoints' / f'step_{step:07d}.pt'
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    assert state['step'] == step
    assert state['config_sha256'] == base.digest(folder / 'config.json')
    assert state['dataset_manifest_sha256'] == base.digest(manifest)
    torch.manual_seed(0)
    model = build_model(dim, case['family']).to(device).eval()
    base.train.restore_model(model, state)
    return model, dict(checkpoint_sha256=base.digest(checkpoint),
                       config_sha256=state['config_sha256'],
                       manifest_sha256=state['dataset_manifest_sha256'])


@torch.no_grad()
def field(model, fixed):
    prediction = torch.cat([model(x, t) for x, t in
                            zip(fixed['x'].split(1024), fixed['t'].split(1024))])
    mse = float((prediction.double() - fixed['oracle'].double()).square().mean())
    return prediction.cpu().numpy(), mse if np.isfinite(mse) else None


def self_test(device):
    from synthetic.data import suite
    law = suite()['gmm_ring8_2d']
    dirs = np.eye(2, dtype=np.float32)
    a = bank_arrays(law, dirs, 64, 2, 2026110400)
    b = bank_arrays(law, dirs, 64, 2, 2026110400)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not np.array_equal(a['source_0'], a['source_1'])
    x, y = (torch.from_numpy(a[k]).to(device) for k in ('source_0', 'target_0'))
    u, v = a['source_0'].astype(float) @ dirs.T, a['target_0'].astype(float) @ dirs.T
    expected = float(np.sqrt(np.mean((np.sort(u, axis=0) - np.sort(v, axis=0)) ** 2)))
    assert abs(base.train.sliced_w2(x.double(), y.double(), torch.from_numpy(dirs).to(device).double()) - expected) < 1e-12
    class Constant(torch.nn.Module):
        def forward(self, x, t):
            return torch.ones_like(x) * .25
    zero = torch.zeros_like(x)
    assert torch.equal(base.train.generated_samples(Constant(), zero, 32), zero + .25)
    rows = []
    for dim in (2, 16):
        for family in ('pd_isru_k1', 'free_isru', 'swiglu64', 'swiglu192'):
            torch.manual_seed(0)
            model = build_model(dim, family).to(device).eval()
            with torch.no_grad():
                model.output.weight.fill_(.01)
                fixture = torch.randn(8, dim, device=device)
                t = torch.full((8, 1), .4, device=device)
                prediction = model(fixture, t).clone()
                state = dict(model={k: v.clone() for k, v in model.state_dict().items()},
                             runtime_buffers={k: v.clone() for k, v in model.named_buffers()})
                torch.manual_seed(99)
                copy = build_model(dim, family).to(device).eval()
                base.train.restore_model(copy, state)
                assert torch.equal(prediction, copy(fixture, t))
                before = tensor_snapshot(copy)
                sample = base.train.generated_samples(copy, fixture, 32)
                assert bool(torch.isfinite(sample).all())
                same_tensors(copy, before)
            rows.append(dict(dim=dim, family=family, restore_output_exact=True, tensors_unchanged=True))
    return dict(passed=True, device=str(device), torch=torch.__version__, rows=rows,
                bank_determinism=True, numpy_sw2_agrees=True, native_constant_heun_exact=True)


def run_case(case, grid, data_root, device, out, witness=False):
    assert device.type == grid['execution_devices'][case['host']]
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    folder, cfg, manifest, law, fixed = inputs(case, grid, data_root, device)
    arrays = bank_arrays(law, fixed['directions'].cpu().numpy(), grid['sample_count'],
                         grid['banks'], grid['evaluation_stream_base'])
    bank_identity = {k: array_hash(v) for k, v in arrays.items()}
    np.savez_compressed(out / 'banks.npz', **arrays)
    base.dump(out / 'binding.json', dict(case=case, source_config=cfg,
              source_sha256=base.digest(__file__), grid_sha256=base.digest(HERE / 'grid.json'),
              bank_identity=bank_identity, kind=grid['kind'], model_seed=0))
    rows, fields, dispatches = [], [], []
    for step in grid['checkpoint_steps']:
        model, identity = restored(case, folder, manifest, law.dim, step, device)
        before = tensor_snapshot(model)
        prediction, mse = field(model, fixed)
        np.savez_compressed(out / f'field_{step}.npz', prediction=prediction,
                            oracle=fixed['oracle'].cpu().numpy())
        fields.append(dict(step=step, oracle_mse=mse, **identity))
        if step == 20000:
            generated = base.train.generated_samples(model, fixed['sample_source'], 32)
            native = generated.cpu().numpy()
            with np.load(folder / 'final_test_examples.npz', allow_pickle=False) as saved:
                archived = saved['generated'].copy()
                exact = np.array_equal(native, archived)
                target_exact = np.array_equal(fixed['sample_target'].cpu().numpy(), saved['target'])
            coordinate_pass = exact if device.type == 'cuda' else np.allclose(native, archived, rtol=1e-5, atol=1e-6)
            native_sw2 = base.train.sliced_w2(generated.double(), fixed['sample_target'].double(), fixed['directions'].double())
            np.savez_compressed(out / 'native_anchor.npz', generated=native, archived_generated=archived,
                target=fixed['sample_target'].cpu().numpy(), directions=arrays['directions'])
            passed = bool(coordinate_pass and target_exact and mse is not None and
                          abs(mse-case['native_field_mse']) < 2e-6 and
                          abs(native_sw2-case['native_test_sw2']) < 2e-6)
            base.dump(out / 'native_anchor.json', dict(passed=passed, generated_exact=exact, target_exact=target_exact,
                generated_max_abs_error=float(np.max(np.abs(native-archived))),
                anchor_policy='bitwise' if device.type == 'cuda' else 'rtol1e-5_atol1e-6',
                oracle_mse=mse, native_sw2=native_sw2, **identity))
            assert passed, 'native anchor failed: retained arrays and native_anchor.json record the errors'
        if witness:
            # Nonzero real-checkpoint fresh-bank dispatch, not the formal cohort.
            small = torch.from_numpy(arrays['source_0'][:64]).to(device)
            finite = bool(torch.isfinite(base.train.generated_samples(model, small, 128)).all())
            dispatches.append(dict(checkpoint_step=step, samples=64, heun_steps=128, finite=finite))
            base.dump(out/'dispatches.json', dispatches)
            assert finite
        else:
            for bank in range(grid['banks']):
                source = torch.from_numpy(arrays[f'source_{bank}']).to(device)
                target = torch.from_numpy(arrays[f'target_{bank}']).to(device).double()
                target2 = torch.from_numpy(arrays[f'target2_{bank}']).to(device).double()
                directions = fixed['directions'].double()
                floor = base.train.sliced_w2(target, target2, directions)
                for steps in grid['heun_steps']:
                    if device.type == 'cuda': torch.cuda.synchronize()
                    tick = time.monotonic()
                    generated = base.train.generated_samples(model, source, steps)
                    if device.type == 'cuda': torch.cuda.synchronize()
                    seconds = time.monotonic() - tick
                    sample = generated.cpu().numpy()
                    finite = np.isfinite(sample).all()
                    sw2 = base.train.sliced_w2(generated.double(), target, directions) if finite else None
                    name = f'generated_{step}_b{bank}_h{steps}.npy'
                    np.save(out / name, sample, allow_pickle=False)
                    row = dict(cell=case['id'], source_id=case['source_id'], dataset=case['dataset'],
                        family=case['family'], optimizer=case['optimizer'], projection_lr=case['projection_lr'],
                        model_seed=0, checkpoint_step=step, bank=bank,
                        evaluation_stream=grid['evaluation_stream_base'] + bank, heun_steps=steps,
                        nfe=2*steps, sw2=sw2, target_target_sw2=floor,
                        finite_fraction=float(np.isfinite(sample).all(axis=1).mean()),
                        generated_file=name, generated_sha256=base.digest(out/name),
                        shared_gpu_sampler_seconds=seconds, **identity)
                    rows.append(row)
                    base.train.json_line(out / 'rows.jsonl', row)
                    base.train.json_line(out / 'metrics.jsonl', dict(step=step,
                        validation_oracle_mse=mse, wall_seconds=time.monotonic()-started,
                        sample=dict(completed=len(rows), total=48, bank=bank, heun_steps=steps, sw2=sw2),
                        kind='frozen_evaluation_no_training'))
        same_tensors(model, before)
        del model
        base.dump(out / 'progress.json', dict(checkpoints=len(fields), evaluations=len(rows), fields=fields))
    assert len(fields) == 3 and (witness or len(rows) == 48)
    result = dict(passed=True, witness=witness, cell=case['id'], device=str(device),
                  kind=grid['kind'], model_seed=0, evaluations=len(rows), fields=fields,
                  tensors_unchanged=True, native_anchor_passed=True,
                  witness_dispatches=dispatches,
                  bank_identity=bank_identity, wall_seconds=time.monotonic()-started)
    base.dump(out / 'result.json', result)
    (out / 'ADAPTER_COMPLETE').write_text('Completed frozen evaluation; no training.\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cell')
    p.add_argument('--host', choices=['145', 'lenovo'])
    p.add_argument('--data-root')
    p.add_argument('--output', required=True)
    p.add_argument('--device', required=True, choices=['cpu', 'cuda'])
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--witness', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(1)
    device = torch.device(a.device)
    grid = cohort()
    if a.self_test:
        result = self_test(device)
        result.update(producer_sha256=base.digest(__file__), grid_sha256=base.digest(HERE/'grid.json'))
        base.dump(a.output, result)
    elif a.witness:
        assert a.host and a.data_root
        root = Path(a.output)
        root.mkdir(parents=True, exist_ok=False)
        kernel = self_test(device)
        rows = [run_case(c, grid, a.data_root, device, root/c['id'], True)
                for c in grid['cells'] if c['host'] == a.host]
        assert len(rows) == 5
        result = dict(passed=True, host=a.host, device=str(device), kernel=kernel,
                      physical_checkpoints=15, native_final_anchors=5, rows=rows)
        base.dump(root / 'witness.json', result)
    else:
        case = next(c for c in grid['cells'] if c['id'] == a.cell)
        result = run_case(case, grid, a.data_root, device,
            Path(a.output)/TAG/case['dataset']/case['id']/'seed000')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
