"""Recompute all matched common-bank scores in NumPy, without Torch metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sw2(x: np.ndarray, y: np.ndarray, directions: np.ndarray) -> float:
    projected = np.sort(np.asarray(x, dtype=float)@directions.T, axis=0)
    projected -= np.sort(np.asarray(y, dtype=float)@directions.T, axis=0)
    return float(np.sqrt(np.mean(projected*projected)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--bindings', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    terminal = read(root/'ops/TERMINAL.json')
    assert terminal['passed']
    controls = read(Path(args.bindings))
    rows, maximum = [], 0.
    tasks = []
    for row in controls['references']:
        assert sha(Path(row['folder'])/'config.json') == row['config_sha256']
        assert sha(Path(row['folder'])/'checkpoints/step_0020000.pt') == row['checkpoint_sha256']
        tasks.append(dict(id=row['id'], dataset=row['dataset'], family=row['family'],
            checkpoint_sha256=row['checkpoint_sha256'], paths={split:Path(v['summary_file']).parent
            for split, v in row['evaluations'].items()}, bound=row))
    for row in terminal['cases']:
        assert [r['actual_terminal_exit'] for r in row['completed']] == [0, 0, 0]
        folder = Path(row['folder'])
        assert (folder/'COMPLETE').read_text().strip() == '20000'
        assert (folder/'ADAPTER_COMPLETE').exists()
        tasks.append(dict(id=row['id'], dataset=row['dataset'], family='state_rk4_2',
            checkpoint_sha256=sha(folder/'checkpoints/step_0020000.pt'),
            paths={split:root/'analysis'/split/row['id'] for split in ['selection', 'confirmation']}))
    for task in tasks:
        for split, folder in task['paths'].items():
            complete, binding = read(folder/'COMPLETE.json'), read(folder/'binding.json')
            assert complete['passed'] and not complete['witness']
            assert complete['steps'] == [20000] or 20000 in complete['steps']
            assert binding['split'] == split and binding['heun_steps'] == 128
            assert binding['banks'] == 8 and binding['samples_per_bank'] == 4096
            asset = binding['assets']
            assert sha(Path(asset['file'])) == asset['sha256']
            manifest = read(Path(controls['assets_root'])/'manifest.json')
            expected = next(r for r in manifest['rows'] if r['dataset'] == task['dataset'] and r['split'] == split)
            assert asset == expected
            if 'bound' in task:
                assert sha(folder/'summary.json') == task['bound']['evaluations'][split]['sha256']
            summary = next(r for r in read(folder/'summary.json') if r['step'] == 20000)
            assert summary['checkpoint_sha256'] == task['checkpoint_sha256']
            with np.load(asset['file']) as bundle:
                arrays = {k:bundle[k].copy() for k in bundle.files}
            directions = arrays['directions'].astype(float)
            prediction_file = folder/'field-20000.npy'
            prediction = np.load(prediction_file).astype(float)
            field_mse = float(np.mean((prediction-arrays['field_oracle'].astype(float))**2))
            generated, targets, scores, hashes = [], [], [], []
            for bank in range(8):
                file = folder/f'20000-b{bank}.npy'
                value = np.load(file)
                target = arrays[f'target_{bank}']
                assert value.shape == target.shape and len(value) == 4096 and np.isfinite(value).all()
                generated.append(value)
                targets.append(target)
                scores.append(sw2(value, target, directions))
                hashes.append(dict(file=str(file), sha256=sha(file)))
            pooled = sw2(np.concatenate(generated), np.concatenate(targets), directions)
            measured = dict(field_mse=field_mse, bank_sw2=scores,
                sw2_mean=float(np.mean(scores)), sw2_sd=float(np.std(scores, ddof=1)), pooled_sw2=pooled)
            for key, value in measured.items():
                error = float(np.max(np.abs(np.asarray(value)-np.asarray(summary[key]))))
                maximum = max(maximum, error)
                assert error < 1e-10, (task['id'], split, key, error)
            rows.append(dict(id=task['id'], dataset=task['dataset'], family=task['family'],
                split=split, **measured, checkpoint_sha256=task['checkpoint_sha256'],
                evaluation_summary_sha256=sha(folder/'summary.json'),
                asset_file=asset['file'], asset_sha256=asset['sha256'],
                field_file=str(prediction_file), field_sha256=sha(prediction_file), sample_files=hashes))
            print(task['id'], split, 'verified', flush=True)
            (out/'progress.json').write_text(json.dumps(dict(completed=len(rows), total=16))+'\n')
    assert len(rows) == 16
    result = dict(passed=True, rows=rows, models=8, splits=2,
        maximum_numpy_difference=maximum, source_sha256=sha(Path(__file__)),
        terminal_sha256=sha(root/'ops/TERMINAL.json'), bindings_sha256=sha(Path(args.bindings)))
    (out/'PASS.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('passed', len(rows), maximum, flush=True)


if __name__ == '__main__':
    main()
