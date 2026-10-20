"""Read saved synthetic runs without importing torch or changing run files."""
import argparse
import itertools
import json
from pathlib import Path
from statistics import mean


def average(records, key):
    values = [record[key] for record in records if record.get(key) is not None]
    return mean(values) if values else None


def layer_summary(layers):
    result = {}
    for group, keys in (
        ('raw', ('sigma_max', 'sigma_median', 'stable_rank', 'spectral_radius')),
        ('effective', ('sigma_max', 'sigma_median', 'stable_rank', 'spectral_radius')),
        ('activation', ('s_mean', 'term1_rms_over_input',
                        'term2_rms_over_input', 'output_rms_over_input')),
    ):
        for key in keys:
            values = [layer[group][key] for layer in layers if key in layer[group]]
            if values:
                result[f'{group}.{key}'] = {
                    'min': min(values), 'mean': mean(values), 'max': max(values)
                }
    return result


def summarize(config_path, output_root):
    cfg = json.loads(config_path.read_text())
    results = []
    for dataset, model, seed in itertools.product(
        cfg['datasets'], cfg['models'], cfg['seeds']
    ):
        folder = output_root / cfg['experiment_id'] / dataset / model / f'seed{seed:03d}'
        result = {'experiment': cfg['experiment_id'], 'dataset': dataset,
                  'model': model, 'seed': seed,
                  'orders': cfg['model_options']['orders'] if model.startswith('ode_') else None,
                  'run_dir': str(folder), 'complete': False, 'last_step': None}
        path = folder / 'metrics.jsonl'
        if not path.is_file():
            result['status'] = 'missing'
            results.append(result)
            continue
        # A writer can leave an unfinished last line while the job is running.
        raw = path.read_text()
        lines = raw.splitlines() if raw.endswith('\n') else raw.splitlines()[:-1]
        records = [json.loads(line) for line in lines if line.strip()]
        if not records:
            result['status'] = 'no_metrics'
            results.append(result)
            continue
        last = records[-1]
        result['last_step'] = last['step']
        done = folder / 'COMPLETE'
        checkpoint = folder / 'checkpoints' / f'step_{cfg["steps"]:07d}.pt'
        result['complete'] = (done.is_file()
                              and done.read_text().strip() == str(cfg['steps'])
                              and last['step'] == cfg['steps'] and checkpoint.is_file())
        result['status'] = 'complete' if result['complete'] else 'incomplete'
        late = [r for r in records if r['step'] >= 0.6 * cfg['steps']]
        result.update({
            'last_pair_mse': last['validation_pair_mse'],
            'last_oracle_mse': last.get('validation_oracle_mse'),
            'late_pair_mse_mean': average(late, 'validation_pair_mse'),
            'late_oracle_mse_mean': average(late, 'validation_oracle_mse'),
            'late_eval_steps': [r['step'] for r in late],
            'last_sample': last.get('sample'),
            'loop_wall_seconds': last.get('wall_seconds'),
            'initial_layers': layer_summary(records[0].get('ode_layers', [])),
            'last_layers': layer_summary(last.get('ode_layers', [])),
            'curve': [{**{k: r.get(k) for k in (
                'step', 'validation_pair_mse', 'validation_oracle_mse', 'sample'
            )}, 'layers': layer_summary(r.get('ode_layers', []))} for r in records],
        })
        results.append(result)
    return results


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=root.parent / 'runs/synthetic')
    parser.add_argument('--json-output', type=Path)
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--configs', nargs='+', default=['train_v1.json', 'train_k1_v1.json'],
                        help='config filenames under configs/synthetic; defaults preserve the historical 15-run report')
    args = parser.parse_args()
    results = []
    for name in args.configs:
        results.extend(summarize(root / 'configs/synthetic' / name,
                                 args.output_root.resolve()))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
    print('dataset | model | K | status | step | oracle MSE | SW2')
    for result in results:
        sample = result.get('last_sample') or {}
        number = lambda value: '-' if value is None else f'{value:.6g}'
        print(' | '.join(str(value) for value in (
            result['dataset'], result['model'], result['orders'] or '-',
            result['status'], result['last_step'] if result['last_step'] is not None else '-',
            number(result.get('last_oracle_mse')), number(sample.get('sliced_w2')),
        )))
    print(f'Complete: {sum(r["complete"] for r in results)}/{len(results)}')
    if args.require_complete and not all(r['complete'] for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
