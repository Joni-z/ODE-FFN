"""Read-only replay of frozen first-wave checkpoints and independent NumPy SW2."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_sw2(x, y, directions):
    # Double-precision NumPy calculation, independent of torch.sort in training.
    u = np.sort(x.astype(np.float64) @ directions.astype(np.float64).T, axis=0)
    v = np.sort(y.astype(np.float64) @ directions.astype(np.float64).T, axis=0)
    return float(np.sqrt(np.mean((u-v)**2)))


def mean(values):
    values = list(values)
    return float(np.mean(values)) if values else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, help='frozen first-wave study source')
    p.add_argument('--output-root', required=True, help='frozen first-wave outputs')
    p.add_argument('--save', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = p.parse_args()
    source = Path(args.source).resolve()
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location('frozen_wave1', source/'run.py')
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    runtime.verify_sources()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    rows, curves, updates = [], [], []
    for folder in sorted((Path(args.output_root)/'muon_pd_20261102').glob('*/*/seed000')):
        cfg = json.loads((folder/'config.json').read_text())
        cell = cfg['cell']
        metrics = [json.loads(x) for x in (folder/'metrics.jsonl').read_text().splitlines()]
        final = metrics[-1]
        test = json.loads((folder/'final_test_sampling.json').read_text())
        manifest_path = Path(cfg['data_root'])/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        ds = manifest['datasets'][cell['dataset']]
        data = Path(cfg['data_root'])/cell['dataset']
        runtime.verify_data(Path(cfg['data_root']).parent, cell)
        checkpoint = torch.load(folder/'checkpoints/step_0020000.pt', map_location=device, weights_only=False)
        assert checkpoint['step'] == final['step'] == 20000
        assert checkpoint['config_sha256'] == sha(folder/'config.json')
        assert checkpoint['dataset_manifest_sha256'] == sha(manifest_path)
        assert checkpoint['identity']['seed'] == 0
        assert (folder/'ADAPTER_COMPLETE').exists()
        torch.manual_seed(0)
        model = runtime.build_model(ds['dim'], cell['family']).to(device)
        runtime.train.restore_model(model, checkpoint)
        model.eval()
        distribution = runtime.train.load_distribution(ds)
        fixed = runtime.train.fixed_eval_data(data, distribution, ds['dim'], 0,
                                             device, 4096, 256, 'test')
        count, error = 0, 0.
        with torch.no_grad():
            for lo in range(0, len(fixed['x']), 1024):
                pred = model(fixed['x'][lo:lo+1024], fixed['t'][lo:lo+1024])
                delta = (pred.double()-fixed['oracle'][lo:lo+1024].double()).cpu().numpy()
                error += float(np.sum(delta**2))
                count += delta.size
            replay = runtime.train.generated_samples(model, fixed['sample_source'], 32).cpu().numpy()
        with np.load(folder/'final_test_examples.npz', allow_pickle=False) as saved:
            generated, target = saved['generated'].copy(), saved['target'].copy()
        assert np.array_equal(target, fixed['sample_target'].cpu().numpy())
        directions = fixed['directions'].cpu().numpy()
        sw2 = independent_sw2(generated, target, directions)
        replay_error = float(np.max(np.abs(replay-generated)))
        field_error = abs(error/count-final['validation_oracle_mse'])
        sample_error = abs(sw2-test['sample']['sliced_w2'])
        assert field_error < 2e-6 and sample_error < 2e-6 and replay_error < 2e-5, (
            cell['id'], field_error, sample_error, replay_error)
        rows.append(dict(cell, replayed_field_mse=error/count, numpy_test_sw2=sw2,
                         field_abs_error=field_error, sw2_abs_error=sample_error,
                         generated_replay_max_abs_error=replay_error,
                         checkpoint_sha256=sha(folder/'checkpoints/step_0020000.pt')))
        for metric in metrics:
            activations = [l['activation'] for l in metric['ode_layers']]
            row = dict(cell=cell['id'], dataset=cell['dataset'], family=cell['family'],
                       optimizer=cell['optimizer'], lr=cell['projection_lr'],
                       step=metric['step'], field_mse=metric['validation_oracle_mse'],
                       validation_sw2=metric.get('sample', {}).get('sliced_w2'))
            for key in ('s_mean', 's_std', 's_sample_std', 's_channel_std',
                        'clock_outer_saturated_fraction', 'output_rms_over_input',
                        'term1_rms_over_input', 'term2_rms_over_input'):
                row[key] = mean(a[key] for a in activations if key in a)
            for key in ('sigma_max', 'stable_rank', 'spectral_radius', 'nonnormality'):
                row['p_'+key] = mean(l['raw'][key] for l in metric['ode_layers'])
            jac = [v for values in metric.get('layer_jacobian_sigma_max', {}).values() for v in values]
            row['probe_jacobian_max'] = max(jac) if jac else None
            curves.append(row)
        for line in (folder/'update_spectra.jsonl').read_text().splitlines():
            event = json.loads(line)
            matrices = list(event['matrices'].values())
            row = dict(cell=cell['id'], dataset=cell['dataset'], family=cell['family'],
                       optimizer=cell['optimizer'], lr=cell['projection_lr'], step=event['step'])
            for label in ('weight', 'delta', 'clipped_gradient', 'muon_momentum'):
                available = [m[label] for m in matrices if label in m]
                for key in ('frobenius', 'stable_rank', 'entropy_rank'):
                    row[label+'_'+key] = mean(m[key] for m in available)
                row[label+'_sigma_max'] = mean(m['singular_values'][0] for m in available)
            updates.append(row)
        print(json.dumps(dict(cell=cell['id'], field_abs_error=field_error,
                              sw2_abs_error=sample_error, replay_max_abs_error=replay_error)), flush=True)
        # Release GPU-resident optimizer state loaded from the checkpoint too.
        del checkpoint, model, fixed
    result = dict(source=str(source), source_manifest_sha256=sha(source/'source_provenance.json'),
                  torch=torch.__version__, device=args.device, passed=True,
                  endpoints=rows, curves=curves, updates=updates)
    path = Path(args.save)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True, endpoints=len(rows), saved=str(path))))


if __name__ == '__main__':
    main()
