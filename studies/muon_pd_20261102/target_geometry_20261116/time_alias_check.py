"""Construct an exactly sampler-aliased field and measure actual learned use."""
import argparse
from pathlib import Path
import numpy as np
from geometry import read, dump, sha, target, TIMES
from frozen_probe import restore, tensor_hashes, assets, torch, base, exact_jacobian


@torch.no_grad()
def prediction(model, x, t):
    return torch.cat([model(x[i:i+256], t[i:i+256]) for i in range(0, len(x), 256)]).numpy()


def jacobians(model, arrays):
    values = []
    for time in TIMES:
        point = (1-time)*arrays['source_0'][:128].astype(float)+time*arrays['target_0'][:128].astype(float)
        x = torch.from_numpy(point)
        _, derivative = exact_jacobian(model, x, x.new_full((len(x), 1), time))
        values.append(derivative.numpy())
    return np.asarray(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    for key in ['study', 'data-root', 'output']:
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    study, data_root, out = map(Path, [args.study, args.data_root, args.output])
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    base.verify_sources()
    anchors = read(study/'anchors.json')
    rows = []
    for row in anchors['references']+anchors['new_cases']:
        model, law, cell, evaluation, identity = restore(study, anchors, row, data_root)
        arrays, _, binding = assets(anchors['assets'], cell['dataset'], 'selection', 'cpu')
        source32 = torch.from_numpy(np.concatenate([arrays[f'source_{b}'][:2] for b in range(8)]))
        gx32 = torch.from_numpy(arrays['field_x'][:1025])
        gt32 = torch.arange(1025, dtype=torch.float32)[:, None]/1024
        snapshot32 = tensor_hashes(model)
        original32 = model.time.frequencies.clone()
        original_grid32 = prediction(model, gx32, gt32)
        original_endpoint32 = base.train.generated_samples(model, source32, 128).numpy()
        try:
            with torch.no_grad():
                model.time.frequencies[-1] = original32[-1]-1024
            alias_grid32 = prediction(model, gx32, gt32)
            alias_endpoint32 = base.train.generated_samples(model, source32, 128).numpy()
        finally:
            with torch.no_grad():
                model.time.frequencies.copy_(original32)
        assert tensor_hashes(model) == snapshot32
        assert np.allclose(original_grid32, alias_grid32, atol=1e-4, rtol=1e-4)
        assert np.allclose(original_endpoint32, alias_endpoint32, atol=3e-4, rtol=3e-4)
        model.double()
        snapshot = tensor_hashes(model)
        frequencies = model.time.frequencies
        original = frequencies.clone()
        x, t = torch.from_numpy(arrays['field_x']).double(), torch.from_numpy(arrays['field_t']).double()
        grid_t = torch.arange(1025, dtype=torch.double)[:, None]/1024
        grid_x = x[:1025]
        source = torch.from_numpy(np.concatenate([arrays[f'source_{b}'][:2] for b in range(8)])).double()
        original_field = prediction(model, x, t)
        original_grid = prediction(model, grid_x, grid_t)
        original_endpoint = base.train.generated_samples(model, source, 128).numpy()
        original_jacobians = jacobians(model, arrays)
        true_jacobians = np.asarray([target(law, (1-time)*arrays['source_0'][:128].astype(float)+
            time*arrays['target_0'][:128].astype(float), time)['velocity_jacobian'] for time in TIMES])
        try:
            with torch.no_grad():
                frequencies[-1] = original[-1]-1024
            alias_field = prediction(model, x, t)
            alias_grid = prediction(model, grid_x, grid_t)
            alias_endpoint = base.train.generated_samples(model, source, 128).numpy()
            alias_jacobians = jacobians(model, arrays)
        finally:
            with torch.no_grad():
                frequencies.copy_(original)
        assert tensor_hashes(model) == snapshot
        assert np.allclose(original_grid, alias_grid, atol=1e-9, rtol=1e-9), row['id']
        assert np.allclose(original_endpoint, alias_endpoint, atol=1e-9, rtol=1e-9), row['id']
        file = row['id']+'.npz'
        np.savez_compressed(out/file, original_field=original_field, alias_field=alias_field,
            original_grid=original_grid, alias_grid=alias_grid,
            original_endpoint=original_endpoint, alias_endpoint=alias_endpoint,
            original_grid32=original_grid32, alias_grid32=alias_grid32,
            original_endpoint32=original_endpoint32, alias_endpoint32=alias_endpoint32,
            original_jacobians=original_jacobians, alias_jacobians=alias_jacobians,
            oracle_jacobians=true_jacobians,
            oracle=arrays['field_oracle'], grid_t=grid_t.numpy(), field_t=t.numpy())
        delta_mse = float(np.mean((original_field-alias_field)**2))
        oracle_mse = float(np.mean((original_field-arrays['field_oracle'])**2))
        alias_mse = float(np.mean((alias_field-arrays['field_oracle'])**2))
        rows.append(dict(id=row['id'], family=cell['family'], dataset=cell['dataset'],
            original_frequency=float(original[-1]), alias_frequency=float(original[-1]-1024),
            field_points=len(x), grid_points=1025, endpoint_sources=16,
            grid_max_difference=float(np.max(np.abs(original_grid-alias_grid))),
            endpoint_max_difference=float(np.max(np.abs(original_endpoint-alias_endpoint))),
            fp32_grid_max_difference=float(np.max(np.abs(original_grid32-alias_grid32))),
            fp32_endpoint_max_difference=float(np.max(np.abs(original_endpoint32-alias_endpoint32))),
            off_grid_difference_mse=delta_mse, native_oracle_field_mse=oracle_mse,
            alias_oracle_field_mse=alias_mse,
            off_grid_change_over_fitting_error_rms=float(np.sqrt(delta_mse/oracle_mse)),
            selection_original_jacobian_error=float(np.mean(np.sum((original_jacobians-true_jacobians)**2, axis=(2, 3))/law.dim)),
            selection_alias_jacobian_error=float(np.mean(np.sum((alias_jacobians-true_jacobians)**2, axis=(2, 3))/law.dim)),
            parameters_and_buffers_restored_exactly=True, raw_file=file, raw_sha256=sha(out/file),
            assets_sha256=binding['sha256'], **identity))
        dump(out/'rows.json', dict(rows=rows))
        print(f'{len(rows)}/12 alias identity and learned response', flush=True)
    dump(out/'COMPLETE.json', dict(passed=True, cases=12, rows_sha256=sha(out/'rows.json'),
        source_sha256=sha(Path(__file__)), anchors_sha256=sha(study/'anchors.json')))


if __name__ == '__main__':
    main()
