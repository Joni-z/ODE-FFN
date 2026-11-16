"""Observe complete output derivatives of the twelve registered frozen models."""
import argparse
import os
from pathlib import Path
import sys
import time
import numpy as np
from geometry import TIMES, read, sha, dump, target, metrics
from oracle_check import HERE, base, torch, exact_jacobian, assets
from gpu_resolution import builder, verify_builder_sources


def tensor_hashes(model) -> dict:
    import hashlib
    return {name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, value in list(model.named_parameters())+list(model.named_buffers())}


def restore(study: Path, anchor: dict, row: dict, data_root: Path):
    folder = Path(row['folder'])
    for name, digest in row.get('bound_files', {}).items():
        assert sha(folder/name) == digest, (row['id'], name)
    cfg = read(folder/'config.json')
    assert cfg['steps'] == 20000 and cfg['seeds'] == [0]
    cell = cfg['cell']
    cache = base.verify_data(data_root, cell)
    manifest = cache/'manifest.json'
    law = base.train.load_distribution(read(manifest)['datasets'][cell['dataset']])
    file = folder/'checkpoints/step_0020000.pt'
    state = torch.load(file, map_location='cpu', weights_only=False)
    assert state['step'] == 20000 and state['config_sha256'] == sha(folder/'config.json')
    assert state['dataset_manifest_sha256'] == sha(manifest)
    if cell['family'] in ('palindrome_operator_midpoint4', 'alternating_operator_midpoint4'):
        portable = study/'portable/sampling-models'
        case = next(c for c in read(portable/'portable.json')['cases'] if c['id'] == row['id'])
        verify_builder_sources(case)
        assert sha(file) == case['checkpoint_sha256']
    torch.manual_seed(0)
    model = builder(law.dim, cell['family']).cpu().eval()
    base.train.restore_model(model, state)
    buffers = dict(model.named_buffers())
    assert all(torch.equal(buffers[n], v) for n, v in state['runtime_buffers'].items())
    evaluation = study/'analysis/selection'/row['id']
    complete = read(evaluation/'COMPLETE.json')
    assert complete['passed']
    summary = next(s for s in read(evaluation/'summary.json') if s['step'] == 20000)
    assert summary['checkpoint_sha256'] == sha(file)
    identity = dict(checkpoint_sha256=sha(file), config_sha256=sha(folder/'config.json'),
                    dataset_manifest_sha256=sha(manifest),
                    native_field_sha256=sha(evaluation/'field-20000.npy'))
    return model, law, cell, evaluation, identity


def main() -> None:
    p = argparse.ArgumentParser()
    for key in ['study', 'data-root', 'output']:
        p.add_argument('--'+key, required=True)
    p.add_argument('--witness', action='store_true')
    args = p.parse_args()
    study, out, data_root = map(Path, [args.study, args.output, args.data_root])
    out.mkdir(parents=True, exist_ok=False)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    torch.set_num_threads(1)
    base.verify_sources()
    anchor = read(study/'anchors.json')
    assert read(study/'analysis/COMPLETE.json')['passed']
    rows = anchor['references']+anchor['new_cases']
    assert len(rows) == 12 and len({r['id'] for r in rows}) == 12
    n, times = (4, (.5,)) if args.witness else (128, TIMES)
    started = time.monotonic()
    dump(out/'RUNNING.json', dict(pid=os.getpid(), cases=12, points_per_time=n,
                                  times=times, witness=args.witness))
    summaries, identities = [], []
    for index, row in enumerate(rows):
        case_start = time.monotonic()
        model, law, cell, evaluation, identity = restore(study, anchor, row, data_root)
        arrays, _, bound = assets(anchor['assets'], cell['dataset'], 'selection', 'cpu')
        # Bind this adapter to actual saved FP32 predictions before conversion.
        x = torch.from_numpy(arrays['field_x'][:128])
        t = torch.from_numpy(arrays['field_t'][:128])
        with torch.no_grad():
            fp32 = model(x, t).numpy()
        saved = np.load(evaluation/'field-20000.npy')[:128]
        native_error = float(np.max(np.abs(fp32-saved)))
        assert np.allclose(fp32, saved, atol=2e-5, rtol=2e-5), (row['id'], native_error)
        model.double()
        snapshot = tensor_hashes(model)
        with torch.no_grad():
            fp64 = model(x.double(), t.double()).numpy()
        precision_error = float(np.max(np.abs(fp64-fp32)))
        assert np.allclose(fp64, fp32, atol=3e-5, rtol=3e-5), (row['id'], precision_error)
        full = dict(time=np.asarray(times), source=arrays['source_0'][:n].astype(float),
                    target=arrays['target_0'][:n].astype(float))
        for key in ['x', 'prediction', 'jacobian', 'oracle', 'oracle_jacobian', 'oracle_image_jacobian']:
            full[key] = []
        time_rows = []
        fd_error = batch_error = None
        for value in times:
            point = (1-value)*full['source']+value*full['target']
            xt = torch.from_numpy(point)
            tt = xt.new_full((n, 1), value)
            pred, jac = exact_jacobian(model, xt, tt)
            truth = target(law, point, value)
            if value == .5:
                direction = torch.sin(torch.arange(1, law.dim+1, dtype=torch.double))
                direction /= direction.norm()
                eps = 1e-5
                with torch.no_grad():
                    fd = (model(xt[:4]+eps*direction, tt[:4])-model(xt[:4]-eps*direction, tt[:4]))/(2*eps)
                    single = torch.cat([model(xt[i:i+1], tt[i:i+1]) for i in range(4)])
                expected = jac[:4]@direction
                fd_error = float((fd-expected).abs().max())
                batch_error = float((single-pred[:4]).abs().max())
                assert torch.allclose(fd, expected, atol=2e-6, rtol=2e-6), (row['id'], fd_error)
                assert batch_error < 1e-10, (row['id'], batch_error)
            summary = dict(id=row['id'], dataset=cell['dataset'], family=cell['family'], time=value,
                **metrics(pred.numpy(), jac.numpy(), truth['velocity'], truth['velocity_jacobian'], value))
            time_rows.append(summary)
            for key, array in [('x', point), ('prediction', pred.numpy()), ('jacobian', jac.numpy()),
                ('oracle', truth['velocity']), ('oracle_jacobian', truth['velocity_jacobian']),
                ('oracle_image_jacobian', truth['image_jacobian'])]:
                full[key].append(array)
        full = {k: np.asarray(v) for k, v in full.items()}
        assert all(np.isfinite(v).all() for v in full.values())
        assert snapshot == tensor_hashes(model), row['id']
        filename = row['id']+'.npz'
        np.savez_compressed(out/filename, **full)
        confirmation = study/'analysis/confirmation'/row['id']/'summary.json'
        conf = next(r for r in read(confirmation) if r['step'] == 20000)
        assert conf['checkpoint_sha256'] == identity['checkpoint_sha256']
        identity.update(id=row['id'], dataset=cell['dataset'], family=cell['family'],
            optimizer=cell['optimizer'], projection_lr=cell['projection_lr'],
            raw_file=filename, raw_sha256=sha(out/filename), assets_sha256=bound['sha256'],
            confirmation_summary_sha256=sha(confirmation), confirmation_pooled_sw2=conf['pooled_sw2'],
            native_fp32_max_error=native_error, precision_conversion_max_error=precision_error,
            finite_difference_jvp_max_error=fd_error, batch_separability_max_error=batch_error,
            parameters_and_buffers_unchanged=True, seconds=time.monotonic()-case_start)
        summaries.extend(time_rows)
        identities.append(identity)
        dump(out/'rows.json', dict(rows=summaries, cases=identities))
        dump(out/'progress.json', dict(pid=os.getpid(), completed=index+1, total=12,
                                      seconds=time.monotonic()-started, last=row['id']))
        print(f'{index+1}/12 {row["id"]}', flush=True)
    dump(out/'COMPLETE.json', dict(passed=True, witness=args.witness, cases=12, rows=len(summaries),
        points_per_time=n, times=times, seconds=time.monotonic()-started,
        rows_sha256=sha(out/'rows.json'), anchors_sha256=sha(study/'anchors.json'),
        source_sha256={f.name: sha(f) for f in [Path(__file__), HERE/'geometry.py', HERE/'oracle_check.py']},
        device='cpu', torch=torch.__version__, numpy=np.__version__))


if __name__ == '__main__':
    main()
