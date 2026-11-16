"""Refine the registered sensitivity test without changing weights or sources."""
import argparse
import os
from pathlib import Path
import time
import numpy as np
from geometry import read, dump, sha
from frozen_probe import restore, tensor_hashes, assets, torch, base
from native_time import pilot_folder, packed_blends
from trajectory import OracleResponse, SensitivityField, integrate, qualify, response_summary
from trajectory_probe import bank_summaries


def subset(data: dict, per_bank: int) -> dict:
    indices = np.concatenate([np.arange(per_bank)+32*b for b in range(8)])
    result = dict(data)
    for name in ['source', 'oracle_endpoint', 'tangent', 'bank_index']:
        result[name] = data[name][indices]
    for name in ['oracle_path', 'tangent_path', 'endpoints']:
        result[name] = data[name][:, indices]
    result['paths'] = data['paths'][:, :, indices]
    return result


def packed_observe(model, oracle, source, steps: int) -> dict:
    augmented, path = integrate(SensitivityField(model, oracle),
                                torch.cat((source, torch.zeros_like(source)), dim=1), steps)
    dim = source.shape[1]
    endpoints, paths = packed_blends(model, oracle, source, steps)
    return dict(steps=np.asarray(steps), lambdas=np.asarray([.125, .25, .5, 1.]),
        oracle_endpoint=augmented[:, :dim].numpy(), tangent=augmented[:, dim:].numpy(),
        oracle_path=path[:, :, :dim].numpy(), tangent_path=path[:, :, dim:].numpy(),
        endpoints=endpoints, paths=paths, source=source.numpy(),
        bank_index=np.repeat(np.arange(8), len(source)//8), record_times=np.linspace(0, 1, 9))


def main() -> None:
    p = argparse.ArgumentParser()
    for key in ['study', 'pilot', 'native', 'data-root', 'case', 'output']:
        p.add_argument('--'+key, required=True)
    p.add_argument('--witness', action='store_true')
    a = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    study, pilot, native, data_root, out = map(Path, [a.study, a.pilot, a.native, a.data_root, a.output])
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    base.verify_sources()
    anchors = read(study/'anchors.json')
    row = next(r for r in anchors['references']+anchors['new_cases'] if r['id'] == a.case)
    model, law, cell, evaluation, identity = restore(study, anchors, row, data_root)
    model.double()
    snapshot = tensor_hashes(model)
    arrays, _, source_binding = assets(anchors['assets'], cell['dataset'], 'selection', 'cpu')
    count = 1 if a.witness else 32
    source = np.concatenate([arrays[f'source_{b}'][:count] for b in range(8)]).astype(float)
    source_tensor = torch.from_numpy(source)
    native_complete = read(native/'COMPLETE.json')
    assert native_complete['passed'] and native_complete['cases'] == 12
    assert native_complete['summary_sha256'] == sha(native/'summary.json')
    native_summary = read(native/'summary.json')
    assert native_summary['qualified_cases'] == 12
    binding = next(r for r in native_summary['bindings'] if r['id'] == a.case)
    native_case = Path(binding['native_folder'])
    assert sha(native_case/'COMPLETE.json') == binding['native_complete_sha256']
    assert sha(native_case/'rows.json') == binding['native_rows_sha256']
    native_rows = read(native_case/'rows.json')
    assert native_rows['checkpoint_sha256'] == identity['checkpoint_sha256']
    original = pilot_folder(pilot, cell['dataset'])
    original_complete = read(original/'COMPLETE.json')
    assert original_complete['passed'] and not original_complete['witness']
    assert sha(original/'rows.json') == original_complete['rows_sha256']
    verification = original.with_name(original.name+'-verified')
    verified_complete = read(verification/'COMPLETE.json')
    assert verified_complete['passed'] and not verified_complete['witness']
    assert sha(verification/'summary.json') == verified_complete['summary_sha256']
    assert read(verification/'summary.json')['input_complete_sha256'] == sha(original/'COMPLETE.json')
    original_rows = read(original/'rows.json')
    old_row = next(r for r in original_rows['rows'] if r['id'] == a.case)
    old_identity = next(r for r in original_rows['identities'] if r['id'] == a.case)
    assert old_identity['checkpoint_sha256'] == identity['checkpoint_sha256']
    old_level = next(r for r in old_row['levels'] if r['steps'] == 1024)
    old_file = original/old_level['file']
    assert sha(old_file) == old_level['sha256']
    with np.load(old_file) as archive:
        previous = subset({k: archive[k].copy() for k in archive.files}, count)
    assert np.array_equal(previous['source'], source)
    np.savez_compressed(out/'heun1024.npz', **previous)
    levels = [dict(steps=1024, reused=True, file='heun1024.npz', sha256=sha(out/'heun1024.npz'),
        **response_summary(previous), banks=bank_summaries(previous, count), qualification=None,
        original_file=str(old_file), original_sha256=sha(old_file), actual_new_field_calls=0)]
    oracle = OracleResponse(law)
    started = time.monotonic()
    dump(out/'RUNNING.json', dict(pid=os.getpid(), id=a.case, witness=a.witness,
        sources=len(source), started_unix=time.time()))
    qualified = False
    for steps in [2048, 4096, 8192]:
        before = time.monotonic()
        result = packed_observe(model, oracle, source_tensor, steps)
        native_level = next((r for r in native_rows['levels'] if r['steps'] == steps), None)
        endpoint_exact = path_exact = None
        if native_level is not None:
            native_file = native_case/native_level['file']
            assert sha(native_file) == native_level['sha256']
            index = np.concatenate([np.arange(count)+32*b for b in range(8)])
            with np.load(native_file) as archive:
                endpoint_exact = np.array_equal(result['endpoints'][-1], archive['endpoint'][index])
                path_exact = np.array_equal(result['paths'][-1], archive['path'][:, index])
            assert endpoint_exact and path_exact, 'Packed lambda=1 changed the native map'
        file = f'heun{steps}.npz'
        np.savez_compressed(out/file, **result)
        gate = qualify(previous, result)
        qualified = gate['passed']
        entry = dict(steps=steps, reused=False, file=file, sha256=sha(out/file),
            **response_summary(result), banks=bank_summaries(result, count), qualification=gate,
            seconds=time.monotonic()-before, lambda1_native_endpoint_exact=endpoint_exact,
            lambda1_native_path_exact=path_exact, actual_new_augmented_stage_calls=2*steps,
            actual_new_packed_stage_calls=2*steps)
        levels.append(entry)
        dump(out/'progress.json', dict(pid=os.getpid(), id=a.case, last_steps=steps,
            qualification=gate, elapsed_seconds=time.monotonic()-started))
        print(f'{a.case} refined Heun{steps}: qualified={qualified}', flush=True)
        previous = result
        if steps >= 4096 and qualified:
            break
    assert snapshot == tensor_hashes(model)
    dump(out/'rows.json', dict(id=a.case, dataset=cell['dataset'], family=cell['family'],
        qualified=bool(qualified), final_steps=int(previous['steps']), levels=levels,
        **response_summary(previous), banks=bank_summaries(previous, count), **identity,
        assets_file=source_binding['file'], assets_sha256=source_binding['sha256'],
        native_qualification_complete=str(native/'COMPLETE.json'),
        native_qualification_complete_sha256=sha(native/'COMPLETE.json'),
        parameters_and_buffers_unchanged=True))
    dump(out/'COMPLETE.json', dict(passed=True, witness=a.witness, id=a.case, sources=len(source), banks=8,
        qualified=bool(qualified), final_steps=int(previous['steps']), rows_sha256=sha(out/'rows.json'),
        anchors_sha256=sha(study/'anchors.json'), seconds=time.monotonic()-started,
        original_pilot_complete_sha256=sha(original/'COMPLETE.json'),
        source_sha256={f.name: sha(f) for f in [Path(__file__), Path(__file__).with_name('native_time.py'),
                                             Path(__file__).with_name('trajectory.py')]},
        device='cpu', torch=torch.__version__))


if __name__ == '__main__':
    main()
