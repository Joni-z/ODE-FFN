"""Registered native-field grid sensitivity and high-resolution qualification."""
import argparse
import os
from pathlib import Path
import time
import numpy as np
from geometry import read, dump, sha
from frozen_probe import restore, tensor_hashes, assets, torch, base
from trajectory import integrate
from native_time import NEW_LEVELS, cached_level, change


def main() -> None:
    p = argparse.ArgumentParser()
    for key in ['study', 'pilot', 'data-root', 'case', 'output']:
        p.add_argument('--'+key, required=True)
    p.add_argument('--witness', action='store_true')
    args = p.parse_args()
    study, pilot, data_root, out = map(Path, [args.study, args.pilot, args.data_root, args.output])
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    base.verify_sources()
    anchors = read(study/'anchors.json')
    row = next(r for r in anchors['references']+anchors['new_cases'] if r['id'] == args.case)
    model, law, cell, evaluation, identity = restore(study, anchors, row, data_root)
    model.double()
    snapshot = tensor_hashes(model)
    arrays, _, source_binding = assets(anchors['assets'], cell['dataset'], 'selection', 'cpu')
    count = 1 if args.witness else 32
    source = np.concatenate([arrays[f'source_{b}'][:count] for b in range(8)]).astype(float)
    tensor = torch.from_numpy(source)
    started = time.monotonic()
    dump(out/'RUNNING.json', dict(pid=os.getpid(), id=args.case, sources=len(source), witness=args.witness, started_unix=time.time()))
    levels, values = [], {}
    for steps in [128, 1024]:
        value, binding = cached_level(pilot, row, identity, arrays, steps, count)
        assert np.array_equal(value['source'], source)
        file = f'heun{steps}.npz'
        np.savez_compressed(out/file, **value, bank_index=np.repeat(np.arange(8), count))
        levels.append(dict(steps=steps, reused=True, binding=binding, file=file, sha256=sha(out/file), actual_new_field_calls=0))
        values[steps] = value
    qualified = False
    for steps in NEW_LEVELS:
        before = time.monotonic()
        endpoint, path = integrate(model, tensor, steps)
        value = dict(source=source, endpoint=endpoint.numpy(), path=path.numpy(), steps=np.asarray(steps))
        file = f'heun{steps}.npz'
        np.savez_compressed(out/file, **value, bank_index=np.repeat(np.arange(8), count))
        entry = dict(steps=steps, reused=False, file=file, sha256=sha(out/file),
                     seconds=time.monotonic()-before, actual_new_field_calls=2*steps)
        if steps in [4096, 8192]:
            entry['qualification'] = change(values[steps//2]['endpoint'], value['endpoint'])
            qualified = entry['qualification']['componentwise_pass']
        levels.append(entry)
        values[steps] = value
        dump(out/'progress.json', dict(pid=os.getpid(), id=args.case, last_steps=steps,
            completed_new_levels=sum(not l['reused'] for l in levels), elapsed_seconds=time.monotonic()-started,
            qualification=entry.get('qualification')))
        print(f'{args.case} native Heun{steps}', flush=True)
        if steps >= 4096 and qualified:
            break
    assert snapshot == tensor_hashes(model)
    contrasts = []
    pairs = [(1000, 1024), (1032, 1024), (1000, 1032), (128, max(values)), (1024, max(values))]
    for left, right in pairs:
        delta = values[left]['endpoint']-values[right]['endpoint']
        contrasts.append(dict(left=left, right=right, **change(values[left]['endpoint'], values[right]['endpoint']),
            bank_rms=[float(np.sqrt(np.mean(delta[b*count:(b+1)*count]**2))) for b in range(8)]))
    dump(out/'rows.json', dict(id=args.case, dataset=cell['dataset'], family=cell['family'],
        qualified=qualified, final_steps=max(values), levels=levels, contrasts=contrasts,
        assets_file=source_binding['file'], assets_sha256=source_binding['sha256'],
        parameters_and_buffers_unchanged=True, **identity))
    dump(out/'COMPLETE.json', dict(passed=True, witness=args.witness, id=args.case, sources=len(source), banks=8,
        qualified=qualified, final_steps=max(values), rows_sha256=sha(out/'rows.json'),
        seconds=time.monotonic()-started, source_sha256={f.name: sha(f) for f in [Path(__file__), Path(__file__).with_name('native_time.py')]},
        anchors_sha256=sha(study/'anchors.json'), device='cpu', torch=torch.__version__))


if __name__ == '__main__':
    main()
