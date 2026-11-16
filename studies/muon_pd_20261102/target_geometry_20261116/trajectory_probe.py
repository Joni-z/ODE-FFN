"""Bounded twelve-checkpoint calibration with registered numerical refinement."""
import argparse
import os
from pathlib import Path
import time
import numpy as np
from geometry import read, dump, sha
from frozen_probe import restore, tensor_hashes, assets, torch, base
from trajectory import OracleResponse, BlendedField, integrate, observe, response_summary, qualify, RESOLUTIONS


def bank_summaries(result: dict, size: int) -> list[dict]:
    rows = []
    for bank in range(8):
        start, stop = bank*size, (bank+1)*size
        subset = dict(result, endpoints=result['endpoints'][:, start:stop],
                      oracle_endpoint=result['oracle_endpoint'][start:stop], tangent=result['tangent'][start:stop])
        rows.append(dict(bank=bank, **response_summary(subset)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    for key in ['study', 'data-root', 'output', 'dataset']:
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--witness', action='store_true')
    args = parser.parse_args()
    assert args.dataset in ['gmm_ring8_2d', 'rotated_product_bimodal_16d']
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    study, data_root, out = map(Path, [args.study, args.data_root, args.output])
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    base.verify_sources()
    anchors = read(study/'anchors.json')
    rows = [r for r in anchors['references']+anchors['new_cases'] if r['dataset'] == args.dataset]
    assert len(rows) == 6
    count = 1 if args.witness else 32
    arrays, _, binding = assets(anchors['assets'], args.dataset, 'selection', 'cpu')
    source = torch.from_numpy(np.concatenate([arrays[f'source_{b}'][:count] for b in range(8)]))
    dump(out/'RUNNING.json', dict(pid=os.getpid(), dataset=args.dataset, cases=6,
        witness=args.witness, sources=8*count, started_unix=time.time()))
    summaries, identities = [], []
    started = time.monotonic()
    for index, row in enumerate(rows):
        model, law, cell, evaluation, identity = restore(study, anchors, row, data_root)
        saved = np.concatenate([np.load(evaluation/f'20000-b{b}.npy')[:count] for b in range(8)])
        fp32 = base.train.generated_samples(model, source, 128)
        native_error = float(np.max(np.abs(fp32.numpy()-saved)))
        assert np.allclose(fp32.numpy(), saved, atol=1e-4, rtol=1e-4), (row['id'], native_error)
        model.double()
        source64 = source.double()
        oracle = OracleResponse(law)
        snapshot = tensor_hashes(model)
        native64 = base.train.generated_samples(model, source64, 128).numpy()
        precision_error = float(np.max(np.abs(native64-fp32.double().numpy())))
        assert np.allclose(native64, fp32.double().numpy(), atol=1e-3, rtol=1e-3), (row['id'], precision_error)
        levels, previous = [], None
        for steps in RESOLUTIONS:
            before = time.monotonic()
            result = observe(model, oracle, source64, steps)
            if steps == 128:
                assert np.array_equal(result['endpoints'][-1], native64)
            result.update(source=source64.numpy(), bank_index=np.repeat(np.arange(8), count),
                          record_times=np.linspace(0, 1, 9))
            file = row['id']+f'__heun{steps}.npz'
            np.savez_compressed(out/file, **result)
            check = qualify(previous, result) if previous is not None else None
            entry = dict(steps=steps, file=file, sha256=sha(out/file),
                **response_summary(result), banks=bank_summaries(result, count),
                qualification=check, seconds=time.monotonic()-before)
            levels.append(entry)
            dump(out/'progress.json', dict(pid=os.getpid(), completed=index, total=6,
                current=row['id'], last_steps=steps, last_qualification=check,
                seconds=time.monotonic()-started))
            print(f'{index+1}/6 {row["id"]} Heun{steps}', flush=True)
            previous = result
            if check is not None and check['passed']:
                break
        assert snapshot == tensor_hashes(model)
        qualified = bool(levels[-1]['qualification']['passed'])
        summaries.append(dict(id=row['id'], dataset=args.dataset, family=cell['family'],
            qualified=qualified, final_steps=levels[-1]['steps'], levels=levels,
            **response_summary(previous), banks=bank_summaries(previous, count)))
        identities.append(dict(id=row['id'], **identity, assets_file=binding['file'],
            assets_sha256=binding['sha256'], native_fp32_endpoint_max_error=native_error,
            precision_conversion_max_error=precision_error, lambda1_native_endpoint_exact=True,
            parameters_and_buffers_unchanged=True, optimizer=cell['optimizer'], projection_lr=cell['projection_lr']))
        dump(out/'rows.json', dict(rows=summaries, identities=identities))
    dump(out/'COMPLETE.json', dict(passed=True, witness=args.witness, cases=6,
        sources_per_case=8*count, banks=8, dataset=args.dataset,
        all_qualified=all(r['qualified'] for r in summaries),
        unqualified=[r['id'] for r in summaries if not r['qualified']],
        rows_sha256=sha(out/'rows.json'), anchors_sha256=sha(study/'anchors.json'),
        source_sha256={p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name('trajectory.py')]},
        seconds=time.monotonic()-started, torch=torch.__version__, numpy=np.__version__))


if __name__ == '__main__':
    main()
