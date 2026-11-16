"""Same-process CPU timing and native dense work for all ten fixed checkpoints."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from sharing_cost import DenseMACs, summary
from contrast_trial import base, torch, build_model
from contrast_evaluate import evaluator


def main() -> None:
    p = argparse.ArgumentParser()
    for key in ['study', 'data-root', 'output']:
        p.add_argument('--'+key, required=True)
    a = p.parse_args()
    study, out = Path(a.study), Path(a.output)
    assert not out.exists()
    torch.set_num_threads(1)
    evaluator.ev.build_model = build_model
    anchors = json.loads((study/'anchors.json').read_text())
    schedule = anchors.get('protocol') == 'schedule_order'
    if schedule:
        from gpu_resolution import builder
        evaluator.ev.build_model = builder
    loaded, rows = [], []
    for case in anchors['references']+anchors['new_cases']:
        folder = Path(case['folder'])
        cfg = json.loads((folder/'config.json').read_text())
        assert (folder/'ADAPTER_COMPLETE').exists()
        cache = base.verify_data(a.data_root, cfg['cell'])
        model, identity = evaluator.ev.restore(folder, cfg, cache/'manifest.json', 20000, 'cpu')
        arrays, fixed, binding = evaluator.assets(anchors['assets'], case['dataset'], 'selection', 'cpu')
        x, t = fixed['x'][:512], fixed['t'][:512]
        source = torch.from_numpy(arrays['source_0'][:64])
        with torch.no_grad():
            with DenseMACs() as counter:
                model(x, t)
            for _ in range(5):
                model(x, t)
        assert counter.macs % 512 == 0
        rows.append(dict(id=case['id'], dataset=case['dataset'], family=cfg['cell']['family'],
            parameters=model.parameter_count(), dense_macs_per_example=counter.macs//512,
            forward_batch=512, sample_batch=64, checkpoint=identity, assets_sha256=binding['sha256'],
            forward_cpu_seconds=[], forward_wall_seconds=[], sample_cpu_seconds=[]))
        loaded.append((model, x, t, source, evaluator.ev.tensor_snapshot(model)))
    expected = 12 if schedule else 10
    assert len(rows) == expected
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for _ in range(9):
            for i in rng.permutation(len(rows)):
                model, x, t, _, _ = loaded[i]
                cpu, wall = time.process_time(), time.perf_counter()
                for _ in range(10):
                    model(x, t)
                rows[i]['forward_cpu_seconds'].append((time.process_time()-cpu)/10)
                rows[i]['forward_wall_seconds'].append((time.perf_counter()-wall)/10)
        for repeat in range(3):
            for i in rng.permutation(len(rows)):
                model, _, _, source, _ = loaded[i]
                start = time.process_time()
                value = base.train.generated_samples(model, source, 128)
                assert torch.isfinite(value).all()
                rows[i]['sample_cpu_seconds'].append(time.process_time()-start)
            base.dump(out.with_name(out.name+'-progress.json'), dict(sample_rounds=repeat+1, total=3))
    for row, (model, _, _, _, snapshot) in zip(rows, loaded):
        evaluator.ev.same_tensors(model, snapshot)
        for key in ['forward_cpu_seconds', 'forward_wall_seconds', 'sample_cpu_seconds']:
            row[key] = summary(row[key])
    for dataset in {r['dataset'] for r in rows}:
        families = (['state_midpoint4', 'palindrome_operator_midpoint4', 'alternating_operator_midpoint4']
                    if schedule else ['state_midpoint4', 'stage_operator_midpoint4', 'contrast_operator_midpoint4'])
        focal = [r for r in rows if r['dataset'] == dataset and r['family'] in families]
        assert len(focal) == 3 and len({r['dense_macs_per_example'] for r in focal}) == 1
        assert focal[1]['parameters'] == focal[2]['parameters']
    base.dump(out, dict(passed=True, scope='same-process one-thread CPU; dense MACs exclude pointwise work',
        forward_repetitions=9, forwards_per_repetition=10, sampling_repetitions=3, sampling_heun_steps=128,
        environment=dict(torch=torch.__version__, threads=torch.get_num_threads()), rows=rows,
        anchors_sha256=base.digest(study/'anchors.json'), source_sha256=base.digest(__file__)))
    print(json.dumps(dict(passed=True, cases=expected, output=str(out))))


if __name__ == '__main__':
    main()
