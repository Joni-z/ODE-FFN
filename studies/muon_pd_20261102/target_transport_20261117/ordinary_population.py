"""Existing ordinary checkpoints, native sampling, current bound target banks."""
import argparse
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from population_resolution import read, dump, sha


def prepare(a):
    import numpy as np
    assert not a.output.exists()
    old, population = read(a.control_root/'portable.json'), read(a.population_root/'portable.json')
    assert old['passed'] and population['training_seed'] == 0
    selected = [c for c in old['cases'] if c['family'] in ('free_isru', 'swiglu64', 'swiglu192')]
    assert len(selected) == 4
    rows = []
    for index, case in enumerate(selected):
        config = read(a.control_root/case['config'])
        assert config['steps'] == 20000 and config['seeds'] == [0]
        for key in ('config', 'checkpoint', 'dataset_manifest', 'assets', 'cpu_field64', 'cpu_sample64'):
            assert sha(a.control_root/case[key]) == case[key+'_sha256']
        references = [c for c in population['cases'] if c['dataset'] == case['dataset']]
        assert len(references) == 2
        splits = {}
        for split in ('selection', 'confirmation'):
            binding = references[0]['splits'][split]['assets']
            assert binding == references[1]['splits'][split]['assets']
            file = a.population_root/binding['file']
            assert sha(file) == binding['sha256']
            original = binding['original_binding']
            assert sha(Path(original['file'])) == original['sha256']
            with np.load(file) as data, np.load(original['file']) as full:
                assert len(data.files) == 17
                assert all(np.array_equal(data[k], full[k]) for k in data.files)
            splits[split] = dict(population_file=str(file), population_sha256=sha(file),
                full_file=original['file'], full_sha256=original['sha256'], stream=original['stream'])
        rows.append(dict(index=index, case=case, splits=splits,
                         optimizer=config['cell'].get('optimizer'), config_cell=config['cell']))
    dump(a.output, dict(passed=True, control_root=str(a.control_root),
        control_manifest_sha256=sha(a.control_root/'portable.json'),
        population_root=str(a.population_root),
        population_manifest_sha256=sha(a.population_root/'portable.json'), cases=rows,
        training_seed=0, updates=20000, steps=[128, 2048], banks=8, points_per_bank=4096,
        training_runs=0, new_draws=0, source_sha256=sha(__file__)))
    print('ORDINARY_INPUTS_BOUND', len(rows), flush=True)


def dependencies():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    from common import base, ev, torch
    from gpu_resolution import load
    from cross_solver_evaluate import cases
    torch.set_num_threads(1)
    assert torch.get_default_dtype() == torch.float32
    base.verify_sources()
    return base, ev, torch, load, cases


def restore(binding, entry):
    import numpy as np
    base, ev, torch, load, _ = dependencies()
    root, case = Path(binding['control_root']), entry['case']
    assert sha(root/'portable.json') == binding['control_manifest_sha256']
    model = load(root, case, 'cpu').double()
    snapshot = {kind+'::'+name: value.detach().clone()
                for kind, named in [('parameter', model.named_parameters()), ('buffer', model.named_buffers())]
                for name, value in named}
    fingerprints = {key: ev.array_hash(value.numpy()) for key, value in snapshot.items()}
    with np.load(root/case['assets']) as old, torch.no_grad():
        prediction = model(torch.from_numpy(old['field_x'][:512]).double(),
                           torch.from_numpy(old['field_t'][:512]).double()).numpy()
        generated = base.train.generated_samples(model, torch.from_numpy(old['source_0'][:64]).double(), 128).numpy()
    assert np.array_equal(prediction, np.load(root/case['cpu_field64']))
    assert np.array_equal(generated, np.load(root/case['cpu_sample64']))
    return model, snapshot, fingerprints


def unchanged(model, snapshot, rng):
    _, _, torch, _, _ = dependencies()
    values = {kind+'::'+name: value for kind, named in
              [('parameter', model.named_parameters()), ('buffer', model.named_buffers())]
              for name, value in named}
    assert set(values) == set(snapshot)
    assert all(torch.equal(values[k], value) for k, value in snapshot.items())
    assert torch.equal(torch.get_rng_state(), rng)


def inputs(binding, entry, split):
    import numpy as np
    root = Path(binding['population_root'])
    assert sha(root/'portable.json') == binding['population_manifest_sha256']
    row = entry['splits'][split]
    assert sha(Path(row['population_file'])) == row['population_sha256']
    assert sha(Path(row['full_file'])) == row['full_sha256']
    with np.load(row['full_file']) as full:
        return {k: full[k].copy() for k in full.files}


def check(a):
    import numpy as np
    assert not a.output.exists()
    binding = read(a.binding)
    base, ev, torch, _, _ = dependencies()
    rows = []
    for entry in binding['cases']:
        model, snapshot, fingerprints = restore(binding, entry)
        rng = torch.get_rng_state().clone()
        for split in ('selection', 'confirmation'):
            data = inputs(binding, entry, split)
            source = torch.from_numpy(data['source_0'][:2]).double()
            with torch.no_grad():
                for steps in (128, 2048):
                    direct = base.train.generated_samples(model, source, steps)
                    recorded, path = ev.sample(model, None, source, steps, record=True)
                    assert torch.equal(direct, recorded) and path.shape == (steps+1, 2, source.shape[1])
                    assert torch.isfinite(direct).all()
        unchanged(model, snapshot, rng)
        rows.append(dict(index=entry['index'], id=entry['case']['id'],
                         exact_old_field_and_sample_anchors=True,
                         native_recording_both_meshes_exact=True, fingerprints=fingerprints,
                         parameters=model.parameter_count()))
        print('ORDINARY_CHECK', entry['case']['id'], flush=True)
    # Benchmark the widest retained ordinary model at the formal batch size.
    entry = next(c for c in binding['cases'] if c['case']['family'] == 'swiglu192')
    model, snapshot, _ = restore(binding, entry)
    rng = torch.get_rng_state().clone()
    data = inputs(binding, entry, 'selection')
    began = time.perf_counter()
    with torch.no_grad():
        sample = base.train.generated_samples(model, torch.from_numpy(data['source_0']).double(), 128)
    seconds = time.perf_counter()-began
    assert sample.shape == (4096, 16) and torch.isfinite(sample).all()
    unchanged(model, snapshot, rng)
    dump(a.output, dict(passed=True, rows=rows, binding_sha256=sha(a.binding),
        source_sha256=sha(__file__), device='cpu', threads=1, torch=torch.__version__,
        benchmark=dict(family='swiglu192', points=4096, steps=128, seconds=seconds,
                       peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)))
    print('ORDINARY_READINESS_PASS', seconds, flush=True)


def run(a):
    import numpy as np
    assert not a.output.exists()
    binding = read(a.binding)
    base, _, torch, _, cases = dependencies()
    ready = read(a.readiness)
    assert ready['passed'] and ready['binding_sha256'] == sha(a.binding)
    assert ready['source_sha256'] == sha(__file__)
    entry = binding['cases'][a.index]
    assert entry['index'] == a.index
    model, snapshot, fingerprints = restore(binding, entry)
    assert fingerprints == ready['rows'][a.index]['fingerprints']
    rng = torch.get_rng_state().clone()
    spec = read(Path(binding['control_root'])/entry['case']['dataset_manifest'])
    law = base.train.load_distribution(spec['datasets'][entry['case']['dataset']])
    a.output.mkdir(parents=True)
    rows = []
    began = time.monotonic()
    count, banks = (2, 1) if a.witness else (4096, 8)
    for split in ('selection', 'confirmation'):
        data = inputs(binding, entry, split)
        directions = torch.from_numpy(data['directions']).double()
        with torch.no_grad():
            x, t, oracle = (torch.from_numpy(data['field_'+k]).double() for k in ('x', 't', 'oracle'))
            prediction = model(x, t)
            field_mse = float((prediction-oracle).square().mean())
        for steps in (128, 2048):
            values, targets, scores, timings = [], [], [], []
            for bank in range(banks):
                dump(a.output/'progress.json', dict(pid=os.getpid(), index=a.index, split=split,
                    steps=steps, next_bank=bank, total_banks=banks, seconds=time.monotonic()-began))
                source = torch.from_numpy(data[f'source_{bank}'][:count]).double()
                target = torch.from_numpy(data[f'target_{bank}'][:count]).double()
                started = time.perf_counter()
                with torch.no_grad():
                    value = base.train.generated_samples(model, source, steps)
                timings.append(time.perf_counter()-started)
                assert torch.isfinite(value).all()
                scores.append(base.train.sliced_w2(value, target, directions))
                values.append(value.numpy()); targets.append(target.numpy())
                print('ORDINARY_BANK', a.index, split, steps, bank, timings[-1], flush=True)
            pooled = np.concatenate(values)
            pooled_sw2 = base.train.sliced_w2(torch.from_numpy(pooled),
                torch.from_numpy(np.concatenate(targets)), directions)
            file = a.output/f'{split}-heun{steps}.npz'
            np.savez_compressed(file, samples=np.stack(values), field=prediction.numpy())
            rows.append(dict(case=entry['case']['id'], split=split, steps=steps,
                points_per_bank=count, banks=banks, field_mse=field_mse,
                pooled_sw2=pooled_sw2, bank_sw2=scores, bank_seconds=timings,
                sample_stats=cases(pooled, law), file=file.name, sha256=sha(file)))
            dump(a.output/'rows.json', rows)
    unchanged(model, snapshot, rng)
    dump(a.output/'COMPLETE.json', dict(passed=True, witness=a.witness, index=a.index,
        rows=len(rows), binding_sha256=sha(a.binding), readiness_sha256=sha(a.readiness),
        case=entry['case'], fingerprints=fingerprints, parameters=model.parameter_count(),
        parameters_buffers_rng_unchanged=True, source_sha256=sha(__file__),
        seconds=time.monotonic()-began, device='cpu', threads=1,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print('ORDINARY_CASE_COMPLETE', a.index, flush=True)


def dispatch(a):
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    script = Path(__file__).resolve()
    freeze = read(script.parents[3].parent/'source-freeze.json')
    repo = script.parents[3]
    assert all(sha(repo/p) == h for p, h in freeze['files'].items())
    pending, active, completed, failed = list(range(4)), {}, [], False
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    while pending or active:
        for pid, (process, log, row) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close(); del active[pid]
            row.update(actual_terminal_exit=code)
            completed.append(row)
            if code != 0:
                failed = True
            elif row['stage'] == 'evaluate':
                command = [a.python, '-u', str(script.with_name('verify_ordinary_population.py')),
                    '--binding', str(a.binding), '--input', row['output'],
                    '--output', str(Path(row['output'])/'independent.json')]
                stream = (a.output/f"index{row['index']}-independent.log").open('x')
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                    stderr=stream, env=environment)
                active[child.pid] = child, stream, dict(index=row['index'], stage='verify',
                    pid=child.pid, output=row['output'], command=command)
        while pending and len(active) < 2 and not failed:
            index = pending.pop(0); folder = a.output/f'index{index}'
            command = [a.python, '-u', str(script), '--mode', 'run', '--binding', str(a.binding),
                '--readiness', str(a.readiness), '--index', str(index), '--output', str(folder)]
            stream = (a.output/f'index{index}.log').open('x')
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=stream, env=environment)
            active[child.pid] = child, stream, dict(index=index, stage='evaluate', pid=child.pid,
                output=str(folder), command=command)
        dump(a.output/'progress.json', dict(pid=os.getpid(), unix_time=time.time(),
            running=[r for _, _, r in active.values()], completed=completed,
            pending=pending, failed=failed))
        if failed and not active:
            break
        if pending or active:
            time.sleep(5)
    assert all(sha(repo/p) == h for p, h in freeze['files'].items())
    result = dict(passed=not failed and not pending and len(completed) == 8,
        completed=completed, unstarted=pending, source_freeze_sha256=sha(repo.parent/'source-freeze.json'))
    dump(a.output/'TERMINAL.json', result)
    raise SystemExit(0 if result['passed'] else 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('prepare', 'check', 'run', 'dispatch'), required=True)
    for name in ('output', 'binding', 'readiness', 'control-root', 'population-root'):
        parser.add_argument('--'+name, type=Path, required=name == 'output')
    parser.add_argument('--index', type=int, choices=range(4))
    parser.add_argument('--witness', action='store_true')
    parser.add_argument('--python', default=sys.executable)
    a = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    {'prepare': prepare, 'check': check, 'run': run, 'dispatch': dispatch}[a.mode](a)


if __name__ == '__main__':
    main()
