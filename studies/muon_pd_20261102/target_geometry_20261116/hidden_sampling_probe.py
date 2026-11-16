"""Fixed-first-layer interventions through the unchanged native Heun sampler."""
import argparse
import os
import shutil
from pathlib import Path
import time
import numpy as np
from geometry import read, dump, sha
from frozen_probe import restore, tensor_hashes, assets, torch, base
from hidden_energy import reference, TOLERANCES
from hidden_readout_probe import candidates, replace_one, MODES
from check_hidden_energy import potential

FOCAL = ('state_midpoint4', 'palindrome_operator_midpoint4', 'alternating_operator_midpoint4')
ORDINARY = ('free_isru', 'swiglu64', 'swiglu192')
FIRST = 'blocks.0.ffn.ode1'


class Intervention:
    def __init__(self, model, mode: str, steps: int, outer: int) -> None:
        self.model, self.mode, self.steps = model, mode, steps
        self.layer = dict(model.ode_layers())[FIRST]
        self.call = 0
        self.capture_at = {0, 64, 128, 192, 2*outer-1}
        self.snapshots = []

    def __call__(self, x, t):
        captured = {}
        def hook(module, args, output):
            native = output[0] if isinstance(output, tuple) else output
            if self.mode == 'identity':
                refined, local = native, native
            else:
                refined = module.solve(args[0], self.steps)
                local = torch.from_numpy(candidates(native.numpy(), refined.numpy())['local_outputs'][MODES.index(self.mode)])
            if self.call in self.capture_at:
                captured.update(x=x.numpy().copy(), time=t.numpy().copy(), entrance=args[0].numpy().copy(),
                                native=native.numpy().copy(), refined=refined.numpy().copy(), local=local.numpy().copy())
            return (local, *output[1:]) if isinstance(output, tuple) else local
        handle = self.layer.register_forward_hook(hook)
        try:
            output = self.model(x, t)
        finally:
            handle.remove()
        if captured:
            captured['field'] = output.numpy().copy()
            captured['call'] = self.call
            self.snapshots.append(captured)
        self.call += 1
        return output


def qualify_snapshots(model, intervention: Intervention) -> tuple:
    layer = intervention.layer
    p = layer.operator()
    weights = (layer.stage_weights() if hasattr(layer, 'stage_weights') else torch.stack([p]*4)).detach().numpy()
    records, reference_failures, local_failures, output_failures = [], 0, 0, 0
    for snapshot in intervention.snapshots:
        h, native = snapshot['entrance'], snapshot['native']
        refs = [reference(h, weights, rtol, atol) for rtol, atol in TOLERANCES]
        ys, work = np.stack([value[0] for value in refs]), np.stack([value[1] for value in refs])
        closure = potential(ys[-1])-potential(h)-work[-1]
        qualified = np.all(np.abs(ys[0]-ys[1]) <= 1e-9+1e-9*np.abs(ys[1]), axis=1)
        qualified &= np.abs(work[0]-work[1]) <= 1e-8*(1+np.abs(work[1]))
        qualified &= np.abs(closure) <= 1e-8*(1+np.abs(potential(h))+np.abs(potential(ys[1]))+np.abs(work[1]))
        dop = candidates(native, ys[-1])
        inputs = dict(x=snapshot['x'], time=snapshot['time'])
        fields = np.stack([replace_one(model, layer, inputs, h, native, value) for value in dop['local_outputs']])
        index = MODES.index(intervention.mode)
        local_ratio = np.linalg.norm(snapshot['local']-dop['local_outputs'][index], axis=-1)/(1e-9+1e-3*np.linalg.norm(native-ys[-1], axis=-1))
        output_ratio = np.linalg.norm(snapshot['field']-fields[index], axis=-1)/(1e-10+1e-3*np.linalg.norm(fields[0]-fields[1], axis=-1))
        reference_failures += int((~qualified).sum())
        local_failures += int((local_ratio > 1).sum())
        output_failures += int((output_ratio > 1).sum())
        records.append(dict(**snapshot, references=ys, reference_work=work, reference_qualified=qualified,
                            dop_local=dop['local_outputs'], dop_fields=fields,
                            local_limit_ratio=local_ratio, output_limit_ratio=output_ratio))
    arrays = {('snapshot_'+key): np.stack([record[key] for record in records]) for key in records[0]}
    arrays['weights'] = weights
    metrics = dict(reference_failures=reference_failures, local_failures=local_failures,
        output_failures=output_failures, captured_batches=len(records),
        captured_points=sum(len(record['entrance']) for record in records),
        local_max_limit_ratio=float(max(record['local_limit_ratio'].max() for record in records)),
        output_max_limit_ratio=float(max(record['output_limit_ratio'].max() for record in records)),
        snapshot_qualified=reference_failures == local_failures == output_failures == 0)
    return arrays, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    for key in ['study', 'surrogate', 'data-root', 'output']:
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--dataset', choices=['gmm_ring8_2d', 'rotated_product_bimodal_16d'])
    parser.add_argument('--witness', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--extend1024-from', help='Reuse bound native/512 arms from the original pilot')
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    torch.set_num_threads(1)
    study, surrogate, data_root, out = map(Path, [args.study, args.surrogate, args.data_root, args.output])
    out.mkdir(parents=True, exist_ok=False)
    base.verify_sources()
    terminal = read(surrogate/'terminal-receipts.json')
    assert terminal['passed'] and all(value == 0 for value in terminal['native_exits'].values())
    assert all(value == 0 for value in terminal['independent_exits'].values())
    anchors = read(study/'anchors.json')
    previous = Path(args.extend1024_from) if args.extend1024_from else None
    inner_levels = (512, 1024) if previous else (256, 512)
    if previous:
        prior_terminal = read(previous/'terminal-receipts.json')
        assert prior_terminal['passed'] and all(value == 0 for value in prior_terminal['native_exits'].values())
        assert all(value == 0 for value in prior_terminal['independent_exits'].values())
    cases = []
    for row in anchors['references']+anchors['new_cases']:
        cell = read(Path(row['folder'])/'config.json')['cell']
        if args.dataset and row['dataset'] != args.dataset:
            continue
        if cell['family'] not in FOCAL+ORDINARY:
            continue
        if args.check_only and cell['family'] not in FOCAL:
            continue
        if args.witness and cell['family'] in FOCAL and cell['family'] != 'state_midpoint4':
            continue
        cases.append((row, cell))
    results, started = [], time.monotonic()
    dump(out/'RUNNING.json', dict(pid=os.getpid(), cases=len(cases), witness=args.witness, check_only=args.check_only))
    for row, cell in cases:
        model, law, _, evaluation, identity = restore(study, anchors, row, data_root)
        data, _, binding = assets(anchors['assets'], cell['dataset'], 'selection', 'cpu')
        with torch.no_grad():
            pred = model(torch.from_numpy(data['field_x'][:128]), torch.from_numpy(data['field_t'][:128])).numpy()
        assert np.allclose(pred, np.load(evaluation/'field-20000.npy')[:128], atol=2e-5, rtol=2e-5)
        model.double()
        before = tensor_hashes(model)
        folder = out/row['id']
        folder.mkdir()
        task = 'ring' if row['dataset'] == 'gmm_ring8_2d' else 'product'
        focal = cell['family'] in FOCAL
        if focal:
            source_root = surrogate/'formal'/task
            original = next(item for item in read(source_root/'rows.json')['rows'] if item['id'] == row['id'])
            level = next(item for item in original['levels'] if item['steps'] == 256)
            first = level['layers'][0]
            assert first['layer'] == FIRST and first['qualified']
            assert sha(source_root/first['file']) == first['sha256']
        if args.check_only:
            with np.load(original['source_inputs']) as bundle:
                inputs = {key: bundle[key].copy() for key in bundle.files}
            with np.load(source_root/first['file']) as bundle:
                expected_local, expected_fields = bundle['local_outputs'].copy(), bundle['fields'].copy()
            for index, mode in enumerate(MODES):
                field = Intervention(model, mode, 256, 128)
                with torch.no_grad():
                    actual = field(torch.from_numpy(inputs['x']), torch.from_numpy(inputs['time'])).numpy()
                assert np.array_equal(actual, expected_fields[index]), (row['id'], mode)
                assert np.array_equal(field.snapshots[0]['local'], expected_local[index])
            results.append(dict(id=row['id'], dataset=cell['dataset'], family=cell['family'],
                native_candidate_and_output_exact=True, inputs=80))
        else:
            per_bank = 1 if args.witness else 8
            source = np.concatenate([data[f'source_{bank}'][:per_bank] for bank in range(8)]).astype(float)
            target = np.concatenate([data[f'target_{bank}'][:per_bank] for bank in range(8)]).astype(float)
            bank_index = np.repeat(np.arange(8), per_bank)
            inputs_file = folder/'inputs.npz'
            np.savez_compressed(inputs_file, source=source, target=target, bank=bank_index, directions=data['directions'])
            prior_root, prior_row = None, None
            if previous:
                prior_root = previous/('readiness' if args.witness else 'formal')/task
                prior_complete = read(prior_root/'COMPLETE.json')
                assert prior_complete['passed'] and prior_complete['witness'] == args.witness
                assert prior_complete['inner_steps'] == [256, 512] and prior_complete['per_bank'] == per_bank
                assert sha(prior_root/'rows.json') == prior_complete['rows_sha256']
                prior_row = next(item for item in read(prior_root/'rows.json')['rows'] if item['id'] == row['id'])
                assert prior_row['checkpoint_sha256'] == identity['checkpoint_sha256']
                assert sha(prior_root/prior_row['inputs_file']) == prior_row['inputs_sha256']
                with np.load(prior_root/prior_row['inputs_file']) as old_inputs, np.load(inputs_file) as new_inputs:
                    assert old_inputs.files == new_inputs.files
                    assert all(np.array_equal(old_inputs[key], new_inputs[key]) for key in old_inputs.files)
            records = []
            for outer in (128, 129):
                with torch.no_grad():
                    baseline = base.train.generated_samples(model, torch.from_numpy(source), outer).numpy()
                variants = [('identity', 0)]+[(mode, level) for level in inner_levels for mode in MODES[1:]] if focal else [('identity', 0)]
                for mode, level in variants:
                    run_start = time.monotonic()
                    snapshots, metrics = {}, {}
                    if prior_row is not None and level in (0, 512):
                        old_arm = next(item for item in prior_row['arms'] if item['mode'] == mode
                                       and item['inner_steps'] == level and item['heun_steps'] == outer)
                        old_file = prior_root/old_arm['file']
                        assert sha(old_file) == old_arm['sha256']
                        if mode == 'identity':
                            with np.load(old_file) as bundle:
                                assert np.array_equal(bundle['endpoints'], baseline)
                        file = folder/f'{mode}-inner{level}-heun{outer}.npz'
                        shutil.copyfile(old_file, file)
                        assert sha(file) == old_arm['sha256']
                        records.append(dict(old_arm, file=str(file.relative_to(out)), reused=True,
                                            source_file=str(old_file), source_file_sha256=sha(old_file)))
                        dump(out/'progress.json', dict(pid=os.getpid(), completed_cases=len(results), cases=len(cases),
                            current=row['id'], completed_arms=len(records), total_arms=18 if focal else 2,
                            seconds=time.monotonic()-started))
                        print(row['id'], mode, level, outer, 'reused', flush=True)
                        continue
                    if focal:
                        field = Intervention(model, mode, level, outer)
                        generated = base.train.generated_samples(field, torch.from_numpy(source), outer).numpy()
                        assert field.call == 2*outer
                        snapshots, metrics = qualify_snapshots(model, field)
                    else:
                        generated = baseline.copy()
                    assert np.isfinite(generated).all()
                    if mode == 'identity':
                        assert np.array_equal(generated, baseline)
                    file = folder/f'{mode}-inner{level}-heun{outer}.npz'
                    np.savez_compressed(file, endpoints=generated, **snapshots)
                    records.append(dict(mode=mode, inner_steps=level, heun_steps=outer, file=str(file.relative_to(out)),
                        sha256=sha(file), **metrics, seconds=time.monotonic()-run_start))
                    dump(out/'progress.json', dict(pid=os.getpid(), completed_cases=len(results), cases=len(cases),
                        current=row['id'], completed_arms=len(records), total_arms=18 if focal else 2,
                        seconds=time.monotonic()-started))
                    print(row['id'], mode, level, outer, metrics.get('snapshot_qualified'), flush=True)
            with torch.no_grad():
                restored = base.train.generated_samples(model, torch.from_numpy(source), 129).numpy()
            native129 = next(record for record in records if record['mode'] == 'identity' and record['heun_steps'] == 129)
            with np.load(out/native129['file']) as bundle:
                assert np.array_equal(restored, bundle['endpoints'])
            results.append(dict(id=row['id'], dataset=cell['dataset'], family=cell['family'], focal=focal,
                **identity, assets_file=binding['file'], assets_sha256=binding['sha256'], inputs_file=str(inputs_file.relative_to(out)),
                inputs_sha256=sha(inputs_file), arms=records, native_roundtrip_exact=True,
                parameters_and_buffers_unchanged=True))
        assert before == tensor_hashes(model)
        if focal:
            assert all(not layer._forward_hooks for _, layer in model.ode_layers())
        dump(out/'rows.json', dict(rows=results))
    dump(out/'COMPLETE.json', dict(passed=True, witness=args.witness, check_only=args.check_only,
        cases=len(results), per_bank=0 if args.check_only else (1 if args.witness else 8),
        first_layer=FIRST, inner_steps=list(inner_levels), heun_steps=[128, 129],
        extension_source=str(previous) if previous else None,
        extension_source_terminal_sha256=sha(previous/'terminal-receipts.json') if previous else None,
        rows_sha256=sha(out/'rows.json'), anchors_sha256=sha(study/'anchors.json'),
        surrogate_terminal_sha256=sha(surrogate/'terminal-receipts.json'), seconds=time.monotonic()-started,
        environment_spec_hash='04f11f75', source_sha256={name:sha(Path(__file__).with_name(name)) for name in
            ['hidden_sampling_probe.py','hidden_readout_probe.py','hidden_energy.py','frozen_probe.py']}))


if __name__ == '__main__':
    main()
