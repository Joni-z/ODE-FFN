"""Replay the complete fixed frame screen, frozen controls and frame ablations."""
from __future__ import annotations

import argparse
import csv
import copy
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

parser = argparse.ArgumentParser();parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True);args = parser.parse_args()
root, out = args.root.resolve(), args.output.resolve();out.mkdir(exist_ok=False)
sys.path.insert(0, str(root/'source'));import frame_study as formal
started = time.monotonic();settings, geometry, native, core, operator = formal.setup(root)
formal.verify_inputs(root);assert formal.load(root/'output/status.json')['state']=='COMPLETE'
original = {str(p): native.sha(p) for p in (root/'output').rglob('*') if p.is_file()}
basis = np.load(root/'basis.npy');reference = Path(settings['geometry_reference'])
comparisons = [];curves = [];spectra = [];clocks = [];interventions = [];frame_checks = []
monitor = [__import__('json').loads(line) for line in (root/'output/monitor.jsonl').read_text().splitlines()]
launches = [__import__('json').loads(line) for line in (root/'output/launches.jsonl').read_text().splitlines()]
memory = {}
for job in launches:
    samples = [p['rss_bytes'] for row in monitor for p in row['processes'] if p['pid']==job['pid']]
    assert samples
    memory[(job['law'], job['arm'])] = max(samples)/2**30
counts = dict(new_checkpoints=0, reference_checkpoints=0, new_panels=0, reference_panels=0,
              new_samples=0, reference_samples=0, stream_batches=0, new_P_matrices=0)


def close(value, expected):
    np.testing.assert_allclose(value, expected, atol=1e-9, rtol=1e-7)


def table(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def field_metrics(prediction, bank):
    error = prediction.astype(np.float64)-bank['oracle'];plane = error@basis
    normal = error-plane@basis.T
    clean = bank['x']+np.maximum(1-bank['t'], .05)*prediction.astype(np.float64)
    return dict(oracle_mse=float(np.mean(error**2)), pair_mse=float(np.mean((prediction-bank['pair'])**2)),
        prediction_rms=float(np.sqrt(np.mean(prediction**2))),
        tangent_mse_per_dimension=float(np.mean(plane**2)),
        normal_mse_per_dimension=float(np.sum(normal**2)/(len(error)*126)),
        clean_oracle_mse=float(np.mean((clean-bank['clean'])**2)))


def replay_field(model, field):
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(field['x'][i:i+1024]),
            torch.from_numpy(field['t'][i:i+1024])).numpy() for i in range(0, len(field['x']), 1024)])


for entry in settings['laws']:
    law = entry['law'];banks = Path(entry['banks']);bank = geometry.training_bank(entry, native)
    with np.load(banks/'field.npz') as data:field = {k: data[k] for k in data.files}
    rng = torch.Generator().manual_seed(100000);digest = native.EMPTY
    digests = {0:digest};rng_states = {0:rng.get_state().clone()}
    for step in range(1, 5001):
        _, t, target, source = native.draw(bank, rng)
        digest = native.legacy.fingerprint_batch(digest, target, source, t)
        if step in geometry.STEPS:digests[step] = digest;rng_states[step] = rng.get_state().clone()
    counts['stream_batches'] += 5000
    old_zero = torch.load(reference/'output/runs'/law/'ode_k2_clean/step_0000000.pt', weights_only=False)
    for arm in (*formal.ARMS, 'ode_k2_clean', 'swiglu192_clean'):
        fresh = arm in formal.ARMS;family = arm.rsplit('_', 1)[0]
        folder = (root if fresh else reference)/'output/runs'/law/arm
        assert formal.load(folder/'status.json')['state']=='COMPLETE'
        records = [__import__('json').loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
        model = formal.build(root, law, arm) if fresh else core.CappedHead(native.old_build(128, family, 'direct').raw, 'clean')
        for step in geometry.STEPS:
            state = torch.load(folder/f'step_{step:07d}.pt', weights_only=False)
            native.restore_model(model, state);model.eval()
            assert state['input_stream_digest']==digests[step]
            assert torch.equal(state['data_rng'], rng_states[step])
            if step==0:
                for key, value in old_zero['model'].items():
                    if fresh and family.startswith(('frame_', 'global_')) or '.ffn.' not in key:
                        assert torch.equal(value, state['model'][key]), key
                for key, value in old_zero['runtime_buffers'].items():assert torch.equal(value, state['runtime_buffers'][key]), key
                assert not state['optimizer']['state']
            else:
                identifiers = state['optimizer']['param_groups'][0]['params']
                assert len(identifiers)==len(list(model.parameters()))
                for parameter, index in zip(model.parameters(), identifiers):
                    moment = state['optimizer']['state'][index];assert int(moment['step'])==step
                    for key in ('exp_avg', 'exp_avg_sq'):
                        assert parameter.shape==moment[key].shape and torch.isfinite(moment[key]).all()
            prediction = replay_field(model, field)
            with np.load(folder/f'field_{step:07d}.npz') as data:np.testing.assert_array_equal(prediction, data['prediction'])
            metrics = field_metrics(prediction, field);stored = next(r for r in records if r['step']==step)
            for key, value in metrics.items():close(value, stored[key])
            curves.append(dict(law=law, family=family, step=step, **metrics))
            counts['new_checkpoints' if fresh else 'reference_checkpoints'] += 1
            if fresh:
                old_P = formal.load(folder/f'generator_{step:07d}.json')
                layers = list(model.ode_layers());assert len(old_P)==len(layers)
                for saved, (name, layer) in zip(old_P, layers):
                    assert saved['layer']==name
                    singular = np.linalg.svd(layer.proj.weight.detach().double().numpy(), compute_uv=False)
                    np.testing.assert_allclose(singular, saved['P_singular_values'], atol=1e-8, rtol=1e-8)
                    counts['new_P_matrices'] += 1
                    spectra.append(dict(law=law, family=family, step=step, layer=name,
                        sigma_max=float(singular[0]), sigma_min=float(singular[-1]),
                        singular_cv=float(np.std(singular)/np.mean(singular)),
                        frame_weight_frobenius=float(layer.frame_weight.detach().norm())))
                saved_visits = formal.load(folder/f'clocks_{step:07d}.json')
                actual_visits = native.legacy.visit_diagnostics(model, banks)
                assert len(saved_visits)==len(actual_visits)
                for saved, actual in zip(saved_visits, actual_visits):
                    assert saved['layer']==actual['layer'] and saved['visit']==actual['visit']
                    assert saved['activation'].keys()==actual['activation'].keys()
                    for key, value in actual['activation'].items():
                        if isinstance(value, (float, int)):close(value, saved['activation'][key])
                        else:assert value==saved['activation'][key]
                    clocks.append(dict(law=law, family=family, step=step, layer=actual['layer'], **actual['activation']))
        result = formal.load(folder/'result.json');cost = native.compute_budget(model)
        for key in ('parameters', 'dense_forward_macs', 'dense_training_forward_backward_macs'):assert cost[key]==result[key]
        for clock in geometry.TIMES:
            with np.load(banks/f'panel_{clock:.1f}.npz') as data:x, t, truth = data['x'], data['t'], data['oracle_jacobian']
            value, jacobian = native.legacy.jacobian(model, torch.from_numpy(x), torch.from_numpy(t))
            with np.load(folder/'diagnostics'/f'panel_{clock:.1f}.npz') as data:
                np.testing.assert_array_equal(value.numpy(), data['prediction']);np.testing.assert_array_equal(jacobian.numpy(), data['jacobian'])
            stored = next(r for r in result['panels'] if abs(r['time']-clock)<1e-6)
            close(np.mean(np.sum((jacobian.numpy()-truth)**2, axis=(1, 2))), stored['jacobian_error'])
            counts['new_panels' if fresh else 'reference_panels'] += 1
        with np.load(banks/'sampling.npz') as data:source, target, directions = [torch.from_numpy(data[k]) for k in ('source', 'target', 'directions')]
        generated = native.legacy.generated_samples(model, source, 32)
        with np.load(folder/'diagnostics/samples.npz') as data:np.testing.assert_array_equal(generated.numpy(), data['generated'])
        close(native.legacy.sliced_w2(generated, target, directions), result['sw2'])
        angles = torch.arange(128)*2*torch.pi/128;td = torch.stack((torch.cos(angles), torch.sin(angles)), -1)
        close(native.legacy.sliced_w2(torch.tensor(generated.numpy()@basis, dtype=torch.float32),
            torch.tensor(target.numpy()@basis, dtype=torch.float32), td), result['tangent_sw2'])
        counts['new_samples' if fresh else 'reference_samples'] += 1
        comparisons.append(dict(law=law, family=family, new_training=fresh,
            cell_seconds=float(formal.load(folder/'status.json')['elapsed_seconds']),
            peak_sampled_rss_gib=memory[(law, arm)] if fresh else None,
            **{key: result[key] for key in ('oracle_mse', 'tangent_mse_per_dimension', 'normal_mse_per_dimension',
                'jacobian_error', 'sw2', 'tangent_sw2', 'normal_sample_energy_per_dimension',
                'parameters', 'dense_forward_macs', 'dense_training_forward_backward_macs')}))
        if fresh and family!='swiglu257':
            from verify_frame import dense
            captured = [];hooks = []
            for name, layer in model.ode_layers():
                def capture(module, inputs, output, name=name):captured.append((name, module, inputs[0].detach().clone()))
                hooks.append(layer.register_forward_hook(capture))
            with np.load(banks/'probe.npz') as data:
                with torch.no_grad():model(torch.from_numpy(data['x'][:32]), torch.from_numpy(data['t'][:32]))
            for hook in hooks:hook.remove()
            assert len(captured)==6
            for name, layer, hidden in captured:
                tested = copy.deepcopy(layer).double();value = hidden[:2].double()
                with torch.no_grad():actual = tested(value);expected, q = dense(tested, value)
                forward_error = float((actual-expected).abs().max());assert forward_error<1e-10
                p = tested.proj.weight.detach();effective = q@p@q.transpose(-1, -2)
                singular_error = float((torch.linalg.svdvals(effective)-torch.linalg.svdvals(p)).abs().max())
                orthogonal_error = float((q@q.transpose(-1, -2)-torch.eye(64)).abs().max())
                assert singular_error<1e-10 and orthogonal_error<1e-10
                frame_checks.append(dict(law=law, family=family, layer=name,
                    dense_frame_forward_max=forward_error, conjugated_singular_value_max=singular_error,
                    orthogonality_max=orthogonal_error))
        if fresh and family in ('frame_k2', 'frame_k1'):
            original_prediction = replay_field(model, field)
            changed = copy.deepcopy(model)
            with torch.no_grad():
                for _, layer in changed.ode_layers():layer.frame_weight.zero_();layer.frame_bias.zero_()
            for name, value in changed.state_dict().items():
                if not name.endswith(('frame_weight', 'frame_bias')):assert torch.equal(value, model.state_dict()[name]), name
            zero_prediction = replay_field(changed, field)
            metrics = field_metrics(zero_prediction, field)
            interventions.append(dict(law=law, family=family,
                output_displacement_rms=float(np.sqrt(np.mean((zero_prediction.astype(float)-original_prediction)**2))),
                original_oracle_mse=result['oracle_mse'], zero_frame_oracle_mse=metrics['oracle_mse'],
                zero_frame_tangent_mse=metrics['tangent_mse_per_dimension']))
        print(dict(law=law, family=family, counts=counts, elapsed=time.monotonic()-started), flush=True)

expected = dict(new_checkpoints=32, reference_checkpoints=16, new_panels=40, reference_panels=20,
    new_samples=8, reference_samples=4, stream_batches=10000, new_P_matrices=144)
assert counts==expected, counts
checks = []
for law in ('thin', 'thick'):
    selected = {r['family']:r for r in comparisons if r['law']==law}
    candidate = selected['frame_k2'];ordinary = [selected[f] for f in ('swiglu192', 'swiglu257')]
    for key in ('oracle_mse', 'tangent_mse_per_dimension', 'jacobian_error', 'sw2', 'tangent_sw2'):
        ratio = candidate[key]/min(r[key] for r in ordinary)
        threshold = .9 if key=='oracle_mse' and law=='thin' else 1.05 if key=='oracle_mse' else 1.10
        checks.append(dict(law=law, control='better_Swi', metric=key, ratio=ratio, threshold=threshold, passed=ratio<=threshold))
    for control in ('global_k2', 'frame_k1'):
        ratio = candidate['oracle_mse']/selected[control]['oracle_mse']
        checks.append(dict(law=law, control=control, metric='oracle_mse', ratio=ratio, threshold=.95, passed=ratio<=.95))
eligible = all(row['passed'] for row in checks)
formal.verify_inputs(root)
for name, digest in original.items():assert native.sha(Path(name))==digest, name
for name, rows in [('comparison.csv', comparisons), ('curves.csv', curves), ('P-history.csv', spectra),
                   ('clocks.csv', clocks), ('zero-frame.csv', interventions), ('frame-checks.csv', frame_checks),
                   ('decision-checks.csv', checks)]:table(out/name, rows)
formal.write(out/'decision.json', dict(context_lead=eligible, decision='schedule_historical_confirmation' if eligible else 'stop_this_frame_setting',
    new_architecture_nominated=False, no_frame_rank_scale_width_initializer_rescue=True))
formal.write(out/'audit.json', dict(status='PASS', counts=counts, formal_files_unchanged=len(original),
    elapsed_seconds=time.monotonic()-started, independent_active_differential_checks='frozen preflight',
    fitted_dense_frame_checks=len(frame_checks),
    peak_sampled_total_rss_gib=max(sum(p['rss_bytes'] for p in row['processes']) for row in monitor)/2**30,
    scope='native full checkpoint/Jacobian/sample replay, stream/optimizer/P/clock checks, frozen-frame intervention and fixed selection'))
shutil.copyfile(root/'preflight.json', out/'preflight.json')
with (root/'results.tsv').open('w') as stream:
    stream.write('commit\tmetric\tmemory_gb\tstatus\tdescription\n')
    for row in comparisons:
        if row['new_training']:
            keep = row['family']=='frame_k2' and eligible
            stream.write(f"{settings['registration_commit'][:7]}\t{row['oracle_mse']:.10g}\t{row['peak_sampled_rss_gib']:.3f}\t{'keep' if keep else 'discard'}\t{row['law']} {row['family']} clean; {'context lead only' if keep else 'fixed screen, evidence retained'}\n")
shutil.copyfile(root/'results.tsv', out/'results.tsv')
print(dict(status='AUDIT_PASS', eligible=eligible, counts=counts, elapsed=time.monotonic()-started), flush=True)
