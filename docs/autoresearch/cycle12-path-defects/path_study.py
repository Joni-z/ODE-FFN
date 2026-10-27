"""Frozen Heun paths and exact oracle-suffix defect accounting."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

FAMILIES = ('ode_k2', 'frame_k2', 'global_k2', 'frame_k1', 'swiglu192', 'swiglu257')


def load(path):return json.loads(Path(path).read_text())
def write(path, value):Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
def table(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys);writer.writeheader();writer.writerows(rows)


def setup(root):
    settings = load(root/'settings.json');parent = Path(settings['parent'])
    if 'frame_study' not in sys.modules:
        spec = importlib.util.spec_from_file_location('frame_study', parent/'source/frame_study.py')
        module = importlib.util.module_from_spec(spec);sys.modules['frame_study'] = module;spec.loader.exec_module(module)
    prior = sys.modules['frame_study'];assert Path(prior.__file__).resolve()==parent/'source/frame_study.py'
    prior_settings, geometry, native, core, _ = prior.setup(parent)
    return settings, prior, prior_settings, geometry, native, core


def verify_inputs(root):
    settings, prior, _, _, native, _ = setup(root)
    for name, digest in load(root/'bindings.json').items():assert native.sha(Path(name))==digest, name
    parent = Path(settings['parent']);prior.verify_inputs(parent)
    for name, digest in load(parent/'sealed-runtime.json').items():assert native.sha(parent/name)==digest, name


def prepare(root, parent):
    root.mkdir(exist_ok=False);source = root/'source';source.mkdir()
    for name in ('path_study.py', 'defect_core.py', 'verify_defects.py', 'protocol.md', 'theory.md'):
        shutil.copyfile(Path(__file__).parent/name, source/name)
    repo = Path(__file__).resolve().parents[3]
    write(root/'settings.json', dict(parent=str(parent.resolve()), registration_commit=subprocess.check_output(
        ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(), models=12, sources=2048, steps=32))
    _, _, _, _, native, _ = setup(root)
    paths = [p for p in source.iterdir() if p.is_file()]+[root/'settings.json', parent/'sealed-runtime.json']
    write(root/'bindings.json', {str(p):native.sha(p) for p in paths});verify_inputs(root)
    print(dict(prepared=str(root), bindings=len(paths), models=12, new_training=False), flush=True)


def model_and_folder(root, law, family):
    settings, prior, old, _, native, core = setup(root);parent = Path(settings['parent'])
    if family in ('ode_k2', 'swiglu192'):
        model = core.CappedHead(native.old_build(128, family, 'direct').raw, 'clean')
        folder = Path(old['geometry_reference'])/'output/runs'/law/(family+'_clean')
    else:
        model = prior.build(parent, law, family+'_clean')
        folder = parent/'output/runs'/law/(family+'_clean')
    state = torch.load(folder/'step_0005000.pt', weights_only=False)
    native.restore_model(model, state);model.eval()
    return model, folder, state


def oracle_for(core, basis, variance):
    def oracle(x, t):
        return torch.from_numpy(core.oracle(x.numpy(), t.numpy(), basis, variance)['oracle']).to(x.dtype)
    return oracle


@torch.no_grad()
def preflight(root):
    from defect_core import trace, normal_suffix_gains
    from verify_defects import verify
    verify_inputs(root);settings, _, old, _, native, core = setup(root)
    mathematics = verify(core, native);basis = np.load(Path(settings['parent'])/'basis.npy')
    rows = [];started = time.monotonic()
    for entry in old['laws']:
        law = entry['law'];banks = Path(entry['banks'])
        with np.load(banks/'sampling.npz') as data:source = torch.from_numpy(data['source'])
        oracle = oracle_for(core, basis, entry['normal_variance'])
        replay = trace(oracle, source)[-1]
        with np.load(banks/'oracle-samples.npz') as data:np.testing.assert_array_equal(replay.numpy(), data['generated'])
        for family in FAMILIES:
            model, folder, _ = model_and_folder(root, law, family)
            replay = trace(model, source)[-1]
            native_replay = native.legacy.generated_samples(model, source, 32)
            with np.load(folder/'diagnostics/samples.npz') as data:np.testing.assert_array_equal(replay.numpy(), data['generated'])
            np.testing.assert_array_equal(replay.numpy(), native_replay.numpy())
            rows.append(dict(law=law, family=family, native_complete_sampler_bitwise=True))
            assert time.monotonic()-started<600
        print(dict(preflight_law=law, completed=len(rows)), flush=True)
    verify_inputs(root)
    write(root/'preflight.json', dict(status='PASS', mathematics=mathematics, endpoints=rows,
        normal_gain_thin=normal_suffix_gains(0.).tolist(), normal_gain_thick=normal_suffix_gains(.04).tolist()))
    write(root/'ready.json', dict(bindings=native.sha(root/'bindings.json'), preflight=native.sha(root/'preflight.json')))
    print(dict(status='PREFLIGHT_PASS', models=12, elapsed=time.monotonic()-started), flush=True)


@torch.no_grad()
def run(root):
    import psutil
    from defect_core import advance, trace, suffix, energy, account, normal_suffix_gains, forced_readout
    started = time.monotonic();verify_inputs(root)
    settings, _, old, _, native, core = setup(root)
    assert load(root/'ready.json')==dict(bindings=native.sha(root/'bindings.json'), preflight=native.sha(root/'preflight.json'))
    out = root/'output';out.mkdir(exist_ok=False);raw = out/'raw';raw.mkdir()
    basis = np.load(Path(settings['parent'])/'basis.npy')
    endpoints = [];steps = [];bins = [];telemetry = []
    write(out/'invocation.json', dict(executable=sys.executable, argv=sys.argv, pid=os.getpid(), **settings))
    def status(complete=False, latest='', stage='', index=0):
        row = dict(state='COMPLETE' if complete else 'RUNNING', completed=len(endpoints), total=12,
            latest=latest, stage=stage, index=index, pid=os.getpid(), elapsed_seconds=time.monotonic()-started,
            rss_bytes=psutil.Process().memory_info().rss)
        assert row['elapsed_seconds']<600
        write(out/'status.json', row);telemetry.append(row)
    status()
    means = 3*np.column_stack((np.cos(np.arange(8)*2*np.pi/8), np.sin(np.arange(8)*2*np.pi/8)))
    for entry in old['laws']:
        law = entry['law'];banks = Path(entry['banks'])
        with np.load(banks/'sampling.npz') as data:source = torch.from_numpy(data['source'])
        assert source.shape==(2048, 128)
        oracle = oracle_for(core, basis, entry['normal_variance']);reference = trace(oracle, source)
        with np.load(banks/'oracle-samples.npz') as data:np.testing.assert_array_equal(reference[-1].numpy(), data['generated'])
        np.savez_compressed(raw/f'{law}-oracle-path.npz', path=reference.numpy())
        gains = normal_suffix_gains(entry['normal_variance'])
        for family in FAMILIES:
            model, folder, state = model_and_folder(root, law, family)
            path = [source.clone()];local = [];stage_rows = []
            for index in range(32):
                following, first, predictor, second = advance(model, path[-1], index)
                exact_next, oracle_first, _, _ = advance(oracle, path[-1], index)
                t = torch.full((len(source), 1), index/32, dtype=source.dtype)
                oracle_second_at_model = oracle(predictor, torch.full_like(t, (index+1)/32))
                ref_model = model(reference[index], t);ref_oracle = oracle(reference[index], t)
                local.append(following.numpy().astype(np.float64)-exact_next.numpy())
                first_error = energy(first.numpy().astype(np.float64)-oracle_first.numpy(), basis)
                second_error = energy(second.numpy().astype(np.float64)-oracle_second_at_model.numpy(), basis)
                reference_error = energy(ref_model.numpy().astype(np.float64)-ref_oracle.numpy(), basis)
                stage_rows.append(dict(**{f'model_stage1_{k}':v for k,v in first_error.items()},
                    **{f'model_stage2_{k}':v for k,v in second_error.items()},
                    **{f'oracle_path_stage1_{k}':v for k,v in reference_error.items()},
                    model_path_normal_rms=float(np.sqrt(energy(path[-1].numpy(), basis)['normal'])),
                    oracle_path_normal_rms=float(np.sqrt(energy(reference[index].numpy(), basis)['normal']))))
                path.append(following);status(latest=f'{law}/{family}', stage='model-prefix', index=index)
            path = torch.stack(path)
            with np.load(folder/'diagnostics/samples.npz') as data:np.testing.assert_array_equal(path[-1].numpy(), data['generated'])
            bridges = [reference[-1].numpy()]
            for index in range(1, 32):
                bridges.append(suffix(oracle, path[index], index).numpy())
                status(latest=f'{law}/{family}', stage='oracle-suffix', index=index)
            bridges.append(path[-1].numpy());bridges = np.stack(bridges);local = np.stack(local)
            summary, contributions, signed = account(bridges, local, basis, gains)
            forced, forced_signed, projector = forced_readout(bridges, source.numpy(), state, basis, gains)
            summary.update(forced)
            for row in contributions:
                row.update(law=law, family=family, stage2_time=(row['step']+1)/32, **stage_rows[row['step']])
                if row['scope']=='normal':
                    row['forced_signed_contribution'] = float(forced_signed[row['step']].mean())
                    row['remaining_signed_contribution'] = row['signed_endpoint_mse_contribution']-row['forced_signed_contribution']
            for scope in ('full', 'tangent', 'normal'):
                total = summary[scope+'_paired_endpoint_mse']
                for group in range(4):
                    selected = [r for r in contributions if r['scope']==scope and group*8<=r['step']<(group+1)*8]
                    signed_sum = sum(r['signed_endpoint_mse_contribution'] for r in selected)
                    bins.append(dict(law=law, family=family, scope=scope, time_bin=f'{group/4:.2f}-{(group+1)/4:.2f}',
                        signed_endpoint_mse_contribution=signed_sum, signed_share=signed_sum/total if total>1e-20 else None,
                        local_step_energy_mean=float(np.mean([r['local_step_energy'] for r in selected])),
                        transported_step_energy_mean=float(np.mean([r['transported_step_energy'] for r in selected])),
                        forced_signed_contribution=sum(r['forced_signed_contribution'] for r in selected) if scope=='normal' else None))
            model_ids = np.argmin(np.sum(((path[-1].numpy()@basis)[:, None, :]-means[None])**2, axis=-1), axis=1)
            oracle_ids = np.argmin(np.sum(((reference[-1].numpy()@basis)[:, None, :]-means[None])**2, axis=-1), axis=1)
            result = load(folder/'result.json')
            endpoints.append(dict(law=law, family=family, **summary,
                sourcewise_mode_disagreement=float(np.mean(model_ids!=oracle_ids)),
                model_endpoint_normal_rms=float(np.sqrt(energy(path[-1].numpy(), basis)['normal'])),
                oracle_endpoint_normal_rms=float(np.sqrt(energy(reference[-1].numpy(), basis)['normal'])),
                existing_field_mse=result['oracle_mse'], existing_sw2=result['sw2'], existing_tangent_sw2=result['tangent_sw2']))
            steps.extend(contributions)
            np.savez_compressed(raw/f'{law}-{family}.npz', path=path.numpy(), bridges=bridges, local_defects=local,
                signed_per_source=signed, forced_signed_per_source=forced_signed, forced_projector=projector,
                model_mode_ids=model_ids, oracle_mode_ids=oracle_ids)
            status(latest=f'{law}/{family}', stage='model-complete')
            print(dict(completed=len(endpoints), law=law, family=family, elapsed=time.monotonic()-started), flush=True)
    table(out/'endpoints.csv', endpoints);table(out/'steps.csv', steps);table(out/'bins.csv', bins)
    verify_inputs(root);status(complete=True);table(out/'telemetry.csv', telemetry)
    write(out/'receipt.json', dict(status='PASS', models=12, sources_per_model=2048, branch_endpoints_per_model=33,
        step_rows=len(steps), bin_rows=len(bins), pid=os.getpid(), elapsed_seconds=time.monotonic()-started,
        peak_sampled_rss_gib=max(r['rss_bytes'] for r in telemetry)/2**30,
        registration_commit=settings['registration_commit'], new_training=False, new_sampler_nominated=False))
    print(dict(status='COMPLETE', models=12, elapsed=time.monotonic()-started), flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser();parser.add_argument('command', choices=('prepare', 'preflight', 'run'))
    parser.add_argument('--root', type=Path, required=True);parser.add_argument('--parent', type=Path)
    args = parser.parse_args()
    if args.command=='prepare':
        assert args.parent is not None
        prepare(args.root.resolve(), args.parent)
    else:globals()[args.command](args.root.resolve())
