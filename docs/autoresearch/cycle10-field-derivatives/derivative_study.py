"""Registered frozen complete-field derivative study."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8*2**20):
            value.update(block)
    return value.hexdigest()


def table(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def setup(root):
    settings = load(root/'settings.json');parent = Path(settings['cycle09'])
    sys.path.insert(0, str(parent/'source'))
    import conditional_study as prior
    assert Path(prior.__file__).resolve()==parent/'source/conditional_study.py'
    _, _, base, previous, native, core = prior.setup(parent)
    return settings, prior, base, previous, native, core


def verify_inputs(root):
    for name, digest in load(root/'bindings.json').items():
        assert sha(name)==digest, name
    settings, prior, *_ = setup(root);parent = Path(settings['cycle09'])
    prior.verify_inputs(parent)
    for name, digest in load(parent/'sealed-runtime.json').items():
        assert sha(parent/name)==digest, name


def prepare(root, parent):
    root.mkdir(exist_ok=False);source = root/'source';source.mkdir()
    for name in ('derivative_study.py', 'field_derivatives.py', 'verify_derivatives.py', 'protocol.md', 'theory.md'):
        shutil.copyfile(Path(__file__).parent/name, source/name)
    repo = Path(__file__).resolve().parents[3]
    write(root/'settings.json', dict(cycle09=str(parent.resolve()), registration_commit=subprocess.check_output(
        ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()))
    paths = list(source.iterdir())+[root/'settings.json', parent/'sealed-runtime.json']
    write(root/'bindings.json', {str(p): sha(p) for p in paths});verify_inputs(root)
    print(dict(prepared=str(root), bindings=len(paths), panels=90), flush=True)


def preflight(root):
    verify_inputs(root);*_, core = setup(root)
    from verify_derivatives import verify
    result = verify(core);write(root/'preflight.json', result)
    write(root/'ready.json', dict(bindings=sha(root/'bindings.json'), preflight=sha(root/'preflight.json')))
    print(result, flush=True)


def summarize_late(rows):
    result = []
    for law in ('thin', 'thick'):
        for family in ('ode_k2', 'swiglu128', 'swiglu192'):
            selected = [r for r in rows if r['law']==law and r['family']==family and r['head']=='clean' and r['time'] in (.7, .9)]
            assert len(selected)==2
            symmetric, skew = [float(np.mean([r['tangent_residual_'+key] for r in selected])) for key in ('symmetric', 'skew')]
            share = symmetric/max(symmetric+skew, 1e-30)
            label = 'symmetric_dominant' if share>=.8 else 'skew_dominant' if share<=.2 else 'mixed'
            result.append(dict(law=law, family=family, head='clean', panels=2, anchors=128,
                tangent_residual=symmetric+skew, tangent_symmetric=symmetric, tangent_skew=skew,
                symmetric_share=share, classification=label))
    return result


def run(root):
    import psutil
    from field_derivatives import conditional, diagnostics
    started = time.monotonic();verify_inputs(root)
    settings, _, base, previous, _, _ = setup(root)
    from conditional_oracle import input_rows
    assert load(root/'ready.json')==dict(bindings=sha(root/'bindings.json'), preflight=sha(root/'preflight.json'))
    out = root/'output';out.mkdir(exist_ok=False);raw = out/'raw';raw.mkdir()
    basis = np.load(base/'basis.npy');records = [];telemetry = []
    write(out/'invocation.json', dict(argv=sys.argv, executable=sys.executable, pid=os.getpid(), **settings))
    def status(complete=False, latest=''):
        row = dict(state='COMPLETE' if complete else 'RUNNING', completed=len(records), total=90,
            latest=latest, pid=os.getpid(), elapsed_seconds=time.monotonic()-started,
            rss_bytes=psutil.Process().memory_info().rss)
        assert row['elapsed_seconds']<600
        write(out/'status.json', row);telemetry.append(row)
    status()
    for entry in load(base/'settings.json')['laws']:
        law = entry['law'];banks = Path(entry['banks'])
        for arm in previous.ARMS:
            family, head = arm.rsplit('_', 1);folder = base/'output/runs'/law/arm
            state = torch.load(folder/'step_0005000.pt', weights_only=False)
            rows, _ = input_rows(state['model']['raw.input.weight'].double().numpy())
            for clock in previous.TIMES:
                with np.load(banks/f'panel_{clock:.1f}.npz') as data:
                    x, t, truth = data['x'], data['t'], data['oracle_jacobian']
                actual_clock = float(t[0, 0])
                assert np.all(t == t[0, 0])
                with np.load(folder/'diagnostics'/f'panel_{clock:.1f}.npz') as data:
                    prediction, model_jacobian = data['prediction'], data['jacobian']
                result = conditional(x, actual_clock, rows, basis, entry['normal_variance'], head)
                metrics = diagnostics(model_jacobian, result['jacobian'], truth, basis)
                field_error = prediction.astype(np.float64)-result['field']
                row = dict(law=law, family=family, head=head, time=clock, actual_time=actual_clock, count=len(x),
                    covariance_identity_max=result['covariance_identity_max'],
                    field_residual_per_ambient=float(np.mean(field_error**2)),
                    field_residual_per_tangent=float(np.mean((field_error@basis)**2)),
                    **{key: float(np.mean(value)) for key, value in metrics.items()})
                records.append(row)
                np.savez_compressed(raw/f'{law}-{arm}-{clock:.1f}.npz', input_rows=rows,
                    reference_field=result['field'], reference_jacobian=result['jacobian'], **metrics)
                status(latest=f'{law}/{arm}/{clock}')
                print(dict(completed=len(records), law=law, arm=arm, time=clock,
                           elapsed=time.monotonic()-started), flush=True)
    table(out/'panels.csv', records);table(out/'late-summary.csv', summarize_late(records))
    verify_inputs(root);status(complete=True);table(out/'telemetry.csv', telemetry)
    write(out/'receipt.json', dict(status='PASS', panels=len(records), anchors=64*len(records),
        elapsed_seconds=time.monotonic()-started, pid=os.getpid(),
        peak_sampled_rss_gib=max(r['rss_bytes'] for r in telemetry)/2**30,
        new_training=False, new_architecture_nominated=False, registration_commit=settings['registration_commit']))
    print(dict(status='COMPLETE', panels=len(records), elapsed=time.monotonic()-started), flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'preflight', 'run'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cycle09', type=Path)
    args = parser.parse_args()
    if args.command=='prepare':
        assert args.cycle09 is not None
        prepare(args.root.resolve(), args.cycle09)
    else:
        globals()[args.command](args.root.resolve())
