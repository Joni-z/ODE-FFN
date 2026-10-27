"""Frozen-checkpoint information/residual attribution; no training entry point."""
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

SCOPES = ('full', 'tangent', 'normal')
METRICS = ('total', 'information', 'residual', 'cross')


def load(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8*2**20):
            value.update(block)
    return value.hexdigest()


def table(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def setup(root: Path):
    settings = load(root/'settings.json')
    parent = Path(settings['cycle08'])
    sys.path.insert(0, str(parent/'source'))
    import invariant_study as prior
    _, base, previous, native, core = prior.setup(parent)
    return settings, prior, base, previous, native, core


def verify_inputs(root: Path) -> None:
    for name, digest in load(root/'bindings.json').items():
        assert sha(Path(name)) == digest, name
    settings, prior, _, _, _, _ = setup(root)
    parent = Path(settings['cycle08'])
    prior.verify_inputs(parent)
    for name, digest in load(parent/'sealed-runtime.json').items():
        assert sha(parent/name) == digest, name


def prepare(root: Path, parent: Path) -> None:
    root.mkdir(exist_ok=False)
    source = root/'source';source.mkdir()
    for name in ('conditional_study.py', 'conditional_oracle.py', 'verify_conditional.py', 'protocol.md', 'theory.md'):
        shutil.copyfile(Path(__file__).parent/name, source/name)
    repo = Path(__file__).resolve().parents[3]
    settings = dict(cycle08=str(parent.resolve()), registration_commit=subprocess.check_output(
        ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip())
    write(root/'settings.json', settings)
    base = Path(load(parent/'settings.json')['cycle07'])
    paths = list(source.iterdir())+[root/'settings.json', parent/'sealed-runtime.json', base/'sealed-runtime.json']
    write(root/'bindings.json', {str(p): sha(p) for p in paths})
    verify_inputs(root)
    print(dict(prepared=str(root), checkpoints=72, bindings=len(paths)), flush=True)


def summarize(values: np.ndarray, t: np.ndarray, metadata: dict) -> list[dict]:
    rows = []
    masks = [('all', np.ones(len(t), dtype=bool))]
    masks += [(f'{i/5:.1f}-{(i+1)/5:.1f}', (t[:, 0] >= i/5) & (t[:, 0] < (i+1)/5)) for i in range(5)]
    for name, mask in masks:
        assert mask.any()
        for index, scope in enumerate(SCOPES):
            average = values[mask, index].mean(axis=0)
            rows.append(dict(**metadata, scope=scope, time_bin=name, count=int(mask.sum()),
                             **{key: float(value) for key, value in zip(METRICS, average)}))
    return rows


def decision(rows: list[dict]) -> dict:
    endpoints = [r for r in rows if r['law']=='thin' and r['head']=='clean' and r['step']==5000
                 and r['scope']=='full' and r['time_bin']=='all']
    ode = next(r for r in endpoints if r['family']=='ode_k2')
    differences = []
    for family in ('swiglu128', 'swiglu192'):
        control = next(r for r in endpoints if r['family']==family)
        difference = {key: control[key]-ode[key] for key in METRICS}
        assert difference['total'] > 0
        differences.append(dict(control=family, **difference,
                                **{key+'_fraction': difference[key]/difference['total'] for key in METRICS[1:]}))
    input_related = all(r['information'] >= .8*r['total'] and abs(r['cross']) <= .2*r['total'] for r in differences)
    residual_related = all(r['residual'] >= .8*r['total'] and abs(r['cross']) <= .2*r['total'] for r in differences)
    label = 'predominantly_input_information' if input_related else (
        'predominantly_remaining_residual' if residual_related else 'mixed_or_unresolved')
    return dict(attribution=label, differences=differences, new_training=False, new_architecture_nominated=False)


def preflight(root: Path) -> None:
    verify_inputs(root)
    *_, core = setup(root)
    from verify_conditional import verify
    result = verify(core)
    write(root/'preflight.json', result)
    write(root/'ready.json', dict(bindings=sha(root/'bindings.json'), preflight=sha(root/'preflight.json')))
    print(result, flush=True)


def run(root: Path) -> None:
    import psutil
    from conditional_oracle import input_rows, best_field, components
    started = time.monotonic();verify_inputs(root)
    settings, _, base, previous, _, _ = setup(root)
    ready = load(root/'ready.json')
    assert ready == dict(bindings=sha(root/'bindings.json'), preflight=sha(root/'preflight.json'))
    out = root/'output';out.mkdir(exist_ok=False)
    raw = out/'raw';raw.mkdir()
    basis = np.load(base/'basis.npy');rows = [];telemetry = [];complete = 0
    write(out/'invocation.json', dict(argv=sys.argv, executable=sys.executable, pid=os.getpid(), **settings))
    def status(terminal=False, latest=''):
        row = dict(state='COMPLETE' if terminal else 'RUNNING', completed=complete, total=72,
                   latest=latest, pid=os.getpid(), elapsed_seconds=time.monotonic()-started,
                   rss_bytes=psutil.Process().memory_info().rss)
        assert row['elapsed_seconds'] < 600, 'registered runtime ceiling'
        write(out/'status.json', row);telemetry.append(row)
    status()
    for entry in load(base/'settings.json')['laws']:
        law = entry['law']
        with np.load(Path(entry['banks'])/'field.npz') as data:
            x, t, truth = [data[key] for key in ('x', 't', 'oracle')]
        assert len(x)==20000 and np.all(t<1)
        for arm in previous.ARMS:
            family, head = arm.rsplit('_', 1)
            folder = base/'output/runs'/law/arm
            for step in previous.STEPS:
                state = torch.load(folder/f'step_{step:07d}.pt', weights_only=False)
                matrix = state['model']['raw.input.weight'].double().numpy()
                observed, singular = input_rows(matrix)
                with np.load(folder/f'field_{step:07d}.npz') as data:
                    prediction = data['prediction']
                optimum = best_field(x, t, observed, basis, entry['normal_variance'], head)
                values = components(prediction, optimum, truth, basis)
                metadata = dict(law=law, family=family, head=head, step=step, input_rank=len(observed),
                                plane_capture=float(np.sum((observed@basis)**2)/2))
                rows.extend(summarize(values, t, metadata))
                np.savez_compressed(raw/f'{law}-{arm}-{step:07d}.npz', components=values,
                                    input_rows=observed, input_singular_values=singular)
                complete += 1;status(latest=f'{law}/{arm}/{step}')
                print(dict(completed=complete, law=law, arm=arm, step=step,
                           elapsed=time.monotonic()-started), flush=True)
    table(out/'decomposition.csv', rows);table(out/'telemetry.csv', telemetry)
    result = decision(rows);write(out/'decision.json', result)
    verify_inputs(root);status(terminal=True);table(out/'telemetry.csv', telemetry)
    write(out/'receipt.json', dict(status='PASS', checkpoints=complete, summary_rows=len(rows), new_training=False,
        elapsed_seconds=time.monotonic()-started, pid=os.getpid(),
        peak_sampled_rss_gib=max(r['rss_bytes'] for r in telemetry)/2**30, registration_commit=settings['registration_commit']))
    print(dict(status='COMPLETE', attribution=result['attribution'], elapsed=time.monotonic()-started), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'preflight', 'run'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cycle08', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        assert args.cycle08 is not None
        prepare(args.root.resolve(), args.cycle08)
    else:
        globals()[args.command](args.root.resolve())
