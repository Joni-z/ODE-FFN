"""Freeze a new runtime without touching historical runs or environments."""
from __future__ import annotations
import argparse
import shutil
import subprocess
from pathlib import Path
from common import sha, write, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    repo, root = args.repo.resolve(), args.root.resolve()
    assert not root.exists(), 'runtime must be new'
    assert shutil.disk_usage(root.parent).free > 20*1024**3
    root.mkdir()
    source = root/'source'
    source.mkdir()
    code = repo/'docs/autoresearch/isru-cifar10'
    for path in code.iterdir():
        if path.is_file() and path.suffix in ('.py', '.md', '.json'):
            shutil.copy2(path, source/path.name)
    shutil.copytree(repo/'src', source/'native', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(repo/'docs/research-cubic-split-2026-10-26/cubic_core.py', source/'synthetic_reference.py')
    shutil.copy2(code/'config.json', root/'config.json')
    prep = Path('/Users/mr.z/ode-workspace-prep')
    paths = dict(data=str(prep/'image-bridge-data/cifar10-v2'),
                 training_python=str(prep/'.venv-image-bridge/bin/python'),
                 metrics_python=str(prep/'.venv-image-metrics-v2/bin/python'),
                 inception=str(next((prep/'image-bridge-inception/checkpoints').glob('*.pth'))))
    write(root/'paths.json', paths)
    for name, rel in (('training-env-spec.json', 'docs/autoresearch/cycle17-cifar-baselines/training-env-spec.json'),
                      ('metrics-env-spec.json', 'docs/autoresearch/image-swiglu-initialization/metrics-env-spec.json')):
        shutil.copy2(repo/rel, root/name)
    for file in ('images.uint8.npy', 'split.npz', 'manifest.json'):
        assert (Path(paths['data'])/file).is_file()
    bindings = {str(p.relative_to(root)): sha(p) for p in sorted(source.rglob('*')) if p.is_file()}
    for file in ('config.json', 'paths.json', 'training-env-spec.json', 'metrics-env-spec.json'):
        bindings[file] = sha(root/file)
    for file in ('images.uint8.npy', 'split.npz', 'manifest.json'):
        p = Path(paths['data'])/file
        bindings[str(p)] = sha(p)
    bindings[paths['inception']] = sha(paths['inception'])
    write(root/'bindings.json', bindings)
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    write(root/'registration.json', dict(source_commit=commit, config=load(root/'config.json'),
          bindings_sha256=sha(root/'bindings.json'), state='FROZEN_NOT_TRAINED'))
    (root/'output').mkdir()
    print(root, flush=True)


if __name__ == '__main__':
    main()
