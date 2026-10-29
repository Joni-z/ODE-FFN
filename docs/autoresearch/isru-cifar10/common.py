"""Small, explicit runtime and artifact helpers."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def append(path, value):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(value, allow_nan=False)+'\n')


def verify(root):
    root = Path(root)
    for relative, digest in load(root/'bindings.json').items():
        path = Path(relative) if Path(relative).is_absolute() else root/relative
        assert sha(path) == digest, str(path)
    return load(root/'config.json'), load(root/'paths.json')


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def save_torch(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    torch.save(cpu_tree(value), tmp)
    os.replace(tmp, path)


def uint8_images(x):
    a = np.asarray(x, dtype=np.float32)
    assert a.ndim == 4 and a.shape[1:] == (3, 32, 32) and np.isfinite(a).all()
    return np.rint((np.clip(a, -1, 1)+np.float32(1))*np.float32(127.5)).astype(np.uint8)
