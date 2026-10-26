"""Explicit, hash-bound copying of shared backbone initialization.

This helper is opt-in: importing it does not change model construction or RNG.
Checkpoint files must be trusted project artifacts; their expected hash comes
from a separately recorded source manifest, not from the file being loaded.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import io
from pathlib import Path

import torch
from torch import nn


def copy_common_backbone(model: nn.Module, checkpoint: Path,
                         expected_sha256: str) -> dict:
    """Copy all non-FFN state and runtime buffers after complete validation.

    FFN tensors and random-generator states are never loaded. Both source and
    destination must have exactly the same non-FFN state/buffer names and
    shapes. All validation and device/dtype conversion precede any copy, so a
    malformed checkpoint cannot leave a partially initialized model.

    Runtime buffers are mandatory, including nonpersistent Fourier constants.
    The in-memory bytes being hashed are the same bytes being deserialized.
    """
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(c not in '0123456789abcdefABCDEF' for c in expected_sha256)):
        raise ValueError('expected_sha256 must be a recorded SHA256 hex digest')
    checkpoint = Path(checkpoint)
    payload = checkpoint.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256.lower():
        raise ValueError('common-backbone checkpoint SHA256 mismatch')
    # Trusted project checkpoints contain optimizer/RNG metadata as well as weights.
    state = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
    if not isinstance(state, Mapping):
        raise ValueError('common-backbone checkpoint must be a mapping')

    def shared(mapping, label):
        if not isinstance(mapping, Mapping) or not all(isinstance(k, str) for k in mapping):
            raise ValueError(f'{label} must be a tensor-name mapping')
        return {name: value for name, value in mapping.items() if '.ffn.' not in name}

    source_state = shared(state.get('model'), 'checkpoint model')
    source_buffers = shared(state.get('runtime_buffers'), 'checkpoint runtime_buffers')
    target_state = shared(model.state_dict(), 'model state')
    target_buffers = shared(dict(model.named_buffers()), 'model buffers')
    prepared = []
    max_difference = 0.
    for label, source, target in (
            ('state', source_state, target_state),
            ('runtime buffer', source_buffers, target_buffers)):
        if set(source) != set(target):
            missing, extra = sorted(set(target)-set(source)), sorted(set(source)-set(target))
            raise ValueError(f'common-backbone {label} names differ: missing={missing}, extra={extra}')
        for name in sorted(target):
            src, dst = source[name], target[name]
            if not isinstance(src, torch.Tensor) or not isinstance(dst, torch.Tensor):
                raise ValueError(f'common-backbone {label} must be a tensor: {name}')
            if src.shape != dst.shape:
                raise ValueError(f'common-backbone {label} shape mismatch: {name}')
            if src.layout != torch.strided or dst.layout != torch.strided:
                raise ValueError(f'common-backbone {label} must be a dense tensor: {name}')
            if src.is_complex() != dst.is_complex():
                raise ValueError(f'common-backbone {label} complex dtype mismatch: {name}')
            if not bool(torch.isfinite(src).all()):
                raise ValueError(f'common-backbone {label} is nonfinite: {name}')
            # Conversion is staged too; device failures cannot precede a partial copy.
            converted = src.detach().to(device=dst.device, dtype=dst.dtype)
            if not bool(torch.isfinite(converted).all()):
                raise ValueError(f'common-backbone {label} conversion is nonfinite: {name}')
            if dst.numel():
                dtype = torch.complex128 if dst.is_complex() else torch.float64
                difference = float((dst.detach().to(dtype)-converted.to(dtype)).abs().max())
                max_difference = max(max_difference, difference)
            prepared.append((dst, converted))
    for name in source_buffers.keys() & source_state.keys():
        if not torch.equal(source_buffers[name], source_state[name]):
            raise ValueError(f'checkpoint state and runtime buffer disagree: {name}')

    with torch.no_grad():
        for target, source in prepared:
            target.copy_(source)
    return dict(checkpoint=str(checkpoint.resolve()), source_sha256=digest,
                copied_state_keys=sorted(target_state),
                copied_runtime_buffer_keys=sorted(target_buffers),
                max_before_difference=max_difference)
