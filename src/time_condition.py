from typing import Optional

import torch
from torch import Tensor, nn


def build_time_mlp(input_dim: int, hidden_dim: int, bias: bool = True) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim, bias=bias),
        nn.SiLU(),
        nn.Linear(hidden_dim, 1, bias=bias),
    )


def resolve_time_scalar(
    x: Tensor,
    t: Optional[Tensor],
    t_func: nn.Module,
    t_from_embed: Optional[nn.Module] = None,
) -> Tensor:
    if t is None:
        return t_func(x)

    if not torch.is_tensor(t):
        t = torch.tensor(t, device=x.device, dtype=x.dtype)
    else:
        t = t.to(device=x.device, dtype=x.dtype)

    batch_size = x.shape[0]
    seq_len = x.shape[1]

    is_tokenwise_scalar = t.dim() == 2 and t.shape == (batch_size, seq_len)
    if t_from_embed is not None and t.dim() >= 2 and t.shape[-1] != 1 and not is_tokenwise_scalar:
        t_scalar = t_from_embed(t)
        if t_scalar.dim() == 2:
            t_scalar = t_scalar.unsqueeze(1)
        return t_scalar

    if t.dim() == 0:
        return t.reshape(1, 1, 1).expand(batch_size, 1, 1)

    if t.dim() == 1:
        if t.shape[0] == 1 and batch_size != 1:
            t = t.expand(batch_size)
        if t.shape[0] != batch_size:
            raise ValueError(f"Expected scalar time with batch {batch_size}, got shape {tuple(t.shape)}")
        return t[:, None, None]

    if t.dim() == 2:
        if t.shape[-1] == 1:
            if t.shape[0] == 1 and batch_size != 1:
                t = t.expand(batch_size, 1)
            if t.shape[0] != batch_size:
                raise ValueError(f"Expected scalar time with batch {batch_size}, got shape {tuple(t.shape)}")
            return t[:, None, :]
        if t.shape == (batch_size, seq_len):
            return t.unsqueeze(-1)
        raise ValueError(
            "Expected scalar-like time with shape (B, 1) or (B, N), "
            f"got shape {tuple(t.shape)}"
        )

    if t.dim() == 3 and t.shape[-1] == 1:
        if t.shape[0] != batch_size:
            raise ValueError(f"Expected time batch {batch_size}, got shape {tuple(t.shape)}")
        if t.shape[1] not in (1, seq_len):
            raise ValueError(f"Expected time sequence length 1 or {seq_len}, got shape {tuple(t.shape)}")
        return t

    raise ValueError(
        "Unsupported time shape. Use scalar-like (), (B,), (B,1), (B,N), (B,N,1), "
        "or pass an embedding with last dim > 1 when t_from_embed is set."
    )
