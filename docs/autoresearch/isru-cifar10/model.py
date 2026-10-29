"""Existing free-ISRU and ordinary FFNs in one larger, unchanged JiT body."""
from __future__ import annotations

import copy
import sys
import types
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ('isru', 'swiglu_matched', 'swiglu_wide')


def sdpa_forward(self, x, rope, t_emb=None, distance_matrix=None):
    """Ordinary attention fast path, with all experimental attention options off."""
    assert not any((self.adaptive_temperature, self.position_bias,
                    self.head_specialization, self.time_dependent_qkv))
    batch, tokens, width = x.shape
    qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads,
                            width//self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = rope(self.q_norm(q)), rope(self.k_norm(k))
    y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
    return self.proj_drop(self.proj(y.transpose(1, 2).reshape(batch, tokens, width)))


class FreeISRU(nn.Module):
    def __init__(self, p: torch.Tensor):
        super().__init__()
        with torch.no_grad():
            initial = torch.eye(len(p), dtype=p.dtype, device=p.device) + .3*p + .045*(p@p)
        self.matrix1 = nn.Parameter(initial.clone())
        self.matrix2 = nn.Parameter(initial.clone())

    def forward(self, x, cond=None, return_aux=False):
        u = F.linear(x, self.matrix1)
        y = F.linear(u*torch.rsqrt(1+1.2*u.square()), self.matrix2)
        return (y, {}) if return_aux else y


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.w12 = nn.Linear(width, 2*hidden)
        self.w3 = nn.Linear(hidden, width)
        for m in (self.w12, self.w3):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, cond=None, return_aux=False):
        gate, value = self.w12(x).chunk(2, dim=-1)
        y = self.w3(F.silu(gate)*value)
        return (y, {}) if return_aux else y


def native_modules(source: Path):
    source = source.resolve()
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from util.compile_control import set_compile_enabled
    set_compile_enabled(False)
    import model_jit, ffn_blocks
    assert Path(model_jit.__file__).resolve().parent == source
    return model_jit, ffn_blocks


def build_all(source: Path, width: int = 192, depth: int = 6, fast_attention: bool = True):
    native, ffn = native_modules(source)
    torch.manual_seed(0)
    base = native.JiT(input_size=32, patch_size=4, in_channels=3,
        hidden_size=width, depth=depth, num_heads=3, mlp_ratio=4,
        num_classes=1, bottleneck_dim=128, in_context_len=0,
        ffn_type='swiglu', attn_drop=0., proj_drop=0.)
    models = {}
    for arm in ARMS:
        model = copy.deepcopy(base)
        if fast_attention:
            for block in model.blocks:
                block.attn.forward = types.MethodType(sdpa_forward, block.attn)
        torch.manual_seed(0)
        for block in model.blocks:
            if arm == 'isru':
                original = ffn.ODELayer(in_features=width, hidden_features=width,
                                        orders=2, t_embed_dim=None)
                block.mlp = FreeISRU(original.proj.weight)
            elif arm == 'swiglu_matched':
                block.mlp = SwiGLU(width, 2*width//3)
            else:
                original = block.mlp
                block.mlp = SwiGLU(width, original.w3.in_features)
                block.mlp.w12.load_state_dict(original.w12.state_dict())
                block.mlp.w3.load_state_dict(original.w3.state_dict())
        models[arm] = model
    names = [n for n in models[ARMS[0]].state_dict()
             if not (n.startswith('blocks.') and '.mlp.' in n)]
    for arm in ARMS[1:]:
        for n in names:
            assert torch.equal(models[ARMS[0]].state_dict()[n], models[arm].state_dict()[n]), n
    return models


def costs(model):
    ffn = model.blocks[0].mlp
    width = model.hidden_size
    forward = 2*width*width if isinstance(ffn, FreeISRU) else 3*width*ffn.w3.in_features
    return dict(parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                FFN_parameters_per_block=sum(p.numel() for p in ffn.parameters()),
                FFN_dense_forward_MACs_per_token_per_block=forward,
                scope='dense FFN matrix work; pointwise and backbone excluded')
