import pathlib
import sys

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ffn_blocks import MultiHeadODELinear, _reshape_headwise_step
from ffn_factory import build_ffn, normalize_ffn_type


def test_reshape_headwise_step_preserves_head_dimension():
    step = torch.randn(2, 4, 3)
    reshaped = _reshape_headwise_step(step, batch_size=2, seq_len=4)
    assert reshaped.shape == (2, 4, 3, 1)


def test_multihead_odelinear_accepts_tokenwise_headwise_step():
    layer = MultiHeadODELinear(d_in=6, d_out=12, num_heads=3, orders=2)
    x = torch.randn(2, 4, 6)
    step = torch.randn(2, 4, 3)
    out = layer(x, step)
    assert out.shape == (2, 4, 12)


def test_new_condition_aware_ffns_build_and_forward():
    x = torch.randn(2, 4, 16)
    cond = torch.randn(2, 16)

    for ffn_type in ("time_split", "clean_target", "time_moe"):
        layer = build_ffn(ffn_type, in_features=16, hidden_features=64, t_embed_dim=16)
        out = layer(x, cond)
        assert out.shape == x.shape


def test_new_ffn_aliases_normalize():
    assert normalize_ffn_type("time_split_dual_path") == "time_split"
    assert normalize_ffn_type("clean_target_ffn") == "clean_target"
    assert normalize_ffn_type("time_routed_moe") == "time_moe"
