import pathlib
import sys

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model_jit import JiT
from util.compile_control import set_compile_enabled
from util.model_util import VisionRotaryEmbeddingFast

set_compile_enabled(False)


def _build_tiny_jit(**overrides):
    base = dict(
        input_size=8,
        patch_size=4,
        in_channels=3,
        hidden_size=16,
        depth=4,
        num_heads=4,
        mlp_ratio=2.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=10,
        bottleneck_dim=8,
        in_context_len=2,
        in_context_start=1,
        ffn_type="swiglu",
    )
    base.update(overrides)
    return JiT(**base)


def test_vision_rotary_embedding_fast_stays_on_cpu():
    rope = VisionRotaryEmbeddingFast(dim=2, pt_seq_len=2, num_cls_token=1)
    x = torch.randn(1, 2, 5, 4)
    out = rope(x)
    assert out.shape == x.shape
    assert out.device.type == "cpu"


def test_attention_design_forward_runs():
    model = _build_tiny_jit(
        attention_kwargs={
            "adaptive_temperature": True,
            "position_bias": True,
            "head_specialization": True,
        }
    )
    x = torch.randn(2, 3, 8, 8)
    t = torch.rand(2)
    y = torch.randint(0, 10, (2,))
    out = model(x, t, y)
    assert out.shape == x.shape


def test_topology_design_forward_runs_with_odd_depth():
    model = _build_tiny_jit(
        depth=5,
        topology_kwargs={
            "long_shortcuts": True,
            "dense_input_shortcuts": True,
            "adaln_lora_rank": 4,
        },
    )
    x = torch.randn(2, 3, 8, 8)
    t = torch.rand(2)
    y = torch.randint(0, 10, (2,))
    out = model(x, t, y)
    assert out.shape == x.shape


def test_adaln_single_forward_runs():
    model = _build_tiny_jit(topology_kwargs={"adaln_single": True})
    x = torch.randn(2, 3, 8, 8)
    t = torch.rand(2)
    y = torch.randint(0, 10, (2,))
    out = model(x, t, y)
    assert out.shape == x.shape


def test_time_dependent_qkv_forward_runs():
    model = _build_tiny_jit(
        attention_kwargs={"time_dependent_qkv": True},
        topology_kwargs={"adaln_mode": "shift_only"},
    )
    x = torch.randn(2, 3, 8, 8)
    t = torch.rand(2)
    y = torch.randint(0, 10, (2,))
    out = model(x, t, y)
    assert out.shape == x.shape


def test_patch_design_forward_runs():
    model = _build_tiny_jit(
        patch_kwargs={
            "adaptive_bottleneck": True,
            "multiscale_patch_embed": True,
            "fine_patch_size": 2,
        }
    )
    x = torch.randn(2, 3, 8, 8)
    t = torch.rand(2)
    y = torch.randint(0, 10, (2,))
    out = model(x, t, y)
    assert out.shape == x.shape


def test_spatial_injection_ffn_forward_runs():
    model = _build_tiny_jit(ffn_type="spatial_adaptive")
    x = torch.randn(2, 3, 8, 8)
    t = torch.rand(2)
    y = torch.randint(0, 10, (2,))
    out = model(x, t, y)
    assert out.shape == x.shape
