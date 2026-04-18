import pathlib
import sys

import torch
import pytest


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

    for ffn_type in ("time_split", "freq_split", "clean_target", "time_moe"):
        layer = build_ffn(ffn_type, in_features=16, hidden_features=64, t_embed_dim=16)
        out = layer(x, cond)
        assert out.shape == x.shape


def test_new_ffn_aliases_normalize():
    assert normalize_ffn_type("time_split_dual_path") == "time_split"
    assert normalize_ffn_type("frequency_split") == "freq_split"
    assert normalize_ffn_type("clean_target_ffn") == "clean_target"
    assert normalize_ffn_type("time_routed_moe") == "time_moe"


def test_six_designs_build_and_forward_square():
    """All 6 PDF designs with square sequence (256 = 16x16 patches)."""
    x = torch.randn(2, 16, 32)  # square: 4x4
    cond = torch.randn(2, 32)

    for ffn_type in (
        "ta_gate", "flow_evolved_gate", "spatial_adaptive",
        "freq_split_dual", "progressive_refine", "multistep_ffn",
    ):
        layer = build_ffn(ffn_type, in_features=32, hidden_features=128, t_embed_dim=32)
        out = layer(x, cond)
        assert out.shape == x.shape, f"{ffn_type}: expected {x.shape}, got {out.shape}"


def test_six_designs_build_and_forward_nonsquare():
    """All 6 PDF designs with non-square sequence (20 tokens, not a perfect square)."""
    x = torch.randn(2, 20, 32)
    cond = torch.randn(2, 32)

    for ffn_type in (
        "ta_gate", "flow_evolved_gate", "spatial_adaptive",
        "freq_split_dual", "progressive_refine", "multistep_ffn",
    ):
        layer = build_ffn(ffn_type, in_features=32, hidden_features=128, t_embed_dim=32)
        out = layer(x, cond)
        assert out.shape == x.shape, f"{ffn_type} non-square: expected {x.shape}, got {out.shape}"


def test_six_designs_return_aux():
    """All 6 designs support return_aux=True."""
    x = torch.randn(2, 9, 16)
    cond = torch.randn(2, 16)

    for ffn_type in (
        "ta_gate", "flow_evolved_gate", "spatial_adaptive",
        "freq_split_dual", "progressive_refine", "multistep_ffn",
    ):
        layer = build_ffn(ffn_type, in_features=16, hidden_features=64, t_embed_dim=16)
        result = layer(x, cond, return_aux=True)
        assert isinstance(result, tuple) and len(result) == 2, f"{ffn_type}: return_aux failed"
        out, aux = result
        assert out.shape == x.shape
        assert "input_norm" in aux and "output_norm" in aux


def test_six_designs_no_cond():
    """All 6 designs work without conditioning."""
    x = torch.randn(2, 9, 16)

    for ffn_type in (
        "ta_gate", "flow_evolved_gate", "spatial_adaptive",
        "freq_split_dual", "progressive_refine", "multistep_ffn",
    ):
        layer = build_ffn(ffn_type, in_features=16, hidden_features=64)
        out = layer(x)
        assert out.shape == x.shape, f"{ffn_type} no cond: expected {x.shape}, got {out.shape}"


def test_six_designs_aliases():
    assert normalize_ffn_type("timestep_adaptive_gate") == "ta_gate"
    assert normalize_ffn_type("flow_evolved") == "flow_evolved_gate"
    assert normalize_ffn_type("spatial_adaptive_value") == "spatial_adaptive"
    assert normalize_ffn_type("freq_split_dual_ffn") == "freq_split_dual"
    assert normalize_ffn_type("progressive_refine_ffn") == "progressive_refine"
    assert normalize_ffn_type("multistep_integrated") == "multistep_ffn"


def test_freq_split_dual_uses_configured_kernel():
    layer = build_ffn(
        "freq_split_dual",
        in_features=16,
        hidden_features=64,
        t_embed_dim=16,
        lowpass_kernel=5,
    )
    assert layer.dw_conv.kernel_size == (5, 5)
    assert layer.dw_conv.padding == (2, 2)


@pytest.mark.parametrize("lowpass_kernel", [0, 2, 4])
def test_freq_split_dual_rejects_invalid_lowpass_kernel(lowpass_kernel):
    with pytest.raises(ValueError, match="lowpass_kernel must be a positive odd integer"):
        build_ffn(
            "freq_split_dual",
            in_features=16,
            hidden_features=64,
            t_embed_dim=16,
            lowpass_kernel=lowpass_kernel,
        )


@pytest.mark.parametrize("steps", [0, -1, 1.5])
def test_multistep_ffn_rejects_invalid_steps(steps):
    with pytest.raises(ValueError, match="steps must be a positive integer"):
        build_ffn(
            "multistep_ffn",
            in_features=16,
            hidden_features=64,
            t_embed_dim=16,
            steps=steps,
        )
