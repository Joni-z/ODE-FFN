import pathlib
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import denoiser as denoiser_module
from denoiser import Denoiser
from main_jit import build_model_args


class DummyFlowNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, z, t, labels):
        del t, labels
        return self.scale * z


def _make_args(**overrides):
    base = dict(
        model="dummy",
        img_size=8,
        class_num=10,
        attn_dropout=0.0,
        proj_dropout=0.0,
        ffn_type="swiglu",
        ffn_kwargs=None,
        P_mean=-0.8,
        P_std=0.8,
        noise_scale=1.0,
        t_eps=5.0e-2,
        label_drop_prob=0.0,
        sampling_method="heun",
        num_sampling_steps=4,
        cfg=1.0,
        interval_min=0.0,
        interval_max=1.0,
        ema_decay1=0.999,
        soft_lipschitz_enabled=False,
        soft_lipschitz_lambda=0.0,
        soft_lipschitz_num_samples=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_build_model_args_reads_soft_lipschitz_config():
    cfg = {
        "model": {
            "name": "JiT-B/16",
            "ffn_type": "swiglu",
            "attn_dropout": 0.0,
            "proj_dropout": 0.0,
        },
        "data": {
            "img_size": 256,
            "class_num": 1000,
        },
        "diffusion": {
            "P_mean": -0.8,
            "P_std": 0.8,
            "noise_scale": 1.0,
            "t_eps": 5.0e-2,
            "label_drop_prob": 0.1,
        },
        "sample": {
            "sampling_method": "heun",
            "num_sampling_steps": 50,
            "cfg": 2.9,
            "interval_min": 0.1,
            "interval_max": 1.0,
        },
        "train": {
            "ema_decay": 0.9999,
        },
        "loss": {
            "soft_lipschitz": {
                "enabled": True,
                "lambda": 0.05,
                "num_samples": 8,
            }
        },
    }

    args = build_model_args(cfg)

    assert args.ffn_type == "swiglu"
    assert args.soft_lipschitz_enabled is True
    assert args.soft_lipschitz_lambda == 0.05
    assert args.soft_lipschitz_num_samples == 8


def test_build_model_args_adds_cond_dim_for_freq_split():
    cfg = {
        "model": {
            "name": "JiT-B/16",
            "ffn_type": "freq_split",
            "attn_dropout": 0.0,
            "proj_dropout": 0.0,
        },
        "data": {
            "img_size": 256,
            "class_num": 1000,
        },
        "diffusion": {
            "P_mean": -0.8,
            "P_std": 0.8,
            "noise_scale": 1.0,
            "t_eps": 5.0e-2,
            "label_drop_prob": 0.1,
        },
        "sample": {
            "sampling_method": "heun",
            "num_sampling_steps": 50,
            "cfg": 2.9,
            "interval_min": 0.1,
            "interval_max": 1.0,
        },
        "train": {
            "ema_decay": 0.9999,
        },
    }

    args = build_model_args(cfg)

    assert args.ffn_type == "freq_split"
    assert args.ffn_kwargs["t_embed_dim"] == 768


def test_denoiser_returns_soft_lipschitz_breakdown():
    old_models = denoiser_module.JiT_models
    denoiser_module.JiT_models = {"dummy": DummyFlowNet}
    try:
        model = Denoiser(_make_args(soft_lipschitz_enabled=True, soft_lipschitz_lambda=0.1))
        model.train()

        x = torch.randn(2, 3, 8, 8)
        labels = torch.randint(0, 10, (2,))
        loss_dict = model(x, labels, return_loss_dict=True)

        assert set(loss_dict) == {"loss", "loss_fm", "loss_lip"}
        assert loss_dict["loss"].ndim == 0
        assert loss_dict["loss_fm"].ndim == 0
        assert loss_dict["loss_lip"].ndim == 0
        assert loss_dict["loss_lip"].item() >= 0.0
        assert torch.allclose(
            loss_dict["loss"],
            loss_dict["loss_fm"] + 0.1 * loss_dict["loss_lip"],
        )
    finally:
        denoiser_module.JiT_models = old_models


def test_denoiser_soft_lipschitz_defaults_to_zero_when_disabled():
    old_models = denoiser_module.JiT_models
    denoiser_module.JiT_models = {"dummy": DummyFlowNet}
    try:
        model = Denoiser(_make_args())
        model.train()

        x = torch.randn(2, 3, 8, 8)
        labels = torch.randint(0, 10, (2,))
        loss_dict = model(x, labels, return_loss_dict=True)

        assert loss_dict["loss_lip"].item() == 0.0
        assert torch.allclose(loss_dict["loss"], loss_dict["loss_fm"])
    finally:
        denoiser_module.JiT_models = old_models


def test_denoiser_soft_lipschitz_supports_subsampled_batch():
    old_models = denoiser_module.JiT_models
    denoiser_module.JiT_models = {"dummy": DummyFlowNet}
    try:
        model = Denoiser(
            _make_args(
                soft_lipschitz_enabled=True,
                soft_lipschitz_lambda=0.1,
                soft_lipschitz_num_samples=1,
            )
        )
        model.train()

        x = torch.randn(4, 3, 8, 8)
        labels = torch.randint(0, 10, (4,))
        loss_dict = model(x, labels, return_loss_dict=True)

        assert loss_dict["loss_lip"].ndim == 0
        assert loss_dict["loss_lip"].item() >= 0.0
    finally:
        denoiser_module.JiT_models = old_models
