#!/usr/bin/env python3
"""
提交 sbatch 前在登录节点跑一遍，快速检查容易出问题的地方，避免排到卡才发现失败。
用法（在项目根目录、激活 VLM 后）:
  python check_training_setup.py
  python check_training_setup.py --config configs/jit_l16_in256_imagenet_ode.yaml
  python check_training_setup.py --no-forward   # 不跑前向，只测导入/配置/模型构建/wandb
"""
import argparse
import os
import sys

def step(name):
    print(f"[CHECK] {name} ...", end=" ", flush=True)

def ok():
    print("OK", flush=True)

def fail(msg):
    print("FAIL", flush=True)
    print(f"  -> {msg}", flush=True)
    sys.exit(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/jit_b16_in256_imagenet.yaml", help="用的 config")
    ap.add_argument("--no-forward", action="store_true", help="不跑一次前向，只测导入/配置/模型/wandb")
    args = ap.parse_args()

    # 1. 导入
    step("导入 yaml, torch, numpy")
    try:
        import yaml
        import torch
        import numpy as np
    except ImportError as e:
        fail(str(e))
    ok()

    step("导入 accelerate")
    try:
        from accelerate import Accelerator
    except ImportError as e:
        fail(str(e))
    ok()

    step("导入项目模块 (util, denoiser, model_jit, ffn_factory)")
    try:
        import util.misc as misc
        from util.crop import center_crop_arr
        from denoiser import Denoiser
        from main_jit import build_model_args
    except Exception as e:
        fail(str(e))
    ok()

    # 2. 配置
    step(f"加载配置 {args.config}")
    if not os.path.isfile(args.config):
        fail(f"文件不存在: {args.config}")
    try:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        assert "model" in cfg and "data" in cfg and "train" in cfg
    except Exception as e:
        fail(str(e))
    ok()

    # 3. 模型构建（部分代码 init 时用 .cuda()，无 GPU 的登录节点会报错，则跳过此项）
    step("构建 Denoiser 模型")
    try:
        model_args = build_model_args(cfg)
        model = Denoiser(model_args)
        n = sum(p.numel() for p in model.parameters())
        print(f"OK ({n/1e6:.2f}M params)", flush=True)
    except Exception as e:
        err = str(e).lower()
        if "cuda" in err or "nvidia" in err or "driver" in err:
            print("跳过 (无 GPU，部分模块 init 时用 .cuda()，排到卡后会自动正常)", flush=True)
        else:
            fail(str(e))

    # 4. Wandb 离线 init（和计算节点上一致）
    step("Wandb 离线 init")
    os.environ["WANDB_MODE"] = "offline"
    try:
        import wandb
        run = wandb.init(project=cfg["logging"].get("project", "jit-training"), mode="offline")
        run.finish()
    except Exception as e:
        fail(str(e))
    ok()

    # 5. 一次前向（小 batch，可 CPU）
    if args.no_forward:
        print("[CHECK] 跳过前向 (--no-forward)")
    else:
        step("一次前向 (batch=2)")
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            img_size = cfg["data"]["img_size"]
            class_num = cfg["data"]["class_num"]
            model = model.to(device)
            model.eval()
            x = torch.randn(2, 3, img_size, img_size, device=device)
            y = torch.randint(0, class_num, (2,), device=device)
            with torch.no_grad():
                loss = model(x, y)
            print(f"OK (loss={loss.item():.4f}, device={device})", flush=True)
        except Exception as e:
            print("FAIL", flush=True)
            print(f"  -> {e}", flush=True)
            print("  -> 若在登录节点无 GPU 或内存不足可加 --no-forward 跳过", flush=True)
            sys.exit(1)

    print("\n[CHECK] 全部通过，可以 sbatch 提交。", flush=True)

if __name__ == "__main__":
    main()
