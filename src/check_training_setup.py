#!/usr/bin/env python3
"""
提交 sbatch 前在登录节点跑一遍，快速检查容易出问题的地方，避免排到卡才发现失败。
用法（在项目根目录、激活 VLM 后）:
  python src/check_training_setup.py
  python src/check_training_setup.py --config configs/jit_l16_in256_imagenet_ode.yaml
  python src/check_training_setup.py --no-forward   # 不跑前向
  python src/check_training_setup.py --no-wandb    # 不测 wandb 登录（如无网或未 login）
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
    ap.add_argument("--no-forward", action="store_true", help="不跑一次前向")
    ap.add_argument("--no-wandb", action="store_true", help="不测 wandb 登录")
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

    step("导入项目模块 (util, denoiser, main_jit)")
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

    # 3. 数据路径（与 main_jit 一致，避免排到卡才发现目录不存在）
    step("数据路径 data_path/train")
    data_path = cfg["data"].get("data_path", "")
    train_dir = os.path.join(data_path, "train") if data_path else ""
    if not data_path or not os.path.isdir(train_dir):
        fail(f"目录不存在: {train_dir}，请检查 config 里 data.data_path")
    n_dirs = len([x for x in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, x))])
    print(f"OK ({n_dirs} 类)", flush=True)

    # 4. 模型构建（无 GPU 时可能因 .cuda() 跳过）
    step("构建 Denoiser 模型")
    model = None
    try:
        model_args = build_model_args(cfg)
        model = Denoiser(model_args)
        n = sum(p.numel() for p in model.parameters())
        print(f"OK ({n/1e6:.2f}M params)", flush=True)
    except Exception as e:
        err = str(e).lower()
        if "cuda" in err or "nvidia" in err or "driver" in err:
            print("跳过 (无 GPU，排到卡后会自动正常)", flush=True)
        else:
            fail(str(e))

    # 5. Wandb 登录（在线 init，能上传则显示账号；401 时仅警告不拦提交）
    if args.no_wandb:
        print("[CHECK] 跳过 Wandb (--no-wandb)")
    else:
        step("Wandb 登录 (在线)")
        try:
            import wandb
            run = wandb.init(
                project=cfg.get("logging", {}).get("project", "jit-training"),
                name=cfg.get("logging", {}).get("run_name", "check"),
                config=cfg,
            )
            print(f"OK (账号={run.entity} | project={run.project})", flush=True)
            run.finish()
        except Exception as e:
            err = str(e)
            if "401" in err or "user is not logged in" in err or "PERMISSION" in err:
                print("警告 (401)，继续", flush=True)
                print("  -> 网页端可能看不到 run；可到 https://wandb.ai/authorize 复制新 key 后 python -m wandb login --relogin", flush=True)
            else:
                fail(f"{e}；或加 --no-wandb 跳过")

    # 6. 一次前向（可选；无 GPU 或未构建模型时跳过）
    if args.no_forward or model is None:
        print("[CHECK] 跳过前向 (--no-forward 或模型未构建)")
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
            print("  -> 登录节点无 GPU/内存不足时可加 --no-forward", flush=True)
            sys.exit(1)

    print("\n[CHECK] 全部通过，可以 sbatch 提交。", flush=True)

if __name__ == "__main__":
    main()
