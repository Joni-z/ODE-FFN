
import argparse
import os
import time
import datetime
from types import SimpleNamespace

import yaml
import torch
import numpy as np
from accelerate import Accelerator
torch.set_float32_matmul_precision('high')

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import util.misc as misc
from util.crop import center_crop_arr

from denoiser import Denoiser
from engine_jit import train_one_epoch, evaluate


def build_model_args(cfg):
    """Wrap config into an args-like object for Denoiser."""
    model_cfg = cfg["model"]
    ffn_type = model_cfg.get("ffn_type", "swiglu")
    ffn_kwargs = model_cfg.get("ffn_kwargs")  # optional dict for ode/ode_swiglu (tau, scale, shift, orders, ode_hidden_features)
    return SimpleNamespace(
        model=model_cfg["name"],
        img_size=cfg["data"]["img_size"],
        class_num=cfg["data"]["class_num"],
        attn_dropout=model_cfg["attn_dropout"],
        proj_dropout=model_cfg["proj_dropout"],
        ffn_type=ffn_type,
        ffn_kwargs=ffn_kwargs,
        P_mean=cfg["diffusion"]["P_mean"],
        P_std=cfg["diffusion"]["P_std"],
        noise_scale=cfg["diffusion"]["noise_scale"],
        t_eps=cfg["diffusion"]["t_eps"],
        label_drop_prob=cfg["diffusion"]["label_drop_prob"],
        sampling_method=cfg["sample"]["sampling_method"],
        num_sampling_steps=cfg["sample"]["num_sampling_steps"],
        cfg=cfg["sample"]["cfg"],
        interval_min=cfg["sample"]["interval_min"],
        interval_max=cfg["sample"]["interval_max"],
        ema_decay1=cfg["train"]["ema_decay"],   # keep field name for compatibility
    )


def get_lr_and_batch(cfg, accelerator):
    eff_batch_size = cfg["train"]["per_device_batch_size"] * accelerator.num_processes
    if cfg["train"]["lr"] is None:
        lr = cfg["train"]["blr"] * eff_batch_size / 512.0
    else:
        lr = cfg["train"]["lr"]

    if accelerator.is_main_process:
        print(f"Effective batch size: {eff_batch_size}")
        print(f"Base lr (blr): {cfg['train']['blr']:.2e}")
        print(f"Actual lr: {lr:.2e}")

    return lr, eff_batch_size


def main():
    # -------------------------
    # Argparse: config + debug
    # -------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    # -------------------------
    # Load YAML config
    # -------------------------
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.debug:
        # Lighter / faster debug settings
        cfg["train"]["epochs"] = min(5, cfg["train"]["epochs"])
        cfg["train"]["log_every"] = 10
        cfg["train"]["save_every"] = 1
        cfg["train"]["eval_every"] = max(1, cfg["train"]["eval_every"])
        cfg["sample"]["num_images"] = min(1000, cfg["sample"]["num_images"])

    # -------------------------
    # Accelerator (W&B 用下面直接 wandb.init，不用 log_with)
    # -------------------------
    accelerator = Accelerator(
        log_with=None,
        mixed_precision=cfg["train"]["mixed_precision"],
    )
    device = accelerator.device

    # Seed
    seed = cfg["train"]["seed"]
    rank = accelerator.process_index
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)

    # W&B：与 ODE_FFN 一致，主进程直接 wandb.init，训练里用 wandb.log
    wandb_enabled = False
    if accelerator.is_main_process and (not args.debug):
        wandb_cfg = cfg.get("wandb", cfg.get("logging", {}))
        wandb_enabled = bool(wandb_cfg.get("enabled", True))
        if wandb_enabled:
            import wandb
            run = wandb.init(
                project=wandb_cfg.get("project", cfg["logging"].get("project", "jit-training")),
                name=wandb_cfg.get("name", cfg["logging"].get("run_name", cfg["train"]["exp_name"])),
                config=cfg,
                resume=wandb_cfg.get("resume", False),
            )
            print(f"W&B 已登录: 账号={run.entity} | project={run.project} | run={run.name}")

    # -------------------------
    # Data
    # -------------------------
    data_path = cfg["data"]["data_path"]
    train_dir = os.path.join(data_path, "train")
    if not os.path.isdir(train_dir):
        if accelerator.is_main_process:
            raise FileNotFoundError(
                f"训练数据目录不存在: {train_dir}\n"
                f"请在各 config 的 data.data_path 中填写本集群上 ImageNet 的实际路径（需含 train/ 子目录）。"
            )
        raise RuntimeError("data_path not found")

    transform_train = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, cfg["data"]["img_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.PILToTensor(),
    ])

    dataset_train = datasets.ImageFolder(
        train_dir,
        transform=transform_train,
    )

    if accelerator.is_main_process:
        print(dataset_train)

    train_dl = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=cfg["train"]["per_device_batch_size"],
        shuffle=True,  # sampler will be replaced by accelerator if distributed
        num_workers=cfg["train"]["num_workers"],
        pin_memory=cfg["train"]["pin_mem"],
        drop_last=True,
    )

    # -------------------------
    # Model & optimizer
    # -------------------------
    model_args = build_model_args(cfg)
    model = Denoiser(model_args)

    if accelerator.is_main_process:
        print("Model =", model)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {n_params / 1e6:.6f}M")

    lr, eff_batch_size = get_lr_and_batch(cfg, accelerator)
    param_groups = misc.add_weight_decay(model, weight_decay=cfg["train"]["weight_decay"])
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, cfg["train"].get("beta2", 0.99)))

    # -------------------------
    # Prepare for distributed / mixed precision
    # -------------------------
    model, optimizer, train_dl = accelerator.prepare(model, optimizer, train_dl)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.ema_params = [p.detach().clone().to(p.device) for p in unwrapped.parameters()]

    # -------------------------
    # Checkpoint directory
    # -------------------------
    out_root = cfg["checkpointing"]["out_dir"]
    exp_name = cfg["train"]["exp_name"]
    checkpoint_dir = os.path.join(out_root, exp_name)
    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # -------------------------
    # Optional resume
    # -------------------------
    global_step = 0
    start_epoch = 0
    resume_path = cfg["checkpointing"]["resume"]
    if resume_path:
        if accelerator.is_main_process:
            print(f"Resuming from checkpoint: {resume_path}")
        map_location = "cpu"
        checkpoint = torch.load(resume_path, map_location=map_location)

        unwrapped = accelerator.unwrap_model(model)
        unwrapped.load_state_dict(checkpoint["model"])

        if "ema" in checkpoint:
            ema_state = checkpoint["ema"]
            # Load EMA into unwrapped.ema_params as a list
            # by matching named parameters
            named = dict(unwrapped.named_parameters())
            unwrapped.ema_params = []
            for name, param in named.items():
                assert name in ema_state
                ema_param = ema_state[name].to(param.device)
                unwrapped.ema_params.append(ema_param)
        else:
            # fallback: init EMA from the loaded model
            unwrapped.ema_params = [p.detach().clone() for p in unwrapped.parameters()]

        optimizer.load_state_dict(checkpoint["opt"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        global_step = checkpoint.get("step", 0)

        del checkpoint

    # -------------------------
    # Training loop
    # -------------------------
    if accelerator.is_main_process:
        print(f"Start training for {cfg['train']['epochs']} epochs")
    start_time = time.time()

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        model.train()
        global_step = train_one_epoch(
            accelerator=accelerator,
            model=model,
            train_loader=train_dl,
            optimizer=optimizer,
            epoch=epoch,
            cfg=cfg,
            global_step=global_step,
        )

        # -------------------------
        # Save checkpoint
        # -------------------------
        if ((epoch + 1) % cfg["train"]["save_every"] == 0) or ((epoch + 1) == cfg["train"]["epochs"]):
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)

                # Build EMA state dict
                ema_state_dict = {}
                named_params = dict(unwrapped.named_parameters())
                for (name, param), ema_param in zip(named_params.items(), unwrapped.ema_params):
                    ema_state_dict[name] = ema_param.detach().clone().cpu()

                checkpoint = {
                    "model": unwrapped.state_dict(),
                    "ema": ema_state_dict,
                    "opt": optimizer.state_dict(),
                    "epoch": epoch,
                    "step": global_step,
                    "config": cfg,
                }
                last_path = os.path.join(checkpoint_dir, "last.pt")
                torch.save(checkpoint, last_path)
                print(f"[Epoch {epoch}] Saved last checkpoint to {last_path}")

        # -------------------------
        # Online evaluation (FID/IS)
        # -------------------------
        if cfg["sample"]["eval_online"]:
            run_eval = ((epoch + 1) % cfg["train"]["eval_every"] == 0) or ((epoch + 1) == cfg["train"]["epochs"])
            if run_eval:
                evaluate(
                    accelerator=accelerator,
                    model=model,
                    cfg=cfg,
                    epoch=epoch,
                    global_step=global_step,
                    checkpoint_dir=checkpoint_dir,
                )

    # -------------------------
    # Done
    # -------------------------
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if accelerator.is_main_process:
        print("Training time:", total_time_str)
        if wandb_enabled:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.finish()
            except Exception:
                pass

    accelerator.end_training()


if __name__ == "__main__":
    main()
