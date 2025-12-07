
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
    return SimpleNamespace(
        model=cfg["model"]["name"],
        img_size=cfg["data"]["img_size"],
        class_num=cfg["data"]["class_num"],
        attn_dropout=cfg["model"]["attn_dropout"],
        proj_dropout=cfg["model"]["proj_dropout"],
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
        lr = cfg["train"]["blr"] * eff_batch_size / 256.0
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
    # Accelerator + logging
    # -------------------------
    accelerator = Accelerator(
        log_with=None if args.debug else "wandb",
        mixed_precision=cfg["train"]["mixed_precision"],
    )
    device = accelerator.device

    # Seed
    seed = cfg["train"]["seed"]
    # Use misc.get_rank() to keep behavior closer to original, but through accelerator, rank is:
    rank = accelerator.process_index
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)

    if accelerator.is_main_process and (not args.debug):
        init_kwargs = {"wandb": {"name": cfg["logging"]["run_name"]}}
        accelerator.init_trackers(
            cfg["logging"]["project"],
            config=cfg,
            init_kwargs=init_kwargs,
        )

    # -------------------------
    # Data
    # -------------------------
    transform_train = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, cfg["data"]["img_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.PILToTensor(),
    ])

    dataset_train = datasets.ImageFolder(
        os.path.join(cfg["data"]["data_path"], "train"),
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
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))

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

    accelerator.end_training()


if __name__ == "__main__":
    main()
