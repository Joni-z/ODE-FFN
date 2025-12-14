# train.py
import argparse
import os
import time
import datetime
import wandb
import yaml
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from accelerate import Accelerator

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models import PatchAE

# If you have your own crop util, plug it in.
# Here we keep it minimal and rely on torchvision CenterCrop.
def build_transforms(img_size: int):
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),  # -> [0,1]
    ])


@torch.no_grad()
def compute_psnr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8):
    # x,y in [0,1]
    mse = F.mse_loss(x, y, reduction="mean").clamp_min(eps)
    return (-10.0 * torch.log10(mse)).item()


@torch.no_grad()
def log_recon_wandb(accelerator, x01, recon01, step: int, tag: str = "val/recon"):
    """
    x01, recon01: [B,3,H,W] in [0,1]
    Logs a grid of 8 rows where each row is [input | recon].
    """
    if not accelerator.is_main_process:
        return
    if accelerator.trackers is None or len(accelerator.trackers) == 0:
        return

    k = min(8, x01.shape[0])
    x = x01[:k].detach().float().clamp(0, 1).cpu()
    r = recon01[:k].detach().float().clamp(0, 1).cpu()

    # [k,3,H,2W] : concat input and recon horizontally
    pair = torch.cat([x, r], dim=-1)

    # grid: k rows, 1 column (so each row is one sample pair)
    grid = torchvision.utils.make_grid(pair, nrow=1, padding=2)

    # Convert to HWC uint8 for wandb.Image
    grid_u8 = (grid * 255.0).round().byte().permute(1, 2, 0).numpy()

    accelerator.log({tag: [wandb.Image(grid_u8, caption=f"step={step}")]}, step=step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.debug:
        cfg["train"]["epochs"] = min(2, cfg["train"]["epochs"])
        cfg["train"]["log_every"] = 10
        cfg["train"]["val_every"] = 30
        cfg["train"]["save_every"] = 1

    accelerator = Accelerator(
        log_with=None if args.debug else ("wandb" if cfg["logging"]["use_wandb"] else None),
        mixed_precision=cfg["train"]["mixed_precision"],
    )
    device = accelerator.device

    # seed
    seed = cfg["train"]["seed"]
    rank = accelerator.process_index
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)

    if accelerator.is_main_process and (not args.debug) and cfg["logging"]["use_wandb"]:
        init_kwargs = {"wandb": {"name": cfg["train"]["exp_name"]}}
        accelerator.init_trackers(cfg["logging"]["project"], config=cfg, init_kwargs=init_kwargs)

    # data
    img_size = cfg["data"]["img_size"]
    train_tf = build_transforms(img_size)

    dataset_train = datasets.ImageFolder(
        os.path.join(cfg["data"]["data_path"], "train"),
        transform=train_tf,
    )

    if accelerator.is_main_process:
        print(dataset_train)

    train_dl = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=cfg["train"]["per_device_batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=cfg["train"]["pin_mem"],
        drop_last=True,
    )

    # model
    model = PatchAE(
        img_size=cfg["data"]["img_size"],
        patch_size=cfg["model"]["patch_size"],
        in_chans=cfg["model"]["in_chans"],
        latent_dim=cfg["model"]["latent_dim"],
        arch=cfg["model"]["arch"],
        ffn_ratio=cfg["model"]["ffn_ratio"],
        drop=cfg["model"]["drop"],
        bias=cfg["model"]["bias"],
        normalize_patches=cfg["model"]["normalize_patches"],
        clamp_output=cfg["model"]["clamp_output"],
    )

    # optim
    lr = cfg["train"]["lr"]
    wd = cfg["train"]["weight_decay"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, cfg["train"]["beta2"]))

    # prepare
    model, optimizer, train_dl = accelerator.prepare(model, optimizer, train_dl)

    # checkpoint dir
    out_root = cfg["checkpointing"]["out_dir"]
    exp_name = cfg["train"]["exp_name"]
    ckpt_dir = os.path.join(out_root, exp_name)
    if accelerator.is_main_process:
        os.makedirs(ckpt_dir, exist_ok=True)

    # optional resume
    global_step = 0
    start_epoch = 0
    resume_path = cfg["checkpointing"]["resume"]
    if resume_path:
        if accelerator.is_main_process:
            print(f"Resuming from: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu")
        accelerator.unwrap_model(model).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["opt"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        global_step = checkpoint.get("step", 0)
        del checkpoint

    # training loop
    if accelerator.is_main_process:
        print(f"Start training for {cfg['train']['epochs']} epochs")

    start_time = time.time()

    from tqdm import tqdm
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        model.train()

        for it, (images, _) in enumerate(train_dl):
            # images already float [0,1], shape [B,3,H,W]
            recon, z = model(images)

            # losses
            loss_mse = F.mse_loss(recon, images, reduction="mean")

            # optional latent regularization (very light)
            loss_lat = torch.tensor(0.0, device=device)
            if cfg["train"]["latent_l2_weight"] > 0:
                loss_lat = (z.pow(2).mean()) * cfg["train"]["latent_l2_weight"]

            loss = loss_mse + loss_lat

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            if cfg["train"]["clip_grad_norm"] is not None:
                accelerator.clip_grad_norm_(model.parameters(), cfg["train"]["clip_grad_norm"])
            optimizer.step()

            global_step += 1

            # logging
            if accelerator.is_main_process and (global_step % cfg["train"]["log_every"] == 0):
                psnr = compute_psnr(images.detach(), recon.detach())
                logs = {
                    "train/loss": loss.item(),
                    "train/loss_mse": loss_mse.item(),
                    "train/loss_latent_l2": loss_lat.item() if isinstance(loss_lat, torch.Tensor) else float(loss_lat),
                    "train/psnr": psnr,
                    "train/epoch": epoch,
                    "train/step": global_step,
                }

                if cfg["logging"]["use_wandb"] and (not args.debug):
                    accelerator.log(logs, step=global_step)
                else:
                    print(global_step, {k: round(v, 6) if isinstance(v, float) else v for k, v in logs.items()})

            # every N steps, quick "validation" visual
            if accelerator.is_main_process and (global_step % cfg["train"]["val_every"] == 0):
                log_recon_wandb(accelerator, images, recon, step=global_step, tag="val/recon")

        # save checkpoint
        if ((epoch + 1) % cfg["train"]["save_every"] == 0) or ((epoch + 1) == cfg["train"]["epochs"]):
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                ckpt = {
                    "model": unwrapped.state_dict(),
                    "opt": optimizer.state_dict(),
                    "epoch": epoch,
                    "step": global_step,
                    "config": cfg,
                }
                last_path = os.path.join(ckpt_dir, "last.pt")
                torch.save(ckpt, last_path)
                print(f"[Epoch {epoch}] Saved checkpoint: {last_path}")

    total_time = time.time() - start_time
    if accelerator.is_main_process:
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time:", total_time_str)

    accelerator.end_training()


if __name__ == "__main__":
    # Optional: helps on Ampere+ for stable speed
    torch.set_float32_matmul_precision("high")
    main()
