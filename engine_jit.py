
import math
import os
import shutil

import torch
import numpy as np
import cv2
from tqdm import tqdm

import util.misc as misc
import util.lr_sched as lr_sched
# import torch_fidelity
import copy

from util.fid import calculate_fid_path

def _build_lr_args(cfg, base_lr):
    """Mimic the old args object for lr_sched.adjust_learning_rate."""
    class LRArgs:
        pass
    args = LRArgs()
    args.lr = base_lr
    args.min_lr = cfg["train"]["min_lr"]
    args.epochs = cfg["train"]["epochs"]
    args.warmup_epochs = cfg["train"]["warmup_epochs"]
    args.lr_schedule = cfg["train"]["lr_schedule"]
    return args


def train_one_epoch(
    accelerator,
    model,
    train_loader,
    optimizer,
    epoch,
    cfg,
    global_step: int,
):
    """One epoch of training with Accelerator + single EMA."""
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 20

    base_lr = optimizer.param_groups[0]["lr"]
    lr_args = _build_lr_args(cfg, base_lr)

    model.train(True)
    optimizer.zero_grad()

    if accelerator.is_main_process:
        print(f"Starting epoch {epoch}")

    log_every = cfg["train"]["log_every"]

    for data_iter_step, (x, labels) in enumerate(
        metric_logger.log_every(train_loader, print_freq, header)
    ):
        # -------------------------
        # Per-iteration LR scheduling
        # -------------------------
        it = data_iter_step / len(train_loader) + epoch
        lr_sched.adjust_learning_rate(optimizer, it, lr_args)

        x = x.to(accelerator.device, non_blocking=True).to(torch.float32).div_(255)
        x = x * 2.0 - 1.0
        labels = labels.to(accelerator.device, non_blocking=True)

        with accelerator.autocast():
            loss = model(x, labels)

        loss_value = loss.detach().float()

        if not math.isfinite(loss_value.item()):
            if accelerator.is_main_process:
                print(f"Loss is {loss_value.item()}, stopping training.")
            raise RuntimeError("Non-finite loss.")

        optimizer.zero_grad()
        accelerator.backward(loss)

        optimizer.step()

        # EMA update on unwrapped model
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.update_ema()

        # Metrics
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss_value.item())
        metric_logger.update(lr=lr)

        # All-reduce mean loss across processes
        loss_reduced = loss_value
        # Use accelerator.gather for cross-process stats
        with torch.no_grad():
            gathered = accelerator.gather(loss_reduced)
            loss_reduced = gathered.mean()

        global_step += 1

        # Logging via accelerator
        if accelerator.is_main_process and (data_iter_step % log_every == 0):
            log_dict = {
                "train/loss": loss_reduced.item(),
                "train/lr": lr,
                "train/epoch": epoch,
                "train/step": global_step,
            }
            accelerator.log(log_dict, step=global_step)

        # # debug
        # if data_iter_step == 50:
        #     break

    return global_step


def evaluate(
    accelerator,
    model,
    cfg,
    epoch: int,
    global_step: int,
    checkpoint_dir: str,
):
    """FID/IS evaluation with EMA weights."""
    device = accelerator.device
    world_size = accelerator.num_processes
    local_rank = accelerator.process_index

    unwrapped = accelerator.unwrap_model(model)
    unwrapped.eval()

    num_images = cfg["sample"]["num_images"]
    batch_size = cfg["sample"]["gen_bsz"]
    img_size = cfg["data"]["img_size"]
    class_num = cfg["data"]["class_num"]

    num_steps = num_images // (batch_size * world_size) + 1

    # Save folder for generated images (per evaluation)
    save_folder = os.path.join(
        cfg["sample"]["tmp_dir"],
        cfg["train"]["exp_name"],
        f"steps{unwrapped.steps}-cfg{unwrapped.cfg_scale}"
        f"-interval{unwrapped.cfg_interval[0]}-{unwrapped.cfg_interval[1]}"
        f"-image{num_images}-res{img_size}",
    )

    if accelerator.is_main_process:
        print(f"[Eval] Saving generated images to: {save_folder}")
        os.makedirs(save_folder, exist_ok=True)

    accelerator.wait_for_everyone()

    # Swap to EMA weights
    model_state_dict = copy.deepcopy(unwrapped.state_dict())
    ema_state_dict = copy.deepcopy(unwrapped.state_dict())
    for i, (name, _value) in enumerate(unwrapped.named_parameters()):
        assert hasattr(unwrapped, "ema_params"), "ema_params not found on model."
        ema_param = unwrapped.ema_params[i]
        ema_state_dict[name] = ema_param.to(device)
    if accelerator.is_main_process:
        print("[Eval] Switching to EMA weights.")
    unwrapped.load_state_dict(ema_state_dict)

    # Ensure equal number of images per class
    assert num_images % class_num == 0, "Number of images per class must be the same."
    class_label_gen_world = np.arange(0, class_num).repeat(num_images // class_num)
    # pad a bit to avoid OOB
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])

    for i in range(num_steps):
        if accelerator.is_main_process:
            print(f"[Eval] Generation step {i+1}/{num_steps}")

        start_idx = world_size * batch_size * i + local_rank * batch_size
        end_idx = start_idx + batch_size
        labels_gen = class_label_gen_world[start_idx:end_idx]
        labels_gen = torch.Tensor(labels_gen).long().to(device)

        if labels_gen.numel() == 0:
            continue

        with accelerator.autocast():
            sampled_images = unwrapped.generate(labels_gen)

        accelerator.wait_for_everyone()

        # Denormalize images from [-1, 1] to [0, 255]
        sampled_images = (sampled_images + 1) / 2
        sampled_images = sampled_images.detach().cpu()

        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            if img_id >= num_images:
                break
            gen_img = sampled_images[b_id].numpy().transpose(1, 2, 0)
            gen_img = np.round(np.clip(gen_img * 255, 0, 255)).astype(np.uint8)
            # BGR for OpenCV
            gen_img_bgr = gen_img[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, f"{str(img_id).zfill(5)}.png"), gen_img_bgr)

    accelerator.wait_for_everyone()

    # Switch back from EMA
    if accelerator.is_main_process:
        print("[Eval] Switching back from EMA weights.")
    unwrapped.load_state_dict(model_state_dict)

    # Compute FID / IS on main process
    if accelerator.is_main_process:
        if img_size == 256:
            fid_statistics_file = cfg["sample"]["fid_stats_256"]
        elif img_size == 512:
            fid_statistics_file = cfg["sample"]["fid_stats_512"]
        else:
            raise NotImplementedError(f"No FID stats configured for img_size={img_size}")

        # metrics_dict = torch_fidelity.calculate_metrics(
        #     input1=save_folder,
        #     input2=None,
        #     fid_statistics_file=fid_statistics_file,
        #     cuda=torch.cuda.is_available(),
        #     isc=True,
        #     fid=True,
        #     kid=False,
        #     prc=False,
        #     verbose=False,
        # )
        # fid = metrics_dict["frechet_inception_distance"]
        # inception_score = metrics_dict["inception_score_mean"]

        # log_dict = {
        #     "val/fid": fid,
        #     "val/is": inception_score,
        #     "val/epoch": epoch,
        #     "val/step": global_step,
        # }
        # accelerator.log(log_dict, step=global_step)
        # print(f"[Eval] FID: {fid:.4f}, Inception Score: {inception_score:.4f}")

        fid_score = calculate_fid_path(save_folder, ref_path=fid_statistics_file)
        log_dict = {
            "val/fid": fid_score,
            "val/epoch": epoch,
            "val/step": global_step,
        }
        accelerator.log(log_dict, step=global_step)
        print(f"[Eval] FID: {fid_score:.2f}")

        # Optionally clean temp folder
        shutil.rmtree(save_folder, ignore_errors=True)

    accelerator.wait_for_everyone()

    # Back to train mode
    unwrapped.train()
