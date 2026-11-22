"""Train to one immutable milestone or score its EMA in a separate Slurm job."""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main_jit import main as train
from util.midpoint_eval import evaluate_pair


def checkpoint_args(path):
    state = torch.load(path, mmap=True, weights_only=False, map_location='cpu')
    cfg = state['args']
    assert (cfg.ffn, cfg.ode_hidden_dim, cfg.dataset, cfg.img_size, cfg.seed, cfg.batch_size) == (
        'midpoint', 1200, 'celeba', 128, 0, 256)
    assert (cfg.sampling_method, cfg.num_sampling_steps, cfg.num_images, cfg.gen_bsz) == ('heun', 50, 50000, 256)
    assert state['optimizer']['state'] and state['rng_state']['cuda'] and state['model_ema1'] and state['model_ema2']
    return state['epoch'] + 1, cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['train', 'fid'])
    parser.add_argument('epoch', type=int, choices=range(20, 161, 20))
    for name in ('data', 'run', 'native', 'validation'):
        parser.add_argument('--' + name, type=Path, required=True)
    stage = parser.parse_args()
    proof = json.loads(stage.validation.read_text())
    assert proof['status'] == 'PASS' and all(proof['results'][mode]['max_abs_difference'] == 0
        for mode in ('midpoint2', 'midpoint4', 'swiglu'))
    last = stage.run / 'checkpoint-last.pth'
    milestone = stage.run / f'checkpoint-{stage.epoch:04d}.pth'
    completed, cfg = checkpoint_args(last if stage.phase == 'train' else milestone)
    cfg.output_dir = str(stage.run)
    cfg.data_path = str(stage.data)
    cfg.native_checkpoint_dir = stage.native
    cfg.device = 'cuda'
    if stage.phase == 'train':
        if not stage.epoch - 20 <= completed <= stage.epoch:
            raise ValueError(f'Unexpected continuation {completed} -> {stage.epoch}')
        if completed < stage.epoch:
            cfg.resume = str(last)
            cfg.epochs = stage.epoch
            cfg.max_train_seconds = 0
            cfg.online_eval = False
            cfg.paired_midpoint_eval = False
            cfg.evaluate_gen = False
            train(cfg)
        assert checkpoint_args(last)[0] == stage.epoch
        # save_model atomically replaces last. This link keeps this exact file on later saves.
        if not milestone.exists():
            os.link(last, milestone)
        if not milestone.samefile(last):
            raise ValueError('A different milestone already exists')
    else:
        assert completed == stage.epoch
        folder = stage.run / 'paired' / f'epoch-{stage.epoch:04d}'
        folder.mkdir(parents=True, exist_ok=True)
        with SummaryWriter(str(folder)) as writer:
            row = evaluate_pair(cfg, stage.epoch, writer, detached=True)
        assert row['num_images'] == 50000 and row['network_forwards_per_batch'] == 99
        from util.misc import write_progress
        cfg.output_dir = str(folder)
        write_progress(cfg, status='complete', completed_epochs=stage.epoch)


if __name__ == '__main__':
    main()
