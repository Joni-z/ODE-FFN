"""Evaluate both internal solvers on one numbered EMA and time native SwiGLU on the same GPU."""
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

import numpy as np
import torch

from denoiser import Denoiser
from engine_jit import evaluate
from ffn import MidpointLayer
from util.misc import write_progress

REFERENCE = Path(__file__).resolve().parents[1] / 'docs/celeba128-wide-20261114/continuation-160/matched-FID.csv'


def atomic_json(path, value):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ema_model(path, epoch, family, device, args):
    state = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    cfg = state['args']
    assert state['epoch'] + 1 == epoch and cfg.ffn == family and cfg.ema_decay1 == args.ema_decay1
    for key in ('seed', 'model', 'img_size', 'class_num', 'dataset', 'noise_scale', 't_eps'):
        assert getattr(cfg, key) == getattr(args, key), key
    model = Denoiser(cfg)
    model.load_state_dict(state['model_ema1'], strict=True)
    assert (model.steps, model.method, model.cfg_scale, model.cfg_interval) == (
        args.num_sampling_steps, args.sampling_method, args.cfg, (args.interval_min, args.interval_max))
    return model.to(device).eval()


@torch.no_grad()
def speed(model, batch, seed, device):
    labels = torch.zeros(batch, device=device, dtype=torch.long)
    times = []
    sync = torch.cuda.synchronize if device.type == 'cuda' else lambda: None
    torch.manual_seed(seed)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        for i in range(4):
            sync()
            start = time.perf_counter()
            image = model.generate(labels)
            sync()
            duration = time.perf_counter() - start
            if not torch.isfinite(image).all():
                raise FloatingPointError('Nonfinite sample during throughput measurement')
            if i:
                times.append(duration)
    median = statistics.median(times)
    return dict(batch=batch, batch_seconds=times, median_batch_seconds=median, images_per_second=batch / median)


def evaluate_pair(args, epoch, writer):
    output = Path(args.output_dir)
    checkpoint = output / f'checkpoint-{epoch:04d}.pth'
    folder = output / 'paired' / f'epoch-{epoch:04d}'
    folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / 'comparison.json'
    digest = file_hash(checkpoint)
    if receipt.exists():
        row = json.loads(receipt.read_text())
        if row['checkpoint_sha256'] != digest:
            raise ValueError('Completed comparison belongs to different weights')
        return row
    device = torch.device(args.device)
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        with torch.random.fork_rng(), torch.no_grad():
            model = ema_model(checkpoint, epoch, 'midpoint', device, args)
            layers = [m for m in model.modules() if isinstance(m, MidpointLayer)]
            assert len(layers) == 2 * len(model.net.blocks) and layers
            comparisons = {}
            for steps in (2, 4):
                for m in layers:
                    m.substeps = steps
                mode = f'midpoint{steps}'
                write_progress(args, status='evaluating', completed_epochs=epoch, mode=mode,
                               target_epochs=args.epochs, evaluation_images=args.num_images)
                timed = speed(model, args.gen_bsz, args.seed, device)
                # Cache fully finished FIDs. Speed is measured together across all modes on recovery.
                fid_file = folder / f'{mode}.json'
                if fid_file.exists():
                    saved = json.loads(fid_file.read_text())
                    assert saved['checkpoint_sha256'] == digest and saved['num_images'] == args.num_images
                    fid = saved['fid']
                else:
                    arm = copy.copy(args)
                    arm.output_dir = str(folder / mode)
                    torch.manual_seed(args.seed)
                    np.random.seed(args.seed)
                    random.seed(args.seed)
                    class ModeWriter:
                        def add_scalar(self, name, value, index):
                            writer.add_scalar(f'{name}/{mode}', value, index)
                    fid = evaluate(model, arm, epoch, batch_size=args.gen_bsz,
                                   log_writer=ModeWriter(), preloaded_ema=True)
                    if not math.isfinite(fid):
                        raise ValueError('Nonfinite FID')
                    atomic_json(fid_file, dict(fid=fid, checkpoint_sha256=digest, num_images=args.num_images))
                comparisons[mode] = dict(fid=fid, internal_steps=steps, **timed)
            del model
            with REFERENCE.open() as file:
                native_fid = {int(r['epoch']): float(r['baseline']) for r in csv.DictReader(file)}
            if epoch in native_fid:
                native_file = args.native_checkpoint_dir / f'checkpoint-{epoch:04d}.pth'
                native = ema_model(native_file, epoch, 'swiglu', device, args)
                comparisons['swiglu'] = dict(fid=native_fid[epoch], checkpoint=str(native_file),
                                              checkpoint_sha256=file_hash(native_file),
                                              **speed(native, args.gen_bsz, args.seed, device))
            else:
                comparisons['swiglu'] = None
            for name, row in comparisons.items():
                if row is not None:
                    writer.add_scalar(f'sampling_images_per_second/{name}', row['images_per_second'], epoch)
            result = dict(epoch=epoch, checkpoint=str(checkpoint), checkpoint_sha256=digest, seed=args.seed,
                          ema=args.ema_decay1, num_images=args.num_images, outer_steps=args.num_sampling_steps,
                          sampler=args.sampling_method, cfg=args.cfg, results=comparisons,
                          gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu')
            if device.type == 'cuda':
                result['gpu_uuid'] = str(torch.cuda.get_device_properties(device).uuid)
            atomic_json(receipt, result)
            writer.flush()
            print('PAIRED_MILESTONE', json.dumps(result), flush=True)
            return result
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
