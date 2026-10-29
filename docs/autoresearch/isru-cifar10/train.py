"""Paired, resumable unconditional CIFAR training on one MPS device."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
import hashlib
import json
import shutil
import signal
import subprocess
import time
from pathlib import Path
import numpy as np
import psutil
import torch
from PIL import Image
from common import append, cpu_tree, load, save_torch, sha, uint8_images, verify, write
from model import ARMS, build_all, costs


STOP = False


def request_stop(*_):
    global STOP
    STOP = True


def make_models(root, config):
    models = build_all(root/'source/native', config['width'], config['depth'])
    for model in models.values():
        model.to('mps')
    optimizers = {arm: torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                  betas=tuple(config['betas']), weight_decay=config['weight_decay'], foreach=True)
                  for arm, model in models.items()}
    emas = {arm: copy.deepcopy(model).requires_grad_(False).eval() for arm, model in models.items()}
    return models, optimizers, emas


def batch(images, indices, rng, count, config, flip=True):
    ix = torch.randint(len(indices), (count,), generator=rng)
    clean = torch.from_numpy(np.array(images[indices[ix.numpy()]], copy=True)).float()/127.5-1
    mask = torch.rand(count, generator=rng) < .5
    if flip and config['horizontal_flip']:
        clean = torch.where(mask[:, None, None, None], clean.flip(-1), clean)
    noise = torch.randn(clean.shape, generator=rng)
    t = torch.sigmoid(config['time_logit_std']*torch.randn(count, generator=rng)+config['time_logit_mean'])
    digest = hashlib.sha256()
    for value in (ix, mask, noise, t):
        digest.update(value.numpy().tobytes())
    shape = t[:, None, None, None]
    z = shape*clean+(1-shape)*noise
    return clean.to('mps'), z.to('mps'), t.to('mps'), digest.hexdigest()


def objective(model, clean, z, t, config):
    pred = model(z, t, torch.zeros(len(t), device=z.device, dtype=torch.long))
    loss = ((pred-clean)/(1-t[:, None, None, None]).clamp_min(config['denominator_floor'])).square().mean()
    return loss


@torch.no_grad()
def update_ema(ema, model, decay):
    torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1-decay)
    for dest, src in zip(ema.buffers(), model.buffers()):
        dest.copy_(src)


def update(model, optimizer, ema, data, step, config):
    lr = config['learning_rate']*min(step/config['warmup_steps'], 1.)
    for group in optimizer.param_groups:
        group['lr'] = lr
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = objective(model, *data[:3], config)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], foreach=False,
                                         error_if_nonfinite=True)
    optimizer.step()
    update_ema(ema, model, config['ema_decay'])
    return float(loss.detach()), float(norm), lr


def snapshot(models, optimizers, emas, rng, step, digest, root):
    return dict(step=step, stream_digest=digest, config_sha256=sha(root/'config.json'),
                bindings_sha256=sha(root/'bindings.json'), rng=rng.get_state(),
                torch_rng=torch.get_rng_state(), mps_rng=torch.mps.get_rng_state(),
                models={a: m.state_dict() for a, m in models.items()},
                optimizers={a: o.state_dict() for a, o in optimizers.items()},
                emas={a: e.state_dict() for a, e in emas.items()})


def restore(value, models, optimizers, emas, rng, root):
    assert value['config_sha256'] == sha(root/'config.json')
    assert value['bindings_sha256'] == sha(root/'bindings.json')
    for arm in ARMS:
        models[arm].load_state_dict(value['models'][arm])
        optimizers[arm].load_state_dict(value['optimizers'][arm])
        emas[arm].load_state_dict(value['emas'][arm])
    rng.set_state(value['rng'])
    torch.set_rng_state(value['torch_rng'])
    torch.mps.set_rng_state(value['mps_rng'])
    return value['step'], value['stream_digest']


@torch.no_grad()
def validation(models, emas, images, indices, config):
    rng = torch.Generator().manual_seed(config['validation_seed'])
    result = {a: {'raw': 0., 'ema': 0.} for a in ARMS}
    done = 0
    while done < config['validation_images']:
        n = min(config['validation_batch'], config['validation_images']-done)
        data = batch(images, indices, rng, n, config, flip=False)
        for arm in ARMS:
            for tag, model in (('raw', models[arm]), ('ema', emas[arm])):
                model.eval()
                result[arm][tag] += n*float(objective(model, *data[:3], config))
        done += n
    return {a: {tag: v/config['validation_images'] for tag, v in row.items()}
            for a, row in result.items()}


@torch.no_grad()
def sample(model, starts, config, steps=None):
    model.eval()
    count = steps or config['sampling_steps']
    z = starts.to('mps').clone()
    labels = torch.zeros(len(z), device=z.device, dtype=torch.long)
    grid = torch.linspace(0, 1, count+1, device=z.device)
    def velocity(x, scalar):
        t = scalar.expand(len(x))
        return (model(x, t, labels)-x)/(1-scalar).clamp_min(config['denominator_floor'])
    for i in range(count):
        t, tnext = grid[i], grid[i+1]
        dt = tnext-t
        first = velocity(z, t)
        proposal = z+dt*first
        z = proposal if i == count-1 else z+dt*.5*(first+velocity(proposal, tnext))
    assert torch.isfinite(z).all()
    return z.cpu().numpy()


def grid(path, raw):
    pixels = uint8_images(raw[:64]).transpose(0, 2, 3, 1)
    canvas = pixels.reshape(8, 8, 32, 32, 3).transpose(0, 2, 1, 3, 4).reshape(256, 256, 3)
    Image.fromarray(canvas).save(path)


def matrix_diagnostics(models):
    rows = []
    for arm, model in models.items():
        for i, block in enumerate(model.blocks):
            for name, value in block.mlp.named_parameters():
                if value.ndim != 2:
                    continue
                matrix = value.detach().cpu().double()
                s = torch.linalg.svdvals(matrix)
                prob = s/s.sum().clamp_min(1e-30)
                rows.append(dict(arm=arm, block=i, matrix=name, shape=list(matrix.shape),
                    singular_min=float(s.min()), singular_max=float(s.max()),
                    singular_mean=float(s.mean()), singular_std=float(s.std()),
                    frobenius=float(torch.linalg.norm(matrix)),
                    effective_rank=float(torch.exp(-(prob*prob.clamp_min(1e-30).log()).sum()))))
    return rows


def milestone(root, step, models, emas, config, paths, status):
    folder = root/'output'/f'eval-{step:06d}'
    folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists():
        return
    write(folder/'matrix-diagnostics.json', matrix_diagnostics(models))
    starts_rng = torch.Generator().manual_seed(config['sampling_seed'])
    count = config['fid_generated_final'] if step == config['steps'] else config['fid_generated_intermediate']
    use_fid = step in config['fid_milestones']
    count = count if use_fid else 64
    starts = torch.randn(count, 3, 32, 32, generator=starts_rng)
    for arm in ARMS:
        status('SAMPLING', step, arm=arm, generated=0, total=count)
        raw_grid = sample(models[arm], starts[:64], config)
        grid(folder/f'{arm}-raw-grid.png', raw_grid)
        generated = []
        for offset in range(0, count, config['sampling_batch']):
            generated.append(sample(emas[arm], starts[offset:offset+config['sampling_batch']], config))
            status('SAMPLING', step, arm=arm, generated=min(offset+config['sampling_batch'], count), total=count)
        generated = np.concatenate(generated)
        np.save(folder/f'{arm}-ema.npy', generated)
        grid(folder/f'{arm}-ema-grid.png', generated)
    if use_fid:
        status('FID', step)
        command = [paths['metrics_python'], '-B', str(root/'source/metrics.py'), 'evaluate',
                   '--root', str(root), '--folder', str(folder)]
        with (folder/'metrics.log').open('a') as handle:
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    write(folder/'complete.json', dict(step=step, state='COMPLETE', generated=count,
          FID_computed=use_fid, sample_model='fixed EMA .9999; raw grids descriptive only'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    root = args.root.resolve()
    config, paths = verify(root)
    assert load(root/'documented-pass.json')['status'] == 'PASS'
    assert torch.backends.mps.is_available()
    torch.set_num_threads(1)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)
    out = root/'output'
    progress = out/'status.json'
    started = time.monotonic()
    process = psutil.Process()
    def status(phase, step, **extra):
        value = dict(phase=phase, step=step, steps=config['steps'], pid=os.getpid(),
            timestamp=time.time(), elapsed_seconds=time.monotonic()-started,
            RSS_bytes=process.memory_info().rss,
            MPS_allocated_bytes=torch.mps.current_allocated_memory(),
            MPS_driver_bytes=torch.mps.driver_allocated_memory(), **extra)
        write(progress, value)
        return value
    if not args.resume:
        assert not progress.exists() and not (out/'latest.pt').exists(), 'use explicit --resume'
    models, optimizers, emas = make_models(root, config)
    write(out/'model-costs.json', {a: costs(m) for a, m in models.items()})
    images = np.load(Path(paths['data'])/'images.uint8.npy', mmap_mode='r')
    split = np.load(Path(paths['data'])/'split.npz')
    assert images.shape == (50000, 3, 32, 32) and images.dtype == np.uint8
    assert len(split['train']) == 45000 and len(split['validation']) == 5000
    assert not set(split['train']).intersection(split['validation'])
    rng = torch.Generator().manual_seed(config['stream_seed'])
    step, digest = 0, '0'*64
    if args.resume:
        value = torch.load(out/'latest.pt', map_location='cpu', weights_only=False)
        step, digest = restore(value, models, optimizers, emas, rng, root)
    else:
        save_torch(out/'cold.pt', snapshot(models, optimizers, emas, rng, step, digest, root))
        write(out/'cold-matrices.json', matrix_diagnostics(models))
    try:
        # Complete any interrupted evaluation before moving past its exact state.
        if step in config['milestones']:
            milestone(root, step, models, emas, config, paths, status)
        window = {a: [] for a in ARMS}
        tick = time.monotonic()
        while step < config['steps'] and not STOP:
            if shutil.disk_usage(root).free < config['minimum_free_gib']*1024**3:
                raise RuntimeError('free space below registered minimum')
            step += 1
            data = batch(images, split['train'], rng, config['batch'], config)
            digest = hashlib.sha256((digest+data[3]).encode()).hexdigest()
            for arm in ARMS:
                window[arm].append(update(models[arm], optimizers[arm], emas[arm], data, step, config))
            if step % 25 == 0 or step == 1:
                elapsed = time.monotonic()-tick
                row = dict(step=step, timestamp=time.time(), stream_digest=digest,
                           cohort_seconds=elapsed/len(window[ARMS[0]]),
                           images_per_second_per_arm=config['batch']*len(window[ARMS[0]])/elapsed,
                           arms={a: dict(loss=float(np.mean([x[0] for x in v])),
                             grad_norm=float(np.mean([x[1] for x in v])),
                             clipped_fraction=float(np.mean([x[1]>config['gradient_clip'] for x in v])),
                             lr=v[-1][2]) for a, v in window.items()})
                append(out/'train.jsonl', row)
                print(json.dumps(row), flush=True)
                status('TRAINING', step, cohort_seconds=row['cohort_seconds'])
                window = {a: [] for a in ARMS}
                tick = time.monotonic()
            elif step % 5 == 0:
                status('TRAINING', step, cohort_seconds=(time.monotonic()-tick)/len(window[ARMS[0]]))
            if step % config['checkpoint_interval'] == 0 or STOP:
                status('CHECKPOINT', step)
                save_torch(out/'latest.pt', snapshot(models, optimizers, emas, rng, step, digest, root))
                write(out/'latest-checkpoint.json', dict(step=step, sha256=sha(out/'latest.pt'), stream_digest=digest))
            if step % config['validation_interval'] == 0:
                status('VALIDATION', step)
                values = validation(models, emas, images, split['validation'], config)
                append(out/'validation.jsonl', dict(step=step, timestamp=time.time(), arms=values))
            if step in config['milestones']:
                shutil.copy2(out/'latest.pt', out/f'checkpoint-{step:06d}.pt')
                milestone(root, step, models, emas, config, paths, status)
            if step % config['validation_interval'] == 0:
                tick = time.monotonic()
        save_torch(out/'latest.pt', snapshot(models, optimizers, emas, rng, step, digest, root))
        status('STOPPED' if STOP else 'TRAINING_COMPLETE', step)
    except BaseException as exc:
        # Preserve a readable failure; do not represent a possibly partial cohort as resumable.
        status('FAILED', step, error=repr(exc), recovery='last successfully saved complete cohort')
        raise


if __name__ == '__main__':
    main()
