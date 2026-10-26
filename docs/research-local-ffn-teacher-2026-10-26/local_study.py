"""Fixed single-FFN teacher fitting with immutable inputs and monitored workers."""
import argparse
import copy
import hashlib
import os
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import psutil
import torch
from teacher_support import (BASE, OLD, BANKS, DATASET, TEACHERS, STUDENTS,
                             TIMES, SNAPSHOTS, load, write, append, sha,
                             setup, student, teacher, verify_cold, signature,
                             assert_equal, replay_hash)

PREVIOUS = BASE / 'realizable_teacher_20261026'
EMPTY = hashlib.sha256(b'').hexdigest()


def cold(family: str, position: int) -> torch.nn.Module:
    full = student(family)
    verify_cold(full, family)
    return copy.deepcopy(full.raw.blocks[position].ffn)


def capture(full: torch.nn.Module, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    values = [None] * 3
    handles = []
    def hook(index: int):
        def save(module, args):
            assert values[index] is None
            values[index] = args[0].detach().clone()
        return save
    try:
        for j, block in enumerate(full.raw.blocks):
            handles.append(block.ffn.register_forward_pre_hook(hook(j)))
        with torch.no_grad():
            output = full(x, t)
        assert all(z is not None for z in values)
        return output, values
    finally:
        for handle in handles:
            handle.remove()


def predict(module: torch.nn.Module, z: np.ndarray) -> np.ndarray:
    result = []
    module.eval()
    with torch.no_grad():
        for start in range(0, len(z), 1024):
            result.append(module(torch.from_numpy(np.array(z[start:start+1024], copy=True))).numpy())
    return np.concatenate(result)


def native_jacobian(module: torch.nn.Module, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    module.eval()
    value = module(z)
    derivative = torch.func.vmap(torch.func.jacrev(lambda row: module(row[None])[0]))(z)
    return value.detach(), derivative.detach()


def file_name(split: str, position: int, family: str | None = None) -> str:
    return f'{split}-{family+"-y" if family else "z"}{position}.npy'


def arrays(cache: Path, split: str, position: int, family: str) -> tuple[np.ndarray, np.ndarray]:
    return tuple(np.load(cache/file_name(split, position, f), mmap_mode='r') for f in (None, family))


def error(prediction: np.ndarray, truth: np.ndarray) -> dict:
    truth = np.asarray(truth, dtype=np.float64)
    mse = float(np.mean((np.asarray(prediction, dtype=np.float64)-truth)**2))
    energy = float(np.mean(truth**2))
    assert energy > 0
    return dict(mse=mse, relative_mse=mse/energy, target_energy=energy)


def prepare(cache: Path, *, train_count: int = 100000, validation_count: int = 20000,
            panel_count: int = 512) -> dict:
    cache.mkdir(exist_ok=False)
    models = {family: teacher(family)[0] for family in TEACHERS}
    initial = {family: signature(model) for family, model in models.items()}
    for family, model in models.items():
        for j in range(3):
            exact = cold(family, j)
            exact.load_state_dict(model.raw.blocks[j].ffn.state_dict())
            assert_equal(exact.state_dict(), model.raw.blocks[j].ffn.state_dict())
    previous_cache = PREVIOUS/'output/cache'
    meta = load(previous_cache/'metadata.json')
    for name in ('x.npy', 't.npy'):
        assert sha(previous_cache/name) == meta['files_sha256'][name]
    train_x = np.load(previous_cache/'x.npy', mmap_mode='r').reshape(-1, 16)[:train_count]
    train_t = np.load(previous_cache/'t.npy', mmap_mode='r').reshape(-1, 1)[:train_count]
    splits = {'train': (train_x, train_t)}
    for split, filename, count in [('validation', 'field.npz', validation_count)] + [
            (f'panel-{t:.1f}', f'panel_{t:.1f}.npz', panel_count) for t in TIMES]:
        with np.load(BANKS/filename) as data:
            splits[split] = (data['x'][:count], data['t'][:count])
    replay = {}
    for split, (x, t) in splits.items():
        destinations = [np.lib.format.open_memmap(cache/file_name(split, j), mode='w+',
                        dtype=np.float32, shape=(len(x), 64)) for j in range(3)]
        outputs = []
        for start in range(0, len(x), 1024):
            value, features = capture(models['swiglu128'], torch.from_numpy(np.array(x[start:start+1024], copy=True)),
                                      torch.from_numpy(np.array(t[start:start+1024], copy=True)))
            outputs.append(value.numpy())
            for j in range(3):
                destinations[j][start:start+len(value)] = features[j].numpy()
        for z in destinations:
            z.flush()
        del destinations
        if split != 'train':
            filename = 'field.npz' if split == 'validation' else f'panel_{split[-3:]}.npz'
            with np.load(OLD/'runs'/DATASET/'swiglu128_direct/diagnostics'/filename) as saved:
                reference = saved['prediction'][:len(x)]
            actual = np.concatenate(outputs)
            difference = float(np.max(np.abs(actual-reference)))
            if split == 'validation' or panel_count == 512:
                assert np.array_equal(actual, reference), (split, difference)
            else:
                assert difference <= 2e-6 * max(1., float(np.max(np.abs(reference))))
            replay[split] = difference
        for j in range(3):
            z = np.load(cache/file_name(split, j), mmap_mode='r')
            for family, full in models.items():
                local = full.raw.blocks[j].ffn
                value = predict(local, z)
                np.save(cache/file_name(split, j, family), value)
                if split == 'validation':
                    _, jac = native_jacobian(local, torch.from_numpy(np.array(z[:128], copy=True)))
                    np.save(cache/f'jac-{family}-{j}.npy', jac.numpy())
    assert initial == {family: signature(model) for family, model in models.items()}
    result = dict(train_count=train_count, validation_count=validation_count,
                  panel_count=panel_count, teachers=initial, native_full_teacher_replay=replay,
                  files_sha256={p.name: sha(p) for p in sorted(cache.glob('*.npy'))})
    write(cache/'metadata.json', result)
    return result


def optimizer(module: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(module.parameters(), lr=3e-4, weight_decay=0.)


def train_step(module: torch.nn.Module, opt: torch.optim.Optimizer,
               z: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    module.train()
    opt.zero_grad(set_to_none=True)
    loss = (module(z)-y).square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(module.parameters(), 1.)
    assert torch.isfinite(grad)
    opt.step()
    return float(loss.detach()), float(grad)


def consumed(z: np.ndarray, y: np.ndarray, rng: torch.Generator, digests: dict) -> tuple[torch.Tensor, torch.Tensor]:
    index = torch.randint(len(z), (512,), generator=rng)
    xx = torch.from_numpy(np.array(z[index.numpy()], copy=True))
    yy = torch.from_numpy(np.array(y[index.numpy()], copy=True))
    digests['indices'] = replay_hash(digests['indices'], index)
    digests['input'] = replay_hash(digests['input'], xx)
    digests['label'] = replay_hash(digests['label'], yy)
    return xx, yy


def checkpoint(module: torch.nn.Module, opt: torch.optim.Optimizer,
               rng: torch.Generator, step: int, digests: dict) -> dict:
    return dict(model=copy.deepcopy(module.state_dict()), optimizer=copy.deepcopy(opt.state_dict()),
                runtime_buffers={k: v.detach().clone() for k, v in module.named_buffers()},
                torch_rng=torch.get_rng_state(), sampling_rng=rng.get_state(), step=step,
                consumed_digests=digests.copy())


def restore(module: torch.nn.Module, opt: torch.optim.Optimizer, rng: torch.Generator, state: dict) -> dict:
    module.load_state_dict(state['model'])
    assert set(dict(module.named_buffers())) == set(state['runtime_buffers'])
    for name, value in module.named_buffers():
        value.copy_(state['runtime_buffers'][name])
    opt.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng'])
    rng.set_state(state['sampling_rng'])
    return state['consumed_digests'].copy()


def spectrum(module: torch.nn.Module) -> dict:
    from synthetic.models import OriginalODELayer
    result = {}
    for name, layer in module.named_modules():
        if isinstance(layer, OriginalODELayer):
            p = layer.proj.weight.detach().double()
            result[name] = dict(singular_values=torch.linalg.svdvals(p).tolist(),
                               spectral_radius=float(torch.linalg.eigvals(p).abs().max()),
                               symmetric_minimum=float(torch.linalg.eigvalsh((p+p.T)/2).min()))
    return result


def budget(module: torch.nn.Module) -> dict:
    from synthetic.conservative_readout import MatrixCounter
    x = torch.linspace(-.4, .7, 64).reshape(1, 64)
    with MatrixCounter() as inference:
        with torch.no_grad():
            module(x)
    with MatrixCounter() as training:
        prediction = module(x)
        torch.autograd.grad(prediction.square().mean(), tuple(module.parameters()))
    return dict(parameters=sum(p.numel() for p in module.parameters()), forward_macs=inference.macs,
                forward_backward_macs=training.macs, matrix_scope='actual dense matrix MACs; pointwise work excluded')


def validate(module: torch.nn.Module, cache: Path, family: str, position: int, path: Path | None = None) -> dict:
    z, y = arrays(cache, 'validation', position, family)
    value = predict(module, z)
    if path is not None:
        np.save(path, value)
    return error(value, y)


def preflight(root: Path) -> None:
    torch.set_num_threads(1)
    from local_math import value_jacobian
    cache = root/'preflight-cache'
    prepare(cache, train_count=2048, validation_count=2048, panel_count=128)
    checks = []
    for j in range(3):
        z = np.load(cache/file_name('validation', j), mmap_mode='r')
        x = torch.from_numpy(np.array(z[:13], copy=True))
        for family in STUDENTS:
            for kind in ('cold', 'teacher') if family in TEACHERS else ('cold',):
                module = cold(family, j) if kind == 'cold' else teacher(family)[0].raw.blocks[j].ffn
                module = copy.deepcopy(module).double()
                actual, jac = native_jacobian(module, x.double())
                explicit, analytic = value_jacobian(module, x.double(), family)
                torch.testing.assert_close(actual, explicit, rtol=1e-10, atol=1e-11)
                torch.testing.assert_close(jac, analytic, rtol=1e-10, atol=1e-11)
                checks.append(dict(position=j, family=family, kind=kind,
                                   forward_max_abs=float((actual-explicit).abs().max()),
                                   jacobian_max_abs=float((jac-analytic).abs().max())))
        for teacher_name in TEACHERS:
            z_train, y_train = arrays(cache, 'train', j, teacher_name)
            target = teacher(teacher_name)[0].raw.blocks[j].ffn
            exact = cold(teacher_name, j)
            exact.load_state_dict(target.state_dict())
            z_val, y_val = arrays(cache, 'validation', j, teacher_name)
            assert np.array_equal(predict(exact, z_val), y_val)
            index = torch.randint(len(z_train), (512,), generator=torch.Generator().manual_seed(0))
            with torch.no_grad():
                live = target(torch.from_numpy(np.array(z_train[index.numpy()], copy=True))).numpy()
            cached = np.array(y_train[index.numpy()], copy=True)
            assert np.max(np.abs(live-cached)) <= 2e-6 * max(1., float(np.max(np.abs(cached))))
            for family in STUDENTS:
                states = []
                for resume in (False, True):
                    module = cold(family, j)
                    initial = signature(module)
                    opt = optimizer(module)
                    rng = torch.Generator().manual_seed(0)
                    digests = dict.fromkeys(('indices', 'input', 'label'), EMPTY)
                    for step in range(1, 4):
                        xx, yy = consumed(z_train, y_train, rng, digests)
                        train_step(module, opt, xx, yy)
                        if resume and step == 2:
                            state = checkpoint(module, opt, rng, step, digests)
                            module = cold(family, j)
                            opt = optimizer(module)
                            digests = restore(module, opt, rng, state)
                    assert signature(module) != initial
                    states.append(checkpoint(module, opt, rng, 3, digests))
                assert_equal(*states)
    costs = {family: budget(cold(family, 0)) for family in STUDENTS}
    write(root/'preflight.json', dict(status='PASS', analytic_checks=checks,
          exact_representers=6, cached_live_checks=6, exact_resume_cells=18, costs=costs))


def cell(root: Path, teacher_name: str, family: str, position: int) -> None:
    torch.set_num_threads(1)
    source = Path(__file__).parent
    for name, digest in load(source/'source-sha256.json').items():
        assert sha(source/name) == digest
    cache = root/'output/cache'
    folder = root/'output/runs'/f'{teacher_name}--{family}--{position}'
    folder.mkdir(exist_ok=False)
    started = time.monotonic()
    def status(state: str = 'RUNNING', step: int = 0, **extra):
        write(folder/'status.json', dict(state=state, step=step, pid=os.getpid(),
              elapsed_seconds=time.monotonic()-started, **extra))
    status()
    try:
        module = cold(family, position)
        opt = optimizer(module)
        rng = torch.Generator().manual_seed(0)
        initial = signature(module)
        digests = dict.fromkeys(('indices', 'input', 'label'), EMPTY)
        z, y = arrays(cache, 'train', position, teacher_name)
        write(folder/'provenance.json', dict(teacher=teacher_name, student=family, position=position,
              initial_signature=initial, seed=0, threads=torch.get_num_threads(), device='cpu',
              source_manifest_sha256=sha(source/'source-sha256.json'),
              cache_metadata_sha256=sha(cache/'metadata.json'), costs=budget(module)))
        def save(step: int):
            torch.save(checkpoint(module, opt, rng, step, digests), folder/f'step_{step:07d}.pt')
            write(folder/f'spectrum_{step:07d}.json', spectrum(module))
            return validate(module, cache, teacher_name, position, folder/f'validation_{step:07d}.npy')
        append(folder/'metrics.jsonl', dict(step=0, **save(0)))
        loss_sum = grad_sum = 0.
        count = clipped = 0
        for step in range(1, 5001):
            xx, yy = consumed(z, y, rng, digests)
            loss, grad = train_step(module, opt, xx, yy)
            loss_sum += loss
            grad_sum += grad
            count += 1
            clipped += int(grad > 1)
            if step % 100 == 0:
                status(step=step)
            if step % 500 == 0:
                record = dict(step=step, training_loss=loss_sum/count,
                              mean_gradient_norm=grad_sum/count, clip_fraction=clipped/count)
                if step in SNAPSHOTS:
                    record.update(save(step))
                append(folder/'metrics.jsonl', record)
                print(record, flush=True)
                loss_sum = grad_sum = 0.
                count = clipped = 0
        panels = []
        for t in TIMES:
            zz, yy = arrays(cache, f'panel-{t:.1f}', position, teacher_name)
            value = predict(module, zz)
            np.save(folder/f'panel_{t:.1f}.npy', value)
            panels.append(dict(time=t, **error(value, yy)))
        zv, _ = arrays(cache, 'validation', position, teacher_name)
        _, jac = native_jacobian(module, torch.from_numpy(np.array(zv[:128], copy=True)))
        np.save(folder/'jacobian.npy', jac.numpy())
        truth_j = np.load(cache/f'jac-{teacher_name}-{position}.npy').astype(np.float64)
        je = float(np.mean(np.sum((jac.numpy().astype(np.float64)-truth_j)**2, axis=(1, 2))))
        result = dict(teacher=teacher_name, student=family, position=position, step=5000,
                      **validate(module, cache, teacher_name, position), panels=panels,
                      jacobian_error=je, relative_jacobian_error=je/float(np.mean(np.sum(truth_j**2, axis=(1, 2)))),
                      consumed_digests=digests, final_signature=signature(module),
                      elapsed_seconds=time.monotonic()-started)
        assert result['final_signature'] != initial
        write(folder/'result.json', result)
        status(state='COMPLETE', step=5000)
    except BaseException:
        status(state='FAILED', error=traceback.format_exc())
        raise


def observe() -> dict:
    parent = psutil.Process()
    rows = []
    for process in [parent]+parent.children(recursive=True):
        try:
            rows.append(dict(pid=process.pid, rss_bytes=process.memory_info().rss,
                             cpu_seconds=sum(process.cpu_times()[:2]), command=process.cmdline()))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                available_memory_bytes=psutil.virtual_memory().available, processes=rows)


def run(root: Path) -> None:
    torch.set_num_threads(1)
    source = Path(__file__).parent
    ready = load(root/'ready.json')
    assert load(root/'preflight.json')['status'] == 'PASS'
    for path, digest in ready['bindings'].items():
        assert sha(path) == digest, path
    out = root/'output'
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    stop, abort, lock = threading.Event(), threading.Event(), threading.Lock()
    processes = set()
    progress = dict(state='RUNNING', phase='prepare', completed=0, total=18, pid=os.getpid())
    def status(**extra):
        progress.update(extra, elapsed_seconds=time.monotonic()-started)
        write(out/'status.json', progress)
        print(progress, flush=True)
    def monitor():
        while not stop.is_set():
            row = observe()
            append(out/'monitor.jsonl', row)
            write(out/'monitor.json', row)
            stop.wait(5)
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    status()
    try:
        prepare(out/'cache')
        (out/'runs').mkdir()
        status(phase='training')
        env = os.environ.copy()
        env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   VECLIB_MAXIMUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
        def launch(job):
            teacher_name, family, position = job
            name = f'{teacher_name}--{family}--{position}'
            command = [sys.executable, str(source/'local_study.py'), 'cell', '--root', str(root),
                       '--teacher', teacher_name, '--student', family, '--position', str(position)]
            with (out/(name+'.log')).open('w') as log:
                with lock:
                    if abort.is_set():
                        raise RuntimeError('study aborted')
                    process = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
                    processes.add(process)
                    append(out/'launches.jsonl', dict(name=name, pid=process.pid, command=command))
                code = process.wait()
                with lock:
                    processes.remove(process)
            if code:
                raise RuntimeError(f'{name} exited {code}')
            return name
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(launch, (t, s, j)) for t in TEACHERS for j in range(3) for s in STUDENTS]
            try:
                for future in as_completed(futures):
                    status(completed=progress['completed']+1, latest=future.result())
            except BaseException:
                abort.set()
                for future in futures:
                    future.cancel()
                with lock:
                    for process in processes:
                        process.terminate()
                raise
        for path, digest in ready['bindings'].items():
            assert sha(path) == digest, path
        status(state='COMPLETE', phase='complete')
    except BaseException:
        status(state='FAILED', error=traceback.format_exc())
        raise
    finally:
        stop.set()
        watcher.join(timeout=6)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['preflight', 'run', 'cell'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--teacher', choices=TEACHERS)
    parser.add_argument('--student', choices=STUDENTS)
    parser.add_argument('--position', type=int, choices=range(3))
    args = parser.parse_args()
    {'preflight': preflight, 'run': run}.get(args.action, lambda r: cell(r, args.teacher, args.student, args.position))(args.root)
