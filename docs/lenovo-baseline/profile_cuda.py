"""Time the frozen upstream epoch loop; discard all diagnostic updates."""
import argparse
import functools
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')


class Window:
    """Take a bounded number of batches from one already-started loader."""
    def __init__(self, iterator, length):
        self.iterator, self.length = iterator, length

    def __len__(self):
        return self.length

    def __iter__(self):
        return itertools.islice(self.iterator, self.length)


def compare_adam_backends(optimizer):
    """Compare independent real-gradient updates, one parameter pair at a time."""
    import torch
    total_error, total_update, maximum = 0.0, 0.0, 0.0
    moment_error = {'exp_avg': [0.0, 0.0], 'exp_avg_sq': [0.0, 0.0]}
    count = 0
    for group in optimizer.param_groups:
        kw = {k: group[k] for k in ('lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'maximize')}
        for original in group['params']:
            if original.grad is None:
                continue
            ref = torch.nn.Parameter(original.detach().clone())
            fused = torch.nn.Parameter(original.detach().clone())
            ref.grad, fused.grad = original.grad.clone(), original.grad.clone()
            native_opt = torch.optim.AdamW([ref], **kw)
            fused_opt = torch.optim.AdamW([fused], **kw, fused=True, foreach=False)
            for opt, param, device in ((native_opt, ref, 'cpu'), (fused_opt, fused, original.device)):
                opt.state[param] = {k: (v.detach().clone() if torch.is_tensor(v) else v)
                                    for k, v in optimizer.state[original].items()}
                opt.state[param]['step'] = opt.state[param]['step'].to(device)
            native_opt.step()
            fused_opt.step()
            delta = ref.detach() - fused.detach()
            total_error += delta.double().square().sum().item()
            total_update += (ref.detach() - original.detach()).double().square().sum().item()
            maximum = max(maximum, delta.abs().max().item())
            for key, sums in moment_error.items():
                left, right = native_opt.state[ref][key], fused_opt.state[fused][key]
                sums[0] += (left-right).double().square().sum().item()
                sums[1] += left.double().square().sum().item()
            assert native_opt.state[ref]['step'].item() == fused_opt.state[fused]['step'].item()
            count += 1
    relative_update_error = (total_error / max(total_update, 1e-30))**.5
    moment_relative = {k: (x/max(y, 1e-30))**.5 for k, (x, y) in moment_error.items()}
    passed = (maximum <= 1e-6 and relative_update_error <= .005
              and all(x <= 1e-5 for x in moment_relative.values()))
    return dict(parameter_tensors=count, maximum_parameter_error=maximum,
        relative_update_error=relative_update_error, moment_relative_errors=moment_relative,
        tolerances={'parameter_max_abs': 1e-6, 'relative_update': .005, 'relative_moments': 1e-5},
        passed=passed, bitwise_equivalence_claim=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--instrument', action='store_true')
    parser.add_argument('--operator-profile', action='store_true')
    parser.add_argument('--adam-fused', action='store_true')
    parser.add_argument('--validate-adam', action='store_true')
    a = parser.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    sys.path.insert(0, str(a.source.resolve()))
    import torch
    import torchvision
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from denoiser import Denoiser
    from engine_jit import train_one_epoch
    from util.crop import center_crop_arr
    from util import misc

    torch.set_num_threads(4)
    torch._dynamo.config.disable = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)
    cp = torch.load(a.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    args = cp['args']
    args.batch_size = a.batch_size
    epoch = max(6, args.warmup_epochs)  # All timings use the registered plateau LR.
    model = Denoiser(args).cuda()
    model.load_state_dict(cp['model'])
    model.ema_params1 = [cp['model_ema1'][n].cuda() for n, _ in model.named_parameters()]
    model.ema_params2 = [cp['model_ema2'][n].cuda() for n, _ in model.named_parameters()]
    optimizer = torch.optim.AdamW(misc.add_weight_decay(model, args.weight_decay),
                                  lr=args.lr, betas=(.9, .95),
                                  fused=True if a.adam_fused else None)
    if a.adam_fused:
        # Set before load_state_dict so PyTorch places step counters on CUDA.
        for group in cp['optimizer']['param_groups']:
            group['fused'], group['foreach'] = True, False
    optimizer.load_state_dict(cp['optimizer'])
    checkpoint_epoch = cp['epoch']
    del cp
    gc.collect()
    transform = transforms.Compose([
        transforms.Lambda(functools.partial(center_crop_arr, image_size=args.img_size)),
        transforms.RandomHorizontalFlip(), transforms.PILToTensor()])
    data = torchvision.datasets.CIFAR10(a.data_path, train=True, transform=transform)
    sampler = torch.utils.data.DistributedSampler(data, num_replicas=1, rank=0,
                                                 shuffle=True, seed=0)
    sampler.set_epoch(checkpoint_epoch + 1)
    loader = DataLoader(data, sampler=sampler, batch_size=a.batch_size,
                        num_workers=2, pin_memory=True, drop_last=True)
    torch.manual_seed(0)
    iterator = iter(loader)
    train_one_epoch(model, model, Window(iterator, a.warmup), optimizer,
                    torch.device('cuda'), epoch, args=args)
    torch.cuda.synchronize()
    assert a.warmup >= 20 and a.steps >= 100
    torch.cuda.reset_peak_memory_stats()
    events, losses = {}, []
    originals = []

    def wrap(owner, name, label):
        original = getattr(owner, name)
        originals.append((owner, name, original))
        events[label] = []

        def timed(*pos, **kw):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            began = time.perf_counter()
            value = original(*pos, **kw)
            host_ms = (time.perf_counter() - began) * 1000
            end.record()
            events[label].append((start, end, host_ms))
            return value

        setattr(owner, name, timed)

    # A detached scalar per update is retained on GPU, with no extra sync.
    def save_loss(module, inputs, output):
        losses.append(output.detach())

    hook = model.register_forward_hook(save_loss)
    if a.instrument:
        wrap(model, 'forward', 'forward_loss')
        wrap(torch.Tensor, 'backward', 'backward')
        wrap(optimizer, 'step', 'adam')
        wrap(model, 'update_ema', 'ema')
    began = time.perf_counter()
    try:
        train_one_epoch(model, model, Window(iterator, a.steps), optimizer,
                        torch.device('cuda'), epoch, args=args)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - began
    finally:
        hook.remove()
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)
    assert len(losses) == a.steps
    values = torch.stack(losses).float().cpu()
    assert torch.isfinite(values).all()
    phases = {}
    for name, rows in events.items():
        assert len(rows) == a.steps, (name, len(rows))
        cuda_ms = [s.elapsed_time(e) for s, e, _ in rows]
        phases[name] = dict(cuda_stream_ms_mean=statistics.mean(cuda_ms),
                            cuda_stream_ms_median=statistics.median(cuda_ms),
                            host_call_ms_mean=statistics.mean(x[2] for x in rows))
    peak = torch.cuda.max_memory_allocated() / 2**30
    reserved = torch.cuda.max_memory_reserved() / 2**30
    # Equality witness for plain versus instrumented runs with the same batch.
    state_hash = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        state_hash.update(name.encode())
        state_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    receipt = dict(batch_size=a.batch_size, warmup_steps=a.warmup, measured_steps=a.steps,
        phase_instrumentation=a.instrument, adam_fused=a.adam_fused, checkpoint=str(a.checkpoint),
        checkpoint_completed_epochs=checkpoint_epoch + 1, benchmark_lr_epoch=epoch,
        torch=torch.__version__, torchvision=torchvision.__version__, gpu=torch.cuda.get_device_name(),
        elapsed_seconds=elapsed, seconds_per_update=elapsed/a.steps,
        images_per_second=a.steps*a.batch_size/elapsed, peak_allocated_GiB=peak,
        peak_reserved_GiB=reserved, phase_times=phases,
        loss_mean=values.mean().item(), loss_last=values[-1].item(),
        final_model_sha256=state_hash.hexdigest(),
        rng_sha256=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
        quality_claim=False, checkpoint_written=False)
    receipt['gpu_snapshot'] = subprocess.run(['nvidia-smi',
        '--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm',
        '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.strip()
    if a.validate_adam:
        receipt['adam_update_comparison'] = compare_adam_backends(optimizer)
    if a.operator_profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            train_one_epoch(model, model, Window(iterator, 3), optimizer,
                            torch.device('cuda'), epoch, args=args)
            torch.cuda.synchronize()
        ops = sorted(prof.key_averages(), key=lambda e: e.self_device_time_total, reverse=True)
        receipt['operators'] = [dict(name=e.key, calls=e.count,
            self_device_ms=e.self_device_time_total/1000,
            self_cpu_ms=e.self_cpu_time_total/1000) for e in ops[:20]]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)
    print('BOUNDED_UPSTREAM_PROFILE_PASS', flush=True)


if __name__ == '__main__':
    main()
