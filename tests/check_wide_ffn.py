"""Check the widened operator and profile the actual JiT training/sampling path."""
import argparse
import copy
import gc
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ffn import ISRU_D, ISRU_DLayer
from denoiser import Denoiser
from main_jit import get_args_parser
from util import misc
from util.crop import center_crop_arr
from util.datasets import celeba_train
from torchvision import transforms
from functools import partial


def options(order, width=1200):
    return get_args_parser().parse_args([
        '--model', 'JiT-B/16', '--img_size', '128', '--class_num', '1',
        '--label_drop_prob', '1', '--cfg', '0', '--noise_scale', '0.5',
        '--ffn', 'isru_d', '--ode_order', str(order), '--ode_hidden_dim', str(width),
    ])


def fingerprint(items):
    digest = hashlib.sha256()
    for name, tensor in items:
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def algebra():
    torch.manual_seed(0)
    for order in (1, 8):
        block = ISRU_D(4, order, 6).double()
        x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)

        def dense(value, layer):
            d = torch.rsqrt(1 + 1.2 * value.square())
            a = layer.proj.weight * d.unsqueeze(-2)
            polynomial = torch.eye(6, dtype=value.dtype).expand_as(a).clone()
            for k in range(1, order + 1):
                polynomial = polynomial + torch.linalg.matrix_power(a, k) / math.factorial(k)
            return (polynomial @ value.unsqueeze(-1)).squeeze(-1)

        u = F.linear(x, block.input_proj.weight)
        reference = F.linear(dense(F.silu(dense(u, block.ode1)), block.ode2), block.output_proj.weight)
        actual = block(x)
        torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
        parameters = (x, *block.parameters())
        direction = torch.randn_like(actual)
        a = torch.autograd.grad(actual, parameters, direction, retain_graph=True)
        b = torch.autograd.grad(reference, parameters, direction)
        for left, right in zip(a, b):
            torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-12)
        assert torch.autograd.gradcheck(block, (x,), fast_mode=True)
    # Original-width state keys, RNG consumption and numerical action survive.
    torch.manual_seed(0)
    old1, old2 = ISRU_DLayer(4, 8), ISRU_DLayer(4, 8)
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(0)
    unchanged = ISRU_D(4, 8)
    assert torch.equal(old_rng, torch.get_rng_state())
    assert set(unchanged.state_dict()) == {'ode1.proj.weight', 'ode2.proj.weight'}
    x = torch.randn(3, 4)
    torch.testing.assert_close(unchanged(x), old2(F.silu(old1(x))), rtol=0, atol=0)
    return {'dense_polynomial_and_all_derivatives': 'PASS', 'gradcheck': 'PASS',
            'original_width_state_rng_and_output': 'PASS'}


def cpu():
    torch._dynamo.config.disable = True
    result = {'algebra': algebra()}
    baseline = options(8, 0)
    baseline.ffn = 'swiglu'
    records = {}
    for name, args in [('native', baseline), ('k1', options(1)), ('k8', options(8))]:
        torch.manual_seed(0)
        model = Denoiser(args)
        records[name] = {
            'parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'backbone_sha256': fingerprint((n, t) for n, t in model.state_dict().items()
                                           if not (n.startswith('net.blocks.') and '.mlp.' in n)),
            'ffn_sha256': fingerprint((n, t) for n, t in model.state_dict().items() if '.mlp.' in n),
            'post_constructor_rng_sha256': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        }
        del model
        gc.collect()
    assert records['native']['parameters'] == 130405632
    assert records['k1']['parameters'] == records['k8']['parameters'] == 130402560
    assert records['k1']['ffn_sha256'] == records['k8']['ffn_sha256']
    for key in ('backbone_sha256', 'post_constructor_rng_sha256'):
        assert len({r[key] for r in records.values()}) == 1
    result['models'] = records
    return result


def cuda(data_root, batch):
    assert torch.cuda.is_available()
    dataset = celeba_train(data_root, transforms.Compose([
        transforms.Lambda(partial(center_crop_arr, image_size=128)), transforms.PILToTensor()]))
    pixels = torch.stack([dataset[i][0] for i in range(batch)])
    records = {}
    for order in (1, 8):
        torch.manual_seed(0)
        model = Denoiser(options(order)).cuda()
        model.ema_params1 = copy.deepcopy(list(model.parameters()))
        model.ema_params2 = copy.deepcopy(list(model.parameters()))
        optimizer = torch.optim.AdamW(misc.add_weight_decay(model, 0), lr=5e-5, betas=(.9, .95))
        images = pixels.cuda().float().div_(255).mul_(2).sub_(1)
        labels = torch.zeros(batch, device='cuda', dtype=torch.long)
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(5):
            torch.cuda.synchronize()
            start = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(images, labels)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            model.update_ema()
            torch.cuda.synchronize()
            times.append(time.monotonic() - start)
        for p in model.parameters():
            assert torch.isfinite(p).all()
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()
        for layer in (model.net.blocks[0].mlp.input_proj, model.net.blocks[0].mlp.ode1.proj,
                      model.net.blocks[0].mlp.ode2.proj, model.net.blocks[0].mlp.output_proj):
            assert layer.weight.grad.abs().sum() > 0
        peak_train = torch.cuda.max_memory_allocated()
        model.eval()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            # Full native Heun50, including upstream unconditional/conditional calls.
            generation = []
            for _ in range(2):
                torch.cuda.synchronize()
                start = time.monotonic()
                samples = model.generate(labels)
                torch.cuda.synchronize()
                generation.append(time.monotonic() - start)
                assert torch.isfinite(samples).all()
        records[f'k{order}'] = dict(batch=batch, parameters=sum(p.numel() for p in model.parameters()),
            loss=loss.item(), step_seconds=times, warm_step_seconds=sum(times[2:]) / 3,
            generation_batch_seconds=generation, warm_generation_batch_seconds=generation[-1],
            peak_training_bytes=peak_train, peak_including_generation_bytes=torch.cuda.max_memory_allocated(),
            finite_weights_gradients_and_samples=True, projections_and_P_receive_gradients=True)
        del optimizer, model, images, labels, loss, samples
        gc.collect()
        torch.cuda.empty_cache()
    return {'arms': records, 'cuda': {'name': torch.cuda.get_device_name(),
             'total_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
             'torch': torch.__version__, 'cuda': torch.version.cuda}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--data-root', default='/scratch/zz5070/ODE_FFN/datasets')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    record = (cpu() if args.device == 'cpu' else cuda(args.data_root, args.batch_size))
    record.update(status='PASS', image_size=128, hidden_dim=1200, formal_training=False,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + '\n')
    print('WIDE_FFN_' + args.device.upper() + '_PASS', flush=True)


if __name__ == '__main__':
    main()
