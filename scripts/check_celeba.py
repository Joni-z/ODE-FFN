"""Verify the selected polynomial, official data adapter and CUDA dispatch."""
import argparse
import copy
import gc
import json
import math
import os
import sys
import time
import tempfile
from pathlib import Path

os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '2')
import torch
from torchvision import transforms
from functools import partial

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ffn import ISRU_DLayer, ISRU_D
from util.crop import center_crop_arr
from util.datasets import celeba_train
from main_jit import get_args_parser
from denoiser import Denoiser
from util import misc


def algebra() -> dict:
    torch.manual_seed(0)
    errors = {}
    for order in (1, 8):
        layer = ISRU_DLayer(4, order).double()
        with torch.no_grad():
            layer.proj.weight.mul_(0.3)
        x = (torch.randn(2, 3, 4, dtype=torch.float64) * 0.3).requires_grad_()
        diagonal = torch.rsqrt(1 + 1.2 * x.square())
        matrix = layer.proj.weight * diagonal.unsqueeze(-2)
        polynomial = torch.eye(4, dtype=x.dtype).expand_as(matrix).clone()
        for k in range(1, order + 1):
            polynomial = polynomial + torch.linalg.matrix_power(matrix, k) / math.factorial(k)
        reference = (polynomial @ x.unsqueeze(-1)).squeeze(-1)
        actual = layer(x)
        torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
        direction = torch.randn_like(actual)
        g1 = torch.autograd.grad(actual, (x, layer.proj.weight), direction, retain_graph=True)
        g2 = torch.autograd.grad(reference, (x, layer.proj.weight), direction, retain_graph=True)
        for left, right in zip(g1, g2):
            torch.testing.assert_close(left, right, rtol=1e-11, atol=1e-12)
        if order == 8:
            exact = (torch.matrix_exp(matrix) @ x.unsqueeze(-1)).squeeze(-1)
            error = (actual - exact).abs().max().item()
            assert error < 1e-7
            errors['K8_matrix_exponential_max_error'] = error
    probe = ISRU_D(4).double()
    x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(probe, (x,), fast_mode=True)
    errors.update(K1_and_K8_dense_polynomial='PASS', input_and_P_derivatives='PASS', gradcheck='PASS')
    return errors


def entrypoint_resume(data_root: str) -> str:
    """Compare the actual entrypoint's two epochs with one epoch + resume."""
    import main_jit as entry
    import model_jit
    from torch.utils.data import Subset

    original_loader = entry.celeba_train
    name = '_celeba_resume_fixture'
    assert name not in model_jit.JiT_models
    model_jit.JiT_models[name] = lambda **kwargs: model_jit.JiT(
        depth=2, hidden_size=32, num_heads=4, patch_size=8, bottleneck_dim=16,
        in_context_len=2, in_context_start=1, **kwargs)
    entry.celeba_train = lambda root, transform: Subset(celeba_train(root, transform), range(4))
    try:
        with tempfile.TemporaryDirectory() as directory:
            full = Path(directory) / 'full'
            resumed = Path(directory) / 'resumed'

            def run(folder: Path, epochs: int, resume: bool = False) -> None:
                options = [
                    '--model', name, '--dataset', 'celeba', '--data_path', data_root,
                    '--img_size', '64', '--class_num', '1', '--label_drop_prob', '1', '--cfg', '0',
                    '--ffn', 'isru_d', '--ode_order', '8', '--seed', '0', '--batch_size', '2',
                    '--num_workers', '0', '--device', 'cpu', '--epochs', str(epochs),
                    '--warmup_epochs', '0', '--lr', '0.0001', '--save_last_freq', '1',
                    '--max_train_seconds', '500', '--log_freq', '1', '--output_dir', str(folder),
                ]
                if resume:
                    options.extend(['--resume', str(folder)])
                entry.main(get_args_parser().parse_args(options))

            run(full, 2)
            run(resumed, 1)
            run(resumed, 2, resume=True)
            a = torch.load(full / 'checkpoint-last.pth', weights_only=False, map_location='cpu')
            b = torch.load(resumed / 'checkpoint-last.pth', weights_only=False, map_location='cpu')

            def same(left, right) -> None:
                if isinstance(left, torch.Tensor):
                    torch.testing.assert_close(left, right, atol=0, rtol=0)
                elif isinstance(left, dict):
                    assert left.keys() == right.keys()
                    for key in left:
                        same(left[key], right[key])
                elif isinstance(left, (tuple, list)):
                    for first, second in zip(left, right):
                        same(first, second)
                else:
                    assert left == right

            for key in ('model', 'optimizer', 'model_ema1', 'model_ema2'):
                same(a[key], b[key])
            same(a['rng_state']['torch'], b['rng_state']['torch'])
            assert a['epoch'] == b['epoch'] == 1
            assert json.loads((resumed / 'progress.json').read_text())['status'] == 'complete'
    finally:
        entry.celeba_train = original_loader
        del model_jit.JiT_models[name]
    return 'PASS: actual entrypoint, checkpoint, optimizer, two EMAs and Torch RNG match bitwise'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--data-root', default='/scratch/zz5070/ODE_FFN/datasets')
    parser.add_argument('--full-model', action='store_true')
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = dict(algebra=algebra(), device=args.device, formal_training=False)
    dataset = celeba_train(args.data_root, transforms.Compose([
        transforms.Lambda(partial(center_crop_arr, image_size=64)), transforms.PILToTensor(),
    ]))
    assert len(dataset) == 162770
    for index in (0, 1234, len(dataset) - 1):
        image, label = dataset[index]
        assert image.shape == (3, 64, 64) and image.dtype == torch.uint8 and label == 0
    result['official_train_adapter'] = dict(status='PASS', images=len(dataset), shape=[3, 64, 64])
    if args.device == 'cpu':
        result['entrypoint_epoch_resume'] = entrypoint_resume(args.data_root)
    if args.device == 'cuda':
        assert torch.cuda.is_available()
        result['cuda'] = dict(torch=torch.__version__, cuda=torch.version.cuda,
                              device=torch.cuda.get_device_name(), arch_list=torch.cuda.get_arch_list())
        torch.manual_seed(0)
        layer = ISRU_D(768).cuda()
        x = torch.randn(8, 256, 768, device='cuda', requires_grad=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = layer(x).square().mean()
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
        del layer, x, loss
        gc.collect()
        torch.cuda.empty_cache()
        if args.full_model:
            records = {}
            for ffn in ('swiglu', 'isru_d'):
                opts = get_args_parser().parse_args([
                    '--model', 'JiT-B/4', '--img_size', '64', '--class_num', '1',
                    '--label_drop_prob', '1', '--cfg', '0', '--ffn', ffn, '--ode_order', '8',
                ])
                torch.manual_seed(0)
                model = Denoiser(opts).cuda()
                model.ema_params1 = copy.deepcopy(list(model.parameters()))
                model.ema_params2 = copy.deepcopy(list(model.parameters()))
                optimizer = torch.optim.AdamW(misc.add_weight_decay(model, 0), lr=1e-4, betas=(.9, .95))
                images = torch.stack([dataset[i][0] for i in range(args.batch_size)]).cuda().float() / 127.5 - 1
                labels = torch.zeros(args.batch_size, device='cuda', dtype=torch.long)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.monotonic()
                for _ in range(3):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        loss = model(images, labels)
                    assert torch.isfinite(loss)
                    loss.backward()
                    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                    optimizer.step()
                    model.update_ema()
                torch.cuda.synchronize()
                if ffn == 'isru_d':
                    assert model.net.blocks[0].mlp.ode1.proj.weight.grad.abs().sum() > 0
                records[ffn] = dict(parameters=sum(p.numel() for p in model.parameters()),
                                    batch=args.batch_size, updates=3, loss=loss.item(),
                                    seconds=time.monotonic() - started,
                                    peak_bytes=torch.cuda.max_memory_allocated())
                del optimizer, model, images, labels, loss
                gc.collect()
                torch.cuda.empty_cache()
            result['full_JiT_B4'] = records
        result['cuda']['kernel_dispatch'] = 'PASS'
    result['status'] = 'PASS'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    print('CELEBA_K8_CUDA_PASS' if args.device == 'cuda' else 'CELEBA_K8_CPU_PASS')


if __name__ == '__main__':
    main()
