"""Bounded real-CIFAR check of the upstream epoch; no benchmark quality claim."""
import os
os.environ['TORCH_COMPILE_DISABLE'] = '1'
import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from main_jit import get_args_parser
from denoiser import Denoiser
from engine_jit import train_one_epoch
from util import misc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--batches', type=int, default=8)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-metrics', action='store_true')
    a = parser.parse_args()
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(4)
    torch._dynamo.config.disable = True
    torch.manual_seed(0)
    args = get_args_parser().parse_args([
        '--model', 'JiT-B/2', '--dataset', 'cifar10', '--img_size', '32',
        '--class_num', '10', '--label_drop_prob', '1', '--cfg', '0',
        '--num_sampling_steps', '3', '--num_workers', '0', '--lr', '0.0001',
        '--ffn', 'swiglu'])
    data = torchvision.datasets.CIFAR10(a.data_path, train=True,
                                      transform=transforms.PILToTensor(), download=False)
    assert len(data) == 50000
    loader = DataLoader(Subset(data, range(a.batch_size * a.batches)),
                        batch_size=a.batch_size, num_workers=0, pin_memory=True)
    model = Denoiser(args).cuda()
    model.ema_params1 = copy.deepcopy(list(model.parameters()))
    model.ema_params2 = copy.deepcopy(list(model.parameters()))
    optimizer = torch.optim.AdamW(misc.add_weight_decay(model, 0), lr=args.lr,
                                 betas=(.9, .95))
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    train_one_epoch(model, model, loader, optimizer, torch.device('cuda'), 6, args=args)
    elapsed = time.perf_counter() - started
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert model.net.blocks[0].mlp.w12.weight.grad.abs().sum() > 0
    training_peak = torch.cuda.max_memory_allocated()
    model.eval()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        samples = model.generate(torch.zeros(2, dtype=torch.long, device='cuda'))
    assert samples.shape == (2, 3, 32, 32) and torch.isfinite(samples).all()
    receipt = dict(torch=torch.__version__, torchvision=torchvision.__version__,
                   gpu=torch.cuda.get_device_name(), batch_size=a.batch_size,
                   batches=a.batches, parameters=sum(p.numel() for p in model.parameters()),
                   elapsed_seconds=elapsed, seconds_per_update=elapsed / a.batches,
                   peak_training_allocated_GiB=training_peak / 2**30,
                   max_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30,
                   dataset_examples=len(data), quality_or_FID_claim=False)
    if a.check_metrics:
        del model, optimizer, samples
        gc.collect()
        torch.cuda.empty_cache()
        from torch_fidelity.registry import DATASETS_REGISTRY
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
        reference = DATASETS_REGISTRY['cifar10-train'](a.data_path, False)
        assert len(reference) == 50000
        extractor = FeatureExtractorInceptionV3('inception-v3-compat', ['2048']).cuda().eval()
        pixels = torch.stack([reference[i] for i in range(4)]).cuda()
        with torch.no_grad():
            features = extractor(pixels)[0]
        assert features.shape == (4, 2048) and torch.isfinite(features).all()
        receipt['official_Inception_CUDA_features'] = list(features.shape)
    if a.output:
        with a.output.open('x') as f:
            json.dump(receipt, f, indent=2)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    print('CUDA_BASELINE_WITNESS_PASS', flush=True)


if __name__ == '__main__':
    main()
