"""Replay the real 60-epoch EMA and native baseline in production CUDA/BF16."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from denoiser import Denoiser
from ffn import MidpointLayer
from util.midpoint_eval import UnconditionalHeunEMA, atomic_json, file_hash


@torch.no_grad()
def verify(path, family, substeps):
    state = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    cfg = state['args']
    assert cfg.ffn == family and cfg.seed == 0 and cfg.ema_decay1 == .9999
    labels = torch.zeros(cfg.gen_bsz, device='cuda', dtype=torch.long)
    result = dict(epoch=state['epoch'] + 1, checkpoint_hash=file_hash(path), batch=cfg.gen_bsz)
    reference = None
    for name, cls in [('original', Denoiser), ('single_branch', UnconditionalHeunEMA)]:
        model = cls(cfg)
        model.load_state_dict(state['model_ema1'], strict=True)
        model.cuda().eval()
        for layer in model.modules():
            if isinstance(layer, MidpointLayer):
                layer.substeps = substeps
        seconds = []
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for trial in range(3):
                torch.manual_seed(0)
                torch.cuda.synchronize()
                start = time.perf_counter()
                image = model.generate(labels)
                torch.cuda.synchronize()
                if trial:
                    seconds.append(time.perf_counter() - start)
            assert image.isfinite().all()
            if reference is None:
                reference = image.cpu()
            else:
                actual = image.cpu()
                result['max_abs_difference'] = (reference - actual).abs().max().item()
                if not torch.equal(reference, actual):
                    raise RuntimeError(f'{family}/{substeps} differs by {result["max_abs_difference"]}')
        result[name + '_seconds'] = statistics.median(seconds)
        del model, image
    result['speedup'] = result['original_seconds'] / result['single_branch_seconds']
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'native', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    rows = {f'midpoint{s}': verify(args.run / 'checkpoint-0060.pth', 'midpoint', s) for s in (2, 4)}
    rows['swiglu'] = verify(args.native / 'checkpoint-0100.pth', 'swiglu', None)
    atomic_json(args.receipt, dict(status='PASS', gpu=torch.cuda.get_device_name(),
                                   gpu_uuid=str(torch.cuda.get_device_properties(0).uuid), results=rows))
