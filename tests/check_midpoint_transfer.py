"""Tiny JiT test of the paired checkpoint path and equality with the synthetic solver."""
import os
os.environ['TORCH_COMPILE_DISABLE'] = '1'
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ffn import MidpointLayer
import main_jit
import model_jit
from denoiser import Denoiser
from util import midpoint_eval
from torch.utils.tensorboard import SummaryWriter


def check_solver():
    torch.manual_seed(8)
    layer = MidpointLayer(5).double()
    x = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    for n in (2, 4):
        layer.substeps = n
        y = x
        def rhs(z):
            return (z * torch.rsqrt(1 + 1.2 * z.square())) @ layer.proj.weight.T
        for _ in range(n):
            y = y + rhs(y + rhs(y) / (2*n)) / n
        actual = layer(x)
        torch.testing.assert_close(actual, y)
        for a, b in zip(torch.autograd.grad(actual.sum(), (x, layer.proj.weight)),
                        torch.autograd.grad(y.sum(), (x, layer.proj.weight))):
            torch.testing.assert_close(a, b)


def check_pair(root):
    model_jit.JiT_models['fixture'] = lambda **kw: model_jit.JiT(
        depth=2, hidden_size=32, num_heads=4, patch_size=16, bottleneck_dim=16,
        in_context_len=2, in_context_start=1, **kw)
    flags = ['--dataset','celeba','--model','fixture','--img_size','64','--class_num','1',
             '--label_drop_prob','1','--cfg','0','--seed','0','--device','cpu','--gen_bsz','2',
             '--num_images','6','--num_sampling_steps','2','--output_dir',str(root),
             '--ffn','midpoint','--paired_midpoint_eval','--native_checkpoint_dir',str(root/'native')]
    cfg = main_jit.get_args_parser().parse_args(flags)
    cfg.native_checkpoint_dir.mkdir()
    source = None
    for arm in ('midpoint','swiglu'):
        args = copy.copy(cfg)
        args.ffn = arm
        model = Denoiser(args)
        if arm == 'midpoint':
            source = model
        state = model.state_dict()
        with torch.no_grad():
            # Force distinct model and EMA to detect wrong-weight selection.
            ema = {k: v.clone() for k,v in state.items()}
            ema['net.final_layer.linear.bias'].fill_(.123)
        for epoch in (20,100):
            path = (root if arm=='midpoint' else cfg.native_checkpoint_dir) / f'checkpoint-{epoch:04d}.pth'
            torch.save(dict(model=state, model_ema1=ema, epoch=epoch-1, args=args), path)

    seen = []
    def fid_fixture(model, args, epoch, **kw):
        assert kw['preloaded_ema']
        assert torch.allclose(model.net.final_layer.linear.bias, torch.tensor(.123))
        steps = {m.substeps for m in model.modules() if isinstance(m, MidpointLayer)}
        seen.append((epoch, steps.pop(), torch.randn(4)))
        return 1.0

    rng = torch.get_rng_state()
    with SummaryWriter(root/'events') as writer, patch.object(midpoint_eval, 'evaluate', fid_fixture):
        for epoch in (20,100):
            row = midpoint_eval.evaluate_pair(cfg, epoch, writer)
            assert bool(row['results']['swiglu']) == (epoch == 100)
            assert set(row['results']) == {'swiglu','midpoint2','midpoint4'}
            assert all(row['results'][k]['images_per_second'] > 0 for k in ('midpoint2','midpoint4'))
            assert row == midpoint_eval.evaluate_pair(cfg, epoch, writer)
        # Resume a partially evaluated milestone and reuse its first completed FID.
        d = root/'paired/epoch-0020'
        (d/'comparison.json').unlink()
        (d/'midpoint4.json').unlink()
        midpoint_eval.evaluate_pair(cfg,20,writer)
    assert [x[:2] for x in seen] == [(20,2),(20,4),(100,2),(100,4),(20,4)]
    assert all(torch.equal(seen[0][2], row[2]) for row in seen)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(m.substeps == 4 for m in source.modules() if isinstance(m, MidpointLayer))


if __name__ == '__main__':
    torch.set_num_threads(1)
    check_solver()
    with tempfile.TemporaryDirectory() as root:
        check_pair(Path(root))
    print('MIDPOINT_TRANSFER_CPU_PASS')
