"""Check real entrypoint scheduling/RNG on a tiny CPU JiT; no FID benchmark."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import argparse
import contextlib
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main_jit
import model_jit


class Images(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = (torch.arange(3 * 64 * 64).reshape(3, 64, 64) + index).to(torch.uint8)
        if (torch.rand(()).item() + np.random.rand() + random.random()) / 3 < .5:
            image = image.flip(-1)
        return image, 0


def same(left, right) -> None:
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same(a, b)
    else:
        assert left == right


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    torch.set_num_threads(1)
    model_jit.JiT_models['_fid_fixture'] = lambda **kw: model_jit.JiT(
        depth=2, hidden_size=32, num_heads=4, patch_size=16,
        bottleneck_dim=16, in_context_len=2, in_context_start=1, **kw)
    main_jit.celeba_train = lambda *args: Images()
    calls, draws = [], []

    def fake_evaluate(model, args, epoch, batch_size=64, log_writer=None):
        model.eval()
        calls.append(epoch)
        draws.append(torch.randn(4))
        torch.randn(17)
        np.random.rand(19)
        random.random()
        assert log_writer is not None

    main_jit.evaluate = fake_evaluate
    flags = ['--dataset', 'celeba', '--model', '_fid_fixture', '--img_size', '64',
             '--class_num', '1', '--label_drop_prob', '1', '--cfg', '0', '--seed', '0',
             '--batch_size', '2', '--num_workers', '0', '--device', 'cpu',
             '--epochs', '100', '--lr', '.00005', '--save_last_freq', '1']
    assert main_jit.get_args_parser().parse_args([]).eval_freq == 20
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        states = []
        for label, online in [('training_only', False), ('periodic', True)]:
            out = root / label
            args = main_jit.get_args_parser().parse_args(flags + ['--output_dir', str(out)] +
                                                      (['--online_eval'] if online else []))
            random.seed(0)
            with (root / f'{label}.log').open('w') as log, contextlib.redirect_stdout(log):
                main_jit.main(args)
            states.append(torch.load(out / 'checkpoint-last.pth', map_location='cpu', weights_only=False))
        assert calls == [20, 40, 60, 80, 100], calls
        for draw in draws[1:]:
            same(draw, draws[0])
        for key in ('model', 'model_ema1', 'model_ema2', 'optimizer', 'rng_state'):
            same(states[0][key], states[1][key])
        assert states[1]['epoch'] == 99
        periodic = root / 'periodic'
        for epoch in calls:
            state = torch.load(periodic / f'checkpoint-{epoch:04d}.pth', map_location='cpu', weights_only=False)
            assert state['epoch'] + 1 == epoch
            assert {'model', 'model_ema1', 'model_ema2', 'optimizer', 'rng_state'} <= state.keys()
        # An explicit milestone file must load and log the completed epoch.
        checkpoint = periodic / 'checkpoint-0020.pth'
        args = main_jit.get_args_parser().parse_args(flags + ['--output_dir', str(root/'eval20'),
                                                               '--resume', str(checkpoint), '--evaluate_gen'])
        with (root/'eval20.log').open('w') as log, contextlib.redirect_stdout(log):
            main_jit.main(args)
        assert calls[-1] == 20
    result = dict(status='PASS', fixture='actual author JiT/training entrypoint, tiny CPU dimensions',
                  actual_FID_computed=False, online_epochs=[20, 40, 60, 80, 100],
                  full_milestone_checkpoints=True, identical_sampling_seed=True,
                  model_optimizer_two_EMAs_and_training_RNG_bitwise_unchanged=True,
                  explicit_milestone_resume_and_epoch_label=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
