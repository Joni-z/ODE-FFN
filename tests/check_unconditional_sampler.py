"""Count Heun50 forwards and compare final samples when every CFG weight is zero."""
import os
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import copy
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model_jit
from denoiser import Denoiser
from ffn import MidpointLayer


def unconditional_velocity(self, image, time, labels):
    null_labels = torch.full_like(labels, self.num_classes)
    prediction = self.net(image, time.flatten(), null_labels)
    return (prediction - image) / (1.0 - time).clamp_min(self.t_eps)


@torch.no_grad()
def check(family, substeps):
    torch.manual_seed(0)
    cfg = SimpleNamespace(
        model='fixture', img_size=32, class_num=1, attn_dropout=0., proj_dropout=0.,
        ffn=family, ode_hidden_dim=0, label_drop_prob=1., P_mean=-.8, P_std=.8,
        t_eps=.05, noise_scale=.5, ema_decay1=.9999, ema_decay2=.9996,
        sampling_method='heun', num_sampling_steps=50, cfg=0., interval_min=0., interval_max=1.,
    )
    model = Denoiser(cfg).eval()
    for layer in model.modules():
        if isinstance(layer, MidpointLayer):
            layer.substeps = substeps
    # A fresh JiT has a zero output head. Make conditioning observable in this fixture.
    for name, parameter in model.named_parameters():
        if 'adaLN_modulation' in name or 'final_layer.linear' in name:
            parameter.normal_(0, .08)
    labels = torch.zeros(2, dtype=torch.long)
    image = torch.randn(2, 3, 32, 32)
    time = torch.full((2,), .5)
    assert not torch.equal(model.net(image, time, labels), model.net(image, time, labels + 1))

    fast = copy.deepcopy(model)
    fast._forward_sample = MethodType(unconditional_velocity, fast)
    samples = []
    for candidate, expected_calls in ((model, 198), (fast, 99)):
        visited = []
        hook = candidate.net.register_forward_pre_hook(
            lambda module, args: visited.append(args[1][0].item()))
        torch.manual_seed(0)
        samples.append(candidate.generate(labels))
        hook.remove()
        assert len(visited) == expected_calls
        assert min(visited) == 0 and max(visited) < 1
    assert torch.isfinite(samples[0]).all() and torch.equal(*samples)
    print(f'{family}, substeps={substeps}: 198 vs 99 calls; identical final samples')


if __name__ == '__main__':
    torch.set_num_threads(1)
    model_jit.JiT_models['fixture'] = lambda **kw: model_jit.JiT(
        depth=2, hidden_size=32, num_heads=4, patch_size=16, bottleneck_dim=16,
        in_context_len=2, in_context_start=1, **kw)
    for family, substeps in [('swiglu', None), ('midpoint', 2), ('midpoint', 4)]:
        check(family, substeps)
