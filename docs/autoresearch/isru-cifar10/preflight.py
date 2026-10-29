"""Seeded actual kernels, original operators, active attention and physical resume."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
import importlib.metadata
import importlib.util
from pathlib import Path
import numpy as np
import torch
from common import cpu_tree, load, save_torch, verify, write
from model import ARMS, FreeISRU, SwiGLU, build_all, native_modules
from train import batch, make_models, objective, restore, sample, snapshot, update, validation


def compare(x, y, atol=2e-5, rtol=2e-4):
    torch.testing.assert_close(x, y, atol=atol, rtol=rtol)
    return float((x.detach().cpu()-y.detach().cpu()).abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    assert not a.output.exists()
    a.output.mkdir()
    config, paths = verify(a.root)
    torch.set_num_threads(1)
    assert torch.backends.mps.is_available()
    for name, spec_name in (('training-env-spec.json', 'image-training'),):
        for phase in load(a.root/name)['pip_phases']:
            for package in phase:
                package_name, expected = package.split('==')
                assert importlib.metadata.version(package_name) == expected
    _, ffn = native_modules(a.root/'source/native')
    spec = importlib.util.spec_from_file_location('original_synthetic', a.root/'source/synthetic_reference.py')
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    torch.manual_seed(0)
    old = ffn.ODELayer(in_features=8, hidden_features=8, orders=2, t_embed_dim=None).double()
    original, new = reference.FreeISRU(old), FreeISRU(old.proj.weight)
    x = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    u, v = original(x), new(y)
    compare(u, v, 1e-12, 1e-12)
    u.square().sum().backward(); v.square().sum().backward()
    compare(x.grad, y.grad, 1e-11, 1e-11)
    for p, q in zip(original.parameters(), new.parameters()):
        compare(p.grad, q.grad, 1e-11, 1e-11)
    assert torch.autograd.gradcheck(new, (x.detach().requires_grad_(True),), fast_mode=True)
    # Independent analytic input Jacobian, including both learned matrices.
    row = x.detach()[0, 0]
    inside = torch.nn.functional.linear(row, new.matrix1)
    expected = new.matrix2@torch.diag((1+1.2*inside.square()).pow(-1.5))@new.matrix1
    compare(torch.autograd.functional.jacobian(new, row), expected, 1e-11, 1e-11)
    native = ffn.SwiGLUFFN(8, 24).double()
    plain = SwiGLU(8, native.w3.in_features).double()
    plain.w12.load_state_dict(native.w12.state_dict())
    plain.w3.load_state_dict(native.w3.state_dict())
    x = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    u, v = native(x), plain(y)
    compare(u, v, 1e-12, 1e-12)
    u.square().sum().backward(); v.square().sum().backward()
    compare(x.grad, y.grad, 1e-12, 1e-12)
    for p, q in zip(native.parameters(), plain.parameters()):
        compare(p.grad, q.grad, 1e-12, 1e-12)
    print('ORIGINAL_ISRU_AND_SWIGLU_PASS', flush=True)
    # Open zero-initialized output and residual gates, so this probes active FFNs.
    manual = build_all(a.root/'source/native', fast_attention=False)['isru'].to('mps')
    with torch.no_grad():
        for name, value in manual.named_parameters():
            if 'adaLN_modulation' in name or name.startswith('final_layer.linear'):
                value.normal_(0, .02)
    fast = build_all(a.root/'source/native')['isru'].to('mps')
    fast.load_state_dict(manual.state_dict())
    rng = torch.Generator().manual_seed(0)
    x = torch.randn(2, 3, 32, 32, generator=rng).to('mps').requires_grad_(True)
    y = x.detach().clone().requires_grad_(True)
    t = torch.tensor([.2, .8], device='mps')
    labels = torch.zeros(2, device='mps', dtype=torch.long)
    u, v = manual(x, t, labels), fast(y, t, labels)
    forward_error = compare(u, v)
    u.square().sum().backward(); v.square().sum().backward()
    input_error = compare(x.grad, y.grad)
    gradient_error = 0.
    active_ffns = 0
    for (name, p), (_, q) in zip(manual.named_parameters(), fast.named_parameters()):
        if p.grad is not None:
            gradient_error = max(gradient_error, compare(p.grad, q.grad, 2e-4, 1e-3))
            if '.mlp.matrix' in name:
                assert p.grad.abs().sum() > 0
                active_ffns += 1
    assert active_ffns == 12
    del manual, fast, x, y, u, v
    torch.mps.empty_cache()
    print('ACTIVE_ATTENTION_FORWARD_AND_GRADIENT_PASS', flush=True)
    images = np.load(Path(paths['data'])/'images.uint8.npy', mmap_mode='r')
    split = np.load(Path(paths['data'])/'split.npz')
    models, optimizers, emas = make_models(a.root, config)
    rng = torch.Generator().manual_seed(config['stream_seed'])
    digest = '0'*64
    for step in range(1, 3):
        data = batch(images, split['train'], rng, 4, config)
        digest = data[3]
        for arm in ARMS:
            loss, norm, _ = update(models[arm], optimizers[arm], emas[arm], data, step, config)
            assert np.isfinite(loss+norm)
    save_torch(a.output/'resume.pt', snapshot(models, optimizers, emas, rng, 2, digest, a.root))
    for step in range(3, 5):
        data = batch(images, split['train'], rng, 4, config)
        for arm in ARMS:
            update(models[arm], optimizers[arm], emas[arm], data, step, config)
    expected_rng = rng.get_state()
    expected = cpu_tree(dict(models={a: m.state_dict() for a, m in models.items()},
                         emas={a: m.state_dict() for a, m in emas.items()},
                         optimizers={a: o.state_dict() for a, o in optimizers.items()}))
    # New model/optimizer objects; a serialized state, not aliased in-memory dicts.
    del models, optimizers, emas
    torch.mps.empty_cache()
    models, optimizers, emas = make_models(a.root, config)
    restored = torch.load(a.output/'resume.pt', map_location='cpu', weights_only=False)
    restored_step, _ = restore(restored, models, optimizers, emas, rng, a.root)
    assert restored_step == 2
    for step in range(3, 5):
        data = batch(images, split['train'], rng, 4, config)
        for arm in ARMS:
            update(models[arm], optimizers[arm], emas[arm], data, step, config)
    assert torch.equal(expected_rng, rng.get_state())
    maximum = 0.
    def recursive_compare(x, y):
        nonlocal maximum
        if isinstance(x, torch.Tensor):
            maximum = max(maximum, compare(x.cpu(), y.cpu(), 2e-6, 2e-4))
        elif isinstance(x, dict):
            assert x.keys() == y.keys()
            for key in x:
                recursive_compare(x[key], y[key])
        elif isinstance(x, (list, tuple)):
            assert len(x) == len(y)
            for p, q in zip(x, y):
                recursive_compare(p, q)
        else:
            assert x == y
    actual = dict(models={a: m.state_dict() for a, m in models.items()},
                  emas={a: m.state_dict() for a, m in emas.items()},
                  optimizers={a: o.state_dict() for a, o in optimizers.items()})
    recursive_compare(expected, actual)
    for arm in ARMS:
        assert sum(float(p.grad.abs().sum()) for p in models[arm].blocks[0].mlp.parameters()) > 0
        assert any(not torch.equal(models[arm].state_dict()[k], emas[arm].state_dict()[k])
                   for k in models[arm].state_dict())
        generated = sample(emas[arm], torch.randn(2, 3, 32, 32, generator=rng), config, steps=3)
        assert generated.shape == (2, 3, 32, 32)
    small = dict(config, validation_images=4, validation_batch=2)
    values = validation(models, emas, images, split['validation'], small)
    assert all(np.isfinite(v) for row in values.values() for v in row.values())
    write(a.output/'result.json', dict(status='PASS', device='mps', original_ISRU_forward_and_gradients=True,
        original_SwiGLU_forward_and_gradients=True, ISRU_analytic_Jacobian=True,
        active_FFN_matrices=active_ffns, attention_forward_max=forward_error,
        attention_input_gradient_max=input_error, attention_parameter_gradient_max=gradient_error,
        physical_resume_max=maximum, RNG_resume_bitwise=True, full_state_resume=True,
        actual_real_CIFAR_optimizer_EMA_validation_and_sampler=True, formal_quality_training=False))
    print('TRAINING_WITNESS_PASS', flush=True)


if __name__ == '__main__':
    main()
