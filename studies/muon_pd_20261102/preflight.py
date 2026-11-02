"""Seeded operator, pairing, official optimizer and physical-resume witnesses."""
import argparse
import copy
import io
import json
from pathlib import Path
import time

from run import (HERE, ProjectionOptimizer, SingleDeviceMuon, VelocityNet,
                 build_model, digest, dump, projection_parameters, torch, train,
                 verify_sources)


def close(a, b, tol=1e-10):
    torch.testing.assert_close(a, b, atol=tol, rtol=tol)


def mathematical_witness(device):
    torch.manual_seed(0)
    original = VelocityNet(2, 'ode_original', orders=2)
    torch.manual_seed(0)
    scalar = build_model(2, 'scalar_k2')
    for name, p in original.state_dict().items():
        close(p, scalar.state_dict()[name], 0)
    with torch.no_grad():
        original.output.weight.normal_(0, .1)
        scalar.output.weight.copy_(original.output.weight)
    x, t = torch.randn(4, 2), torch.rand(4, 1)
    close(original(x, t), scalar(x, t), 0)
    models = {}
    for family in ('scalar_k2', 'vector_dp_k2', 'vector_pd_k2', 'swiglu128', 'swiglu192'):
        torch.manual_seed(0)
        models[family] = build_model(2, family).to(device)
    source = models['scalar_k2']
    for family, model in models.items():
        for name, p in model.named_parameters():
            if '.ffn.' not in name:
                close(p, dict(source.named_parameters())[name], 0)
        for name, b in model.named_buffers():
            close(b, dict(source.named_buffers())[name], 0)
    dp, pd = models['vector_dp_k2'], models['vector_pd_k2']
    for name, p in dp.state_dict().items():
        close(p, pd.state_dict()[name], 0)
    for (_, a), (_, b) in zip(source.ode_layers(), dp.ode_layers()):
        close(a.proj.weight, b.proj.weight, 0)
    maximum_error = 0.
    outputs = {}
    for family in ('scalar_k2', 'vector_dp_k2', 'vector_pd_k2'):
        layer = copy.deepcopy(next(models[family].ode_layers())[1]).double()
        h = torch.linspace(-1, 1, 64, device=device, dtype=torch.float64).requires_grad_()
        def dense(z):
            gate = layer.step_size(z[None])[0]
            d = torch.diag(gate.expand(64))
            p = layer.proj.weight
            a = p@d if family == 'vector_pd_k2' else d@p
            return z + a@z + a@(a@z)/2
        y, expected = layer(h[None])[0], dense(h)
        close(y, expected)
        maximum_error = max(maximum_error, float((y-expected).abs().max().detach()))
        jy = torch.autograd.functional.jacobian(lambda z: layer(z[None])[0], h)
        je = torch.autograd.functional.jacobian(dense, h)
        close(jy, je)
        direction = torch.cos(torch.arange(64, dtype=h.dtype, device=device))
        epsilon = 1e-5
        finite = (layer((h+epsilon*direction)[None])[0]-layer((h-epsilon*direction)[None])[0])/(2*epsilon)
        close(jy@direction, finite, 1e-8)
        outputs[family] = y
    if torch.allclose(outputs['vector_dp_k2'], outputs['vector_pd_k2']):
        raise AssertionError('PD and DP unexpectedly identical')
    return dict(dense_polynomial_max_abs_error=maximum_error,
                counts={f: m.parameter_count() for f, m in models.items()},
                pairing=True, gate_derivatives=True, original_forward_identity=True)


def optimizer_witness(device, name):
    torch.manual_seed(0)
    model = build_model(2, 'scalar_k2').to(device)
    with torch.no_grad():
        model.output.weight.normal_(0, .1)
    lr = .006 if name == 'muon' else .0003
    optimizer = ProjectionOptimizer(model, name, lr)
    p = optimizer.selected[0][1]
    reference_p = torch.nn.Parameter(p.detach().clone())
    native = (SingleDeviceMuon([reference_p], lr=lr, weight_decay=0) if name == 'muon'
              else torch.optim.AdamW([reference_p], lr=lr, weight_decay=0))
    generator = torch.Generator(device=device).manual_seed(100000)

    def update(native_check=False):
        x = torch.randn(32, 2, device=device, generator=generator)
        t = torch.rand(32, 1, device=device, generator=generator)
        target = torch.sin(x)
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x, t)-target).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if native_check:
            reference_p.grad = p.grad.clone()
            native.step()
        optimizer.step()
        assert torch.isfinite(loss)
        return float(loss.detach())

    update(True)
    close(p, reference_p, 0)
    update()
    state = train.capture_state(model, optimizer, generator, 2, 'config', 'data', {'seed': 0})
    storage = io.BytesIO()
    torch.save(state, storage)
    expected_loss = update()
    expected = {n: v.detach().clone() for n, v in model.named_parameters()}
    storage.seek(0)
    restored = torch.load(storage, weights_only=False, map_location=device)
    train.restore_model(model, restored)
    optimizer.load_state_dict(restored['optimizer'])
    generator.set_state(restored['data_rng'].cpu())
    actual_loss = update()
    assert actual_loss == expected_loss
    for n, p in model.named_parameters():
        close(p, expected[n], 0)
    assert optimizer.count == 3
    return dict(native_update_exact=True, serialized_resume_exact=True, loss=actual_loss)


def benchmark(device, updates):
    rows = []
    for family, name in [('scalar_k2', 'adamw'), ('scalar_k2', 'muon'),
                         ('vector_pd_k2', 'adamw'), ('swiglu192', 'muon')]:
        torch.manual_seed(0)
        model = build_model(16, family).to(device)
        optimizer = ProjectionOptimizer(model, name, .006 if name == 'muon' else .0003)
        rng = torch.Generator(device=device).manual_seed(100000)
        bank = torch.randn(100000, 16, generator=rng, device=device)
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        def one():
            x1 = bank[torch.randint(len(bank), (512,), device=device, generator=rng)]
            x0 = torch.randn(x1.shape, device=device, generator=rng)
            t = torch.rand(512, 1, device=device, generator=rng)
            optimizer.zero_grad(set_to_none=True)
            loss = (model((1-t)*x0+t*x1, t)-(x1-x0)).square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            assert torch.isfinite(norm)
            optimizer.step()
            return float(loss.detach()), float(norm)
        for _ in range(15):
            one()
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(updates):
            loss, norm = one()
        if device == 'cuda':
            torch.cuda.synchronize()
        rows.append(dict(family=family, optimizer=name, seconds_per_update=(time.perf_counter()-start)/updates,
                         loss=loss, gradient_norm=norm,
                         peak_cuda_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', required=True, choices=['cpu', 'cuda'])
    p.add_argument('--output', required=True)
    p.add_argument('--updates', type=int, default=120)
    args = p.parse_args()
    torch.set_num_threads(1)
    verify_sources()
    result = dict(device=args.device, torch=torch.__version__,
                  cuda_name=torch.cuda.get_device_name() if args.device == 'cuda' else None,
                  source_manifest_sha256=digest(HERE/'source_provenance.json'),
                  mathematics=mathematical_witness(args.device),
                  optimizers={name: optimizer_witness(args.device, name) for name in ('adamw', 'muon')},
                  benchmark=benchmark(args.device, args.updates), passed=True)
    dump(args.output, result)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
