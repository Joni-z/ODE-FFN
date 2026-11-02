"""Native wave2 witnesses: frozen controls, direction norm, sandwich and resume."""
import argparse
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import types

from variants import PARENT, MatchedMomentum, base, build_model, optimizer, torch


def close(a, b, tolerance=0.):
    torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)


def frozen_runtime(path):
    if path:
        filename = Path(path)/'run.py'
        source = filename.read_bytes()
    else:
        source = subprocess.check_output(['git', '-C', str(PARENT), 'show',
                                         '295dd3e:studies/muon_pd_20261102/run.py'])
        filename = PARENT/'run.py'
    # The numerical reference modules themselves are immutable and hash-checked.
    module = types.ModuleType('first_wave_frozen_reference')
    module.__file__ = str(filename)
    exec(compile(source, str(filename), 'exec'), module.__dict__)
    module.verify_sources()
    return module, hashlib.sha256(source).hexdigest()


def control_identity(old, device):
    checks = []
    for family in ('scalar_k2', 'vector_dp_k2', 'vector_pd_k2', 'swiglu128', 'swiglu192'):
        for method, lr in (('adamw', .0003), ('muon', .006)):
            torch.manual_seed(0)
            a = old.build_model(16, family).to(device)
            torch.manual_seed(0)
            b = build_model(16, family).to(device)
            for name, p in a.named_parameters():
                close(p, dict(b.named_parameters())[name])
            for name, p in a.named_buffers():
                close(p, dict(b.named_buffers())[name])
            with torch.no_grad():
                a.output.weight.normal_(0, .1)
                b.output.weight.copy_(a.output.weight)
            oa = old.ProjectionOptimizer(a, method, lr)
            ob = optimizer(b, dict(optimizer=method, projection_lr=lr, projection_decay=0.))
            for _ in range(3):
                x, t, y = torch.randn(16, 16, device=device), torch.rand(16, 1, device=device), torch.randn(16, 16, device=device)
                for m, o in ((a, oa), (b, ob)):
                    o.zero_grad(set_to_none=True)
                    loss = (m(x, t)-y).square().mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
                    o.step()
                for name, p in a.named_parameters():
                    close(p, dict(b.named_parameters())[name])
            checks.append(f'{family}/{method}')
    return checks


def matched_update(device):
    maximum_error = 0.
    for shape in ((64, 64), (170, 64), (64, 85)):
        p = torch.nn.Parameter(torch.randn(shape, device=device))
        reference = torch.nn.Parameter(p.detach().clone())
        candidate = MatchedMomentum([p], .02)
        native = base.SingleDeviceMuon([reference], lr=.02, momentum=.95, weight_decay=0.)
        for index in range(3):
            gradient = torch.randn_like(p) if index else torch.zeros_like(p)
            previous_momentum = candidate.state.get(p, {}).get('momentum_buffer', torch.zeros_like(p)).clone()
            expected_momentum = previous_momentum.lerp(gradient, .05)
            raw = gradient.lerp(expected_momentum, .95)
            before = p.detach().clone()
            old_reference = reference.detach().clone()
            p.grad, reference.grad = gradient.clone(), gradient.clone()
            candidate.step()
            native.step()
            delta = p.detach()-before
            muon_delta = reference.detach()-old_reference
            close(candidate.state[p]['momentum_buffer'], native.state[reference]['momentum_buffer'])
            error = float(abs(delta.norm()-muon_delta.norm()))
            maximum_error = max(maximum_error, error)
            close(delta.norm(), muon_delta.norm(), 2e-6)
            if index:
                cosine = torch.nn.functional.cosine_similarity((-delta).flatten(), raw.flatten(), dim=0)
                assert float(cosine) > .99999
            else:
                close(delta, torch.zeros_like(delta))
    return dict(maximum_actual_update_norm_difference=maximum_error, direction_preserved=True)


def sandwich(device):
    torch.manual_seed(0)
    original = base.build_model(2, 'vector_dp_k2')
    torch.manual_seed(0)
    model = build_model(2, 'vector_sandwich_k2')
    for name, p in original.state_dict().items():
        close(p, model.state_dict()[name])
    assert model.parameter_count() == original.parameter_count()
    layer = next(model.ode_layers())[1].to(device).double()
    h = torch.linspace(-1., 1., 64, device=device, dtype=torch.float64)

    def dense(x):
        s = layer.step_size(x[None])[0]
        d = torch.diag(s)
        root = torch.diag(s.sqrt())
        inverse = torch.diag(s.rsqrt())
        p = layer.proj.weight
        a = root@p@root
        close(inverse@a@root, p@d, 1e-12)
        close(root@a@inverse, d@p, 1e-12)
        return x+a@x+a@(a@x)/2

    y = layer(h[None])[0]
    close(y, dense(h), 1e-12)
    jac = torch.autograd.functional.jacobian(lambda x: layer(x[None])[0], h)
    close(jac, torch.autograd.functional.jacobian(dense, h), 1e-11)
    direction = torch.cos(torch.arange(64, device=device, dtype=torch.float64))
    epsilon = 1e-5
    numerical = (layer((h+epsilon*direction)[None])[0]-layer((h-epsilon*direction)[None])[0])/(2*epsilon)
    close(jac@direction, numerical, 1e-8)
    stats = layer.activation_stats(h[None])
    close(torch.tensor(stats['output_rms_over_input']), (y.square().mean()/h.square().mean()).sqrt().float(), 1e-6)
    assert 'centered_curvature_rms_over_input' not in stats
    return dict(parameter_count=model.parameter_count(), dense_and_derivatives=True,
                similarity_identities=True, activation_diagnostics=True)


def resume(device):
    passed = []
    for family, method, decay in [('scalar_k2', 'matched_momentum', 0.),
                                   ('swiglu128', 'matched_momentum', 0.),
                                   ('scalar_k2', 'muon', .1), ('swiglu128', 'muon', .5),
                                   ('vector_sandwich_k2', 'adamw', 0.),
                                   ('vector_sandwich_k2', 'muon', 0.)]:
        torch.manual_seed(0)
        model = build_model(16, family).to(device)
        cell = dict(optimizer=method, projection_lr=.0003 if method == 'adamw' else .02,
                    projection_decay=decay)
        opt = optimizer(model, cell)
        if method != 'matched_momentum':
            assert all(g['weight_decay'] == decay for g in opt.projections.param_groups)
        assert all(g['weight_decay'] == 0. for g in opt.aux.param_groups)
        rng = torch.Generator(device=device).manual_seed(100000)

        def update():
            x = torch.randn(32, 16, device=device, generator=rng)
            t = torch.rand(32, 1, device=device, generator=rng)
            opt.zero_grad(set_to_none=True)
            loss = (model(x, t)-torch.sin(x)).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            assert torch.isfinite(loss)
            return float(loss.detach())
        update()
        update()
        state = base.train.capture_state(model, opt, rng, 2, 'cfg', 'data', dict(seed=0))
        buffer = io.BytesIO()
        torch.save(state, buffer)
        expected_loss = update()
        expected = {n: p.detach().clone() for n, p in model.named_parameters()}
        buffer.seek(0)
        restored = torch.load(buffer, map_location=device, weights_only=False)
        base.train.restore_model(model, restored)
        opt.load_state_dict(restored['optimizer'])
        rng.set_state(restored['data_rng'].cpu())
        assert update() == expected_loss
        for n, p in model.named_parameters():
            close(p, expected[n])
        passed.append(f'{family}/{method}/wd{decay}')
    return passed


def native_smoke(data_root, output, device):
    grid = json.loads((Path(__file__).with_name('grid.json')).read_text())
    dataset = ('rotated_product_bimodal_16d' if (Path(data_root)/'product16_v1').exists()
               and not (Path(data_root)/'v1/gmm_ring8_2d').exists() else 'gmm_ring8_2d')
    selected = []
    predicates = [lambda c: c['family'] == 'scalar_k2' and c['optimizer'] == 'matched_momentum' and c['projection_lr'] == .006,
                  lambda c: c['family'] == 'scalar_k2' and c['projection_decay'] == .1,
                  lambda c: c['family'] == 'vector_sandwich_k2' and c['optimizer'] == 'muon' and c['projection_lr'] == .006]
    import sys
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for predicate in predicates:
        cell = next(c for c in grid['cells'] if c['dataset'] == dataset and predicate(c))
        command = [sys.executable, str(Path(__file__).with_name('train_wave2.py')),
                   '--cell', cell['id'], '--data-root', str(data_root), '--output', str(root),
                   '--device', device, '--steps', '8']
        with (root/(cell['id']+'.log')).open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        folder = root/'muon_pd_wave2_20261102'/dataset/cell['id']/'seed000'
        assert (folder/'ADAPTER_COMPLETE').exists()
        selected.append(dict(cell=cell['id'], steps=8, complete=True))
    # Exercise the actual inherited CUDA checkpoint-resume path, not only an
    # isolated optimizer roundtrip. Preserve all smoke snapshots.
    first = selected[0]
    folder = root/'muon_pd_wave2_20261102'/dataset/first['cell']/'seed000'
    command = [sys.executable, str(Path(__file__).with_name('train_wave2.py')),
               '--cell', first['cell'], '--data-root', str(data_root), '--output', str(root),
               '--device', device, '--steps', '16', '--resume', str(folder/'checkpoints/step_0000008.pt')]
    with (root/'resume.log').open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    assert (folder/'checkpoints/step_0000016.pt').exists() and (folder/'ADAPTER_COMPLETE').exists()
    return dict(cells=selected, native_resume_to_step=16)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--frozen-source')
    parser.add_argument('--data-root')
    parser.add_argument('--smoke-output')
    args = parser.parse_args()
    torch.set_num_threads(1)
    base.verify_sources()
    old, reference_hash = frozen_runtime(args.frozen_source)
    result = dict(passed=True, device=args.device, torch=torch.__version__,
                  frozen_run_sha256=reference_hash,
                  exact_three_update_controls=control_identity(old, args.device),
                  matched_update=matched_update(args.device), sandwich=sandwich(args.device),
                  exact_physical_resume=resume(args.device))
    if args.data_root or args.smoke_output:
        assert args.data_root and args.smoke_output
        result['native_smoke'] = native_smoke(args.data_root, args.smoke_output, args.device)
    base.dump(args.output, result)
    print(json.dumps(result))
