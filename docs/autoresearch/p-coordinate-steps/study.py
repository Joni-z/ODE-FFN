"""Read-only optimizer-coordinate identities at real nonzero image states."""
from __future__ import annotations
import os
import sys
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'PYTHONDONTWRITEBYTECODE'):
    os.environ[key] = '1'
sys.dont_write_bytecode = True
import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
import time
from pathlib import Path
import numpy as np
import psutil
import torch

BASE = Path('/Users/mr.z/ode-workspace-prep/autoresearch')
PARENT = BASE/'oct29-image-swiglu-init-v1'
STEPS = (1000, 2500, 5000, 10000, 15000, 20000)
TOL = dict(rtol=1e-8, atol=1e-10)


def module(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec)
        sys.modules[name] = value
        spec.loader.exec_module(value)
    assert Path(sys.modules[name].__file__).resolve() == path.resolve()
    return sys.modules[name]


def parents():
    support = module('pcoordinate_parent_support', PARENT/'source/init_study.py')
    c, n, _ = support.parents()
    gm = module('pcoordinate_literal_math', support.GEOMETRY/'source/geometry_math.py')
    return support, c, n, gm


def verify(root):
    support, c, _, _ = parents()
    c.verify(root)
    for parent, expected in (
        (PARENT, '4a6af4a2fc61ca8a1f60d145014041f52f23379b1cb2ec454d1f12da744927cd'),
        (support.COLD, 'aad29b214ba537a2ee2034c60987c98f220eef00ca97b0105fb2e3bbadffff9c'),
        (support.GEOMETRY, 'd9d7ca3ab2ed93835cfc8b620c0dd4af54209ba4403cad87f47da95188df05f3')):
        assert support.sha(parent/'sealed-runtime.json') == expected
        for name, digest in support.load(parent/'sealed-runtime.json').items():
            assert support.sha(parent/name) == digest, (parent, name)
    for name, digest in support.load(support.NATIVE/'bindings.json').items():
        assert support.sha(name) == digest


def prepare(root):
    support, _, _, _ = parents()
    repo = Path(__file__).resolve().parents[3]
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True)
    assert not root.exists()
    root.mkdir()
    shutil.copytree(Path(__file__).parent, root/'source')
    (root/'doc-validation').mkdir()
    support.write(root/'settings.json', dict(source_commit=subprocess.check_output(
        ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(), steps=STEPS,
        new_training=False, model_seed=0, clipping_limit=1., native_eta=1e-4,
        native_epsilon=1e-8, environment_hash='475428fb', ceiling_seconds=300))
    paths = [*Path(root/'source').rglob('*'), root/'settings.json', PARENT/'sealed-runtime.json']
    for step in STEPS:
        for prefix in ('before_', 'step_'):
            paths.append(support.COLD/'output/runs/fixed06'/f'{prefix}{step:07d}.pt')
    support.write(root/'bindings.json', {str(p.resolve()): support.sha(p)
                  for p in paths if p.is_file() and p.suffix != '.pyc'})
    verify(root)
    print('P_COORDINATE_STEPS_PREPARED', flush=True)


def is_matrix(name):
    return name.startswith('blocks.') and '.mlp.fc' in name and name.endswith('.weight')


def source_name(name):
    if not is_matrix(name):
        return name
    return name.replace('.fc1.weight', '.ode1.proj.weight').replace('.fc2.weight', '.ode2.proj.weight')


def native_name_to_direct(name):
    return name.replace('.ode1.proj.weight', '.fc1.weight').replace('.ode2.proj.weight', '.fc2.weight')


def restore_flags(target, original):
    flags = {name: p.requires_grad for name, p in original.named_parameters()}
    for name, p in target.named_parameters():
        p.requires_grad_(flags[name])


def direct_optimizer(model, source, source_opt, c, corrected):
    named = dict(model.named_parameters())
    original = dict(source.named_parameters())
    group = source_opt.param_groups[0]
    assert group['lr'] == 1e-4 and group['eps'] == 1e-8 and group['weight_decay'] == 0
    assert group['betas'] == (.9, .99) and not group['amsgrad']
    opt = torch.optim.AdamW([
        dict(params=[v for k, v in named.items() if not is_matrix(k)], lr=group['lr'], eps=group['eps']),
        dict(params=[v for k, v in named.items() if is_matrix(k)],
             lr=group['lr']*c if corrected else group['lr'],
             eps=group['eps']/c if corrected else group['eps'])],
        betas=group['betas'], weight_decay=0.)
    for name, value in named.items():
        old = source_opt.state.get(original[source_name(name)])
        if old:
            state = {key: v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v)
                     for key, v in old.items()}
            assert set(state) == {'step', 'exp_avg', 'exp_avg_sq'}
            if is_matrix(name):
                state['exp_avg'].div_(c)
                state['exp_avg_sq'].div_(c*c)
            opt.state[value] = state
    return opt


def backward(model, optimizer, batch):
    optimizer.zero_grad(set_to_none=True)
    output = model(batch['x'].double(), batch['t'], torch.zeros(64, dtype=torch.long))
    # Promote the original native float32 denominators as fixed constants.
    delta = (1-batch['t']).clamp_min(.05).double()[:, None, None, None]
    loss = ((output-batch['target'].double())/delta).square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    return output.detach(), float(loss.detach())


def run(root):
    tick = time.monotonic()
    process = psutil.Process()
    process.cpu_percent()
    support, c, n, gm = parents()
    support.environment(root, 'training')
    verify(root)
    out = root/'output'
    out.mkdir()
    images, ids = n.data_for(support.NATIVE)
    bank = {k: v[:64] for k, v in c.bank().items()}
    rows, matrix_rows = [], []
    clock = c.ops.C
    for step in STEPS:
        assert time.monotonic()-tick < 300
        folder = support.COLD/'output/runs/fixed06'
        post = torch.load(folder/f'step_{step:07d}.pt', weights_only=False, map_location='cpu')
        native = c.construct('fixed06', post).eval()
        actual = n.predict(native, bank)
        with np.load(folder/f'validation_{step:07d}.npz') as saved:
            assert actual.tobytes() == saved['prediction'][:64].tobytes()
        literal32 = gm.literal_model(native, promote=False)
        assert n.predict(literal32, bank).tobytes() == actual.tobytes()
        lifted = gm.literal_model(native)
        with torch.no_grad():
            promoted = lifted(torch.from_numpy(bank['x']).double(), torch.from_numpy(bank['t']),
                              torch.zeros(64, dtype=torch.long)).numpy()
        np.testing.assert_allclose(promoted, actual, rtol=1e-4, atol=1e-5)
        bridge_error = float(np.max(np.abs(promoted-actual)))
        del native, literal32, lifted

        state = torch.load(folder/f'before_{step:07d}.pt', weights_only=False, map_location='cpu')
        original, old_opt, rng, digest = c.restored('fixed06', state)
        batch = n.draw(images, ids, rng)
        assert n.fingerprint(digest, batch) == post['input_stream_digest']
        a = gm.literal_model(original)
        restore_flags(a, original)
        opt_a = n.optimizer(a)
        opt_a.load_state_dict(copy.deepcopy(old_opt.state_dict()))
        b = c.ops.direct(copy.deepcopy(a))
        naive = copy.deepcopy(b)
        opt_b = direct_optimizer(b, a, opt_a, clock, True)
        opt_naive = direct_optimizer(naive, a, opt_a, clock, False)
        models = (a, b, naive)
        opts = (opt_a, opt_b, opt_naive)
        parameters = [dict(m.named_parameters()) for m in models]
        pa, pb, pc = parameters
        assert sum(p.numel() for p in a.parameters() if p.requires_grad) == 185744
        assert sum(p.numel() for p in b.parameters() if p.requires_grad) == 185744
        matrix_names = [name for name in pb if is_matrix(name)]
        assert len(matrix_names) == 6
        assert all(float(opt_a.state[pa[source_name(name)]]['step']) > 0 for name in matrix_names)
        before = {name: pb[name].detach().clone() for name in matrix_names}
        predictions, losses = [], []
        for m, opt in zip(models, opts):
            prediction, loss = backward(m, opt, batch)
            predictions.append(prediction)
            losses.append(loss)
        torch.testing.assert_close(predictions[0], predictions[1], **TOL)
        assert torch.equal(predictions[1], predictions[2])
        gradient_error = 0.
        ordered_b = []
        for name, value in pa.items():
            if '.t_func.' in name:
                assert value.grad is None and not value.requires_grad
                continue
            target_name = native_name_to_direct(name)
            other = pb[target_name]
            assert (value.grad is None) == (other.grad is None)
            ordered_b.append(other)
            if value.grad is not None:
                expected = other.grad*clock if is_matrix(target_name) else other.grad
                torch.testing.assert_close(value.grad, expected, **TOL)
                gradient_error = max(gradient_error, float((value.grad-expected).abs().max()))
        norm_a = torch.nn.utils.clip_grad_norm_(a.parameters(), 1.)
        for name in matrix_names:
            pb[name].grad.mul_(clock)
        norm_b = torch.nn.utils.clip_grad_norm_(ordered_b, 1.)
        for name in matrix_names:
            pb[name].grad.div_(clock)
        norm_c = torch.nn.utils.clip_grad_norm_([pc[name] for name in pb], 1.)
        torch.testing.assert_close(norm_a, norm_b, **TOL)
        for opt in opts:
            opt.step()
        parameter_error = moment_error = 0.
        for name, value in pb.items():
            source = pa[source_name(name)]
            expected = torch.eye(64, dtype=source.dtype)+clock*source if is_matrix(name) else source
            torch.testing.assert_close(value, expected, **TOL)
            parameter_error = max(parameter_error, float((value-expected).detach().abs().max()))
            if source in opt_a.state:
                sa, sb = opt_a.state[source], opt_b.state[value]
                assert torch.equal(sa['step'], sb['step'])
                for key, scale in (('exp_avg', clock), ('exp_avg_sq', clock*clock)):
                    expected_state = sa[key]/scale if is_matrix(name) else sa[key]
                    torch.testing.assert_close(sb[key], expected_state, **TOL)
                    moment_error = max(moment_error, float((sb[key]-expected_state).abs().max()))
        with torch.no_grad():
            outputs = [m(batch['x'].double(), batch['t'], torch.zeros(64, dtype=torch.long)) for m in models]
        torch.testing.assert_close(outputs[0], outputs[1], **TOL)
        for name in matrix_names:
            source = pa[source_name(name)]
            expected = torch.eye(64, dtype=source.dtype)+clock*source.detach()
            da = expected-before[name]
            db = pb[name].detach()-before[name]
            dc = pc[name].detach()-before[name]
            denominator = float(da.norm())
            matrix_rows.append(dict(step=step, layer=name, P_effective_increment=denominator,
                transported_W_increment=float(db.norm()), native_W_increment=float(dc.norm()),
                transported_ratio=float(db.norm())/denominator if denominator else None,
                native_ratio=float(dc.norm())/denominator if denominator else None))
        rows.append(dict(step=step, native_prefix_rows_bitwise=64, literal32_prefix_bitwise=True,
            promoted_native_max_error=bridge_error, active_gradients_checked=sum(p.grad is not None for p in a.parameters()),
            gradient_pullback_max_error=gradient_error, P_coordinate_norm=float(norm_a),
            transported_W_norm=float(norm_b), ordinary_W_norm=float(norm_c),
            P_clip_factor=float((1/(norm_a+1e-6)).clamp(max=1)),
            ordinary_clip_factor=float((1/(norm_c+1e-6)).clamp(max=1)),
            mapped_parameter_max_error=parameter_error, mapped_moment_max_error=moment_error,
            post_prediction_max_error=float((outputs[0]-outputs[1]).abs().max()),
            naive_post_prediction_RMS_difference=float((outputs[0]-outputs[2]).square().mean().sqrt()),
            pre_loss=losses[0], transported_pre_loss=losses[1], native_W_pre_loss=losses[2],
            CPU_percent=process.cpu_percent(), RSS=process.memory_info().rss))
        support.write(out/'progress.json', dict(completed=len(rows), total=6))
        print(json.dumps(rows[-1]), flush=True)
        del models, opts, a, b, naive, original, old_opt, opt_a, opt_b, opt_naive
    verify(root)
    assert time.monotonic()-tick < 300
    assert len(rows) == 6 and len(matrix_rows) == 36
    support.write(out/'result.json', dict(status='PASS', rows=rows, matrix_rows=matrix_rows,
        new_training=False, new_FID=False, new_ODE_method=False,
        full_native_float32_coordinate_equivalence_claimed=False,
        seconds=time.monotonic()-tick))
    print('P_COORDINATE_REAL_STATES_PASS', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.action](args.root.resolve())
