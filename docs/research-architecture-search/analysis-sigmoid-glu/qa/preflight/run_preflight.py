"""Generic sigmoid-GLU control preflight; never launches a formal worker."""
from argparse import Namespace
from pathlib import Path
import hashlib, importlib.util, json, os, platform, re, shutil, subprocess, sys, threading, time
from unittest.mock import patch
import numpy as np
import psutil
import scipy
import torch

root = Path(__file__).resolve().parents[1]
source, out = root/'source', root/'preflight/attempt1'
prior = Path('/Users/mr.z/ode-workspace-prep/architecture-search/state_clock_20261025_m1')
assert prior != root and prior.name == 'state_clock_20261025_m1'
assert root.name == 'sigmoid_glu_20261025_m1'
out.mkdir()
sys.path.insert(0, str(source))
torch.set_num_threads(1)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path, value): path.write_text(json.dumps(value, indent=2)+'\n')
manifest = json.loads((root/'source-manifest.json').read_text())
oldmanifest = json.loads((prior/'source-manifest.json').read_text())
assert manifest['parent_source'] == str(prior/'source') and len(oldmanifest['all_files']) == 580
assert manifest['parent_manifest_sha256'] == sha(prior/'source-manifest.json')
def verify():
    for directory, record in ((source, manifest), (prior/'source', oldmanifest)):
        names = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        assert names == set(record['all_files'])
        for name, digest in record['all_files'].items(): assert sha(directory/name) == digest, name
def equal(a, b):
    if isinstance(a, torch.Tensor): torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, np.ndarray): np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a: equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal(x, y)
    else: assert a == b
verify()
commands = [
    ['-m', 'unittest', 'tests.test_sigmoid_glu', 'tests.test_state_clock', 'tests.test_state_clock_integration',
     'tests.test_forced_gate', 'tests.test_forced_gate_integration', 'tests.test_diagonal_flow',
     'tests.test_diagonal_integration', 'tests.test_shared_features', 'tests.test_controller_tangent',
     'tests.test_common_backbone', '-v'],
    ['scripts/check_rate_order_reference.py', '--reference-file', str(prior/'source/synthetic/models.py')]]
test_count = None
for i, args in enumerate(commands):
    result = subprocess.run([sys.executable, *args], cwd=source, capture_output=True, text=True)
    text = result.stdout+result.stderr
    (out/f'check-{i}.log').write_text(text)
    assert result.returncode == 0, (i, text)
    if i == 0: test_count = int(re.search(r'Ran (\d+) tests?', text).group(1))

from synthetic.models import VelocityNet
from synthetic.initialization import copy_common_backbone
from synthetic.train import run
from scripts.run_clock_study import monitor
import scripts.run_architecture_study as runner
from torch.nn import functional as F

def actual_macs(model, dim):
    count, linear, parameter = [0], F.linear, next(model.parameters())
    def instrument(x, weight, bias=None):
        count[0] += (x.numel()//x.shape[-1])*weight.numel()
        return linear(x, weight, bias)
    with patch.object(F, 'linear', instrument), torch.no_grad():
        model(torch.zeros(1, dim, dtype=parameter.dtype), torch.zeros(1, 1, dtype=parameter.dtype))
    return count[0]

template = json.loads((source/'configs/synthetic/gates_20261021/ode_original.json').read_text())
labels = ['swiglu_budget', 'swiglu', 'sigmoid_glu']
initial, references, configs = [], {}, {}
for suffix in ('v1', 'product16'):
    path = source/f'configs/synthetic/architecture_search/sigmoid_glu_{suffix}_20261025_m1.json'
    plan = json.loads(path.read_text())
    assert [a['label'] for a in plan['arms']] == labels
    assert (plan['seed'], plan['steps'], plan['device'], plan['workers']) == (0, 5000, 'cpu', 2)
    for dataset in plan['datasets']:
        dim = 16 if dataset.startswith('rotated') else 2
        for arm in plan['arms']:
            ref = arm['backbone_initialization'][dataset]
            assert sha(ref['checkpoint']) == ref['sha256']
            references[dataset] = ref
            saved = torch.load(ref['checkpoint'], weights_only=False, map_location='cpu')
            options = dict(template['model_options']); options.update(arm['options'])
            torch.manual_seed(0); model = VelocityNet(dim, arm['model'], **options)
            native = {k: v.clone() for k, v in model.state_dict().items()}
            rng = torch.get_rng_state().clone()
            receipt = copy_common_backbone(model, Path(ref['checkpoint']), ref['sha256'])
            equal(rng, torch.get_rng_state())
            for name, value in model.state_dict().items(): equal(value, native[name] if '.ffn.' in name else saved['model'][name])
            equal(dict(model.named_buffers()), saved['runtime_buffers'])
            assert list(model.ode_layers()) == [] and list(model.state_clock_layers()) == []
            macs = actual_macs(model, dim)
            assert macs == runner.macs(model, dim)
            if dim == 16 and arm['label'] in ('swiglu', 'sigmoid_glu'):
                assert (model.parameter_count(), macs) == (83598, 81792)
            initial.append(dict(dataset=dataset, label=arm['label'], parameters=model.parameter_count(),
                                forward_macs=macs, native_ffn_and_rng_unchanged=True, **receipt))
    data = Path('/Users/mr.z/ode-workspace-prep/data/synthetic')/('v1' if suffix == 'v1' else 'product16_v1')
    try:
        with patch.object(runner.subprocess, 'run', side_effect=RuntimeError('PREFLIGHT_WORKER_DISABLED')):
            runner.run(path, data, out/f'runner-{suffix}')
    except RuntimeError as error: assert str(error) == 'PREFLIGHT_WORKER_DISABLED'
    else: raise AssertionError('worker interception not reached')
    for arm in plan['arms']:
        cfg = json.loads((out/f'runner-{suffix}/configs'/f"{arm['label']}.json").read_text())
        configs[suffix, arm['label']] = cfg
        assert cfg['backbone_initialization'] == arm['backbone_initialization']
        assert cfg['models'] == [arm['model']] and cfg['seeds'] == [0] and cfg['steps'] == 5000
        for key, value in arm['options'].items(): assert cfg['model_options'][key] == value
    assert not (out/f'runner-{suffix}/runs').exists()
assert len(initial) == 9
write(out/'all-arm-initialization.json', dict(state='PASS', arms=initial, references=references))

# The inherited constructor gives identical complete tensors/RNG, not the same
# hidden function. The formal zero head masks values but not its own gradients.
paired = []
for dtype in (torch.float32, torch.float64):
    for dim, dataset, suffix in ((2, 'gmm_ring8_2d', 'v1'), (16, 'rotated_product_bimodal_16d', 'product16')):
        ref = references[dataset]
        generator = torch.Generator().manual_seed(0)
        x = torch.randn(19, dim, generator=generator, dtype=dtype)
        t = torch.rand(19, 1, generator=generator, dtype=dtype)
        target = torch.randn(19, dim, generator=generator, dtype=dtype)
        head = torch.randn(dim, 64, generator=generator, dtype=dtype)*.2
        models, rngs, features = [], [], {}
        for label in ('swiglu', 'sigmoid_glu'):
            cfg = configs[suffix, label]
            torch.manual_seed(0); model = VelocityNet(dim, cfg['models'][0], **cfg['model_options']).to(dtype)
            rngs.append(torch.get_rng_state().clone())
            copy_common_backbone(model, Path(ref['checkpoint']), ref['sha256'])
            equal(rngs[-1], torch.get_rng_state())
            def capture(module, args, result, label=label): features[label] = result.detach().clone()
            hook = model.norm.register_forward_hook(capture)
            value = model(x, t)
            assert torch.count_nonzero(value) == 0
            (value-target).square().sum().backward(); hook.remove()
            models.append(model)
        old, new = models
        equal(old.state_dict(), new.state_dict()); equal(dict(old.named_buffers()), dict(new.named_buffers())); equal(*rngs)
        feature_difference = float((features['swiglu']-features['sigmoid_glu']).abs().max())
        head_gradient_difference = float((old.output.weight.grad-new.output.weight.grad).abs().max())
        assert feature_difference > 1e-6 and head_gradient_difference > 1e-6
        for model in models:
            with torch.no_grad(): model.output.weight.copy_(head)
        function_difference = float((old(x, t)-new(x, t)).abs().max().detach())
        assert function_difference > 1e-6
        assert actual_macs(old, dim) == actual_macs(new, dim)
        paired.append(dict(dtype=str(dtype), dim=dim, full_native_tensor_buffer_RNG_equal=True,
            initial_formal_output_zero=True, hidden_feature_max_difference=feature_difference,
            zero_head_output_weight_gradient_max_difference=head_gradient_difference,
            nonzero_head_function_max_difference=function_difference,
            parameters=new.parameter_count(), forward_macs=actual_macs(new, dim)))
write(out/'full-width-initialization.json', dict(state='PASS', rows=paired,
    interpretation='Identical native tensors and RNG; functions and head tangents differ. No initial NTK equivalence is claimed.'))

# The default8 reference checker above covers the old matrix family; also keep
# all3 prior forced-gate and all3 prior state-clock training trajectories exact.
spec = importlib.util.spec_from_file_location('stage21_frozen_models', prior/'source/synthetic/models.py')
old_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(old_module)
historical = []
cases = [('ode_forced_gate', dict(forced_gate_solver=solver, forced_gate_clock=clock))
         for solver, clock in (('exact', 'static'), ('k2', 'adaptive'), ('exact', 'adaptive'))]
cases += [('ode_state_clock', dict(state_clock_solver=solver)) for solver in ('euler', 'frozen', 'midpoint')]
for kind, opts in cases:
    torch.manual_seed(0); old = old_module.VelocityNet(16, kind, **opts); rng = torch.get_rng_state().clone()
    torch.manual_seed(0); new = VelocityNet(16, kind, **opts); equal(rng, torch.get_rng_state())
    generator = torch.Generator().manual_seed(100000)
    with torch.no_grad():
        head = torch.randn(new.output.weight.shape, generator=generator)
        old.output.weight.copy_(head); new.output.weight.copy_(head)
    optimizers = [torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.) for m in (old, new)]
    for _ in range(3):
        x = torch.randn(31, 16, generator=generator); t = torch.rand(31, 1, generator=generator)
        target = torch.randn(31, 16, generator=generator)
        equal(old(x, t), new(x, t))
        for model, optimizer in zip((old, new), optimizers):
            optimizer.zero_grad(); (model(x, t)-target).square().mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
        equal(old.state_dict(), new.state_dict()); equal(optimizers[0].state_dict(), optimizers[1].state_dict())
    historical.append(dict(kind=kind, options=opts, steps=3, state='BIT_IDENTICAL'))
write(out/'historical-model-trajectories.json', dict(state='PASS', historical_source=str(prior/'source'), rows=historical))

heartbeat = out/'monitor-heartbeat'; heartbeat.mkdir(); stop = threading.Event()
watcher = threading.Thread(target=monitor, args=(heartbeat, stop, dict(state='PREFLIGHT', phase='monitor-check', completed=0, active=[])), daemon=True)
watcher.start()
for _ in range(50):
    if (heartbeat/'monitor.json').exists(): break
    time.sleep(.1)
assert watcher.is_alive() and (heartbeat/'telemetry.jsonl').is_file()
stop.set(); watcher.join(timeout=2); assert not watcher.is_alive()
assert json.loads((heartbeat/'monitor.json').read_text())['pid'] == os.getpid()

def smoke_config(label):
    cfg = dict(configs['product16', label])
    cfg.update(experiment_id=label+'_smoke', steps=3, batch_size=16, eval_interval=1,
               sample_interval=3, checkpoint_interval=1, sample_count=32, sliced_directions=4, heun_steps=4)
    config = out/(label+'-smoke.json'); write(config, cfg)
    return cfg, config

cfg, config = smoke_config('sigmoid_glu')
def options(folder, steps=None, resume=None):
    return Namespace(config=str(config), dataset='rotated_product_bimodal_16d', model='glu', seed=0,
                     device='cpu', output_root=str(out/'glu'/folder), steps=steps, dry_run=False, resume=resume)
def checkpoint(folder, step):
    return out/'glu'/folder/cfg['experiment_id']/'rotated_product_bimodal_16d/glu/seed000/checkpoints'/f'step_{step:07d}.pt'
run(options('uninterrupted')); run(options('resumed', 2)); run(options('resumed', 3, str(checkpoint('resumed', 2))))
a = torch.load(checkpoint('uninterrupted', 3), weights_only=False, map_location='cpu')
b = torch.load(checkpoint('resumed', 3), weights_only=False, map_location='cpu')
state_keys = ('model', 'optimizer', 'runtime_buffers', 'data_rng', 'torch_rng', 'numpy_rng', 'python_rng')
for key in state_keys: equal(a[key], b[key])
metrics = [json.loads(line) for line in (checkpoint('uninterrupted', 0).parent.parent/'metrics.jsonl').read_text().splitlines()]
assert metrics[-1]['sample']['heun_steps'] == 4 and metrics[-1]['sample']['nfe'] == 8
assert np.isfinite(metrics[-1]['sample']['sliced_w2'])
assert all(row['ode_layers'] == [] for row in metrics)
first = torch.load(checkpoint('uninterrupted', 0), weights_only=False, map_location='cpu')
ffn_change = max(float((value-first['model'][key]).abs().max()) for key, value in a['model'].items() if '.ffn.' in key)
assert ffn_change > 0
write(out/'glu-trainer-resume.json', dict(state='PASS', resume_bit_identical=True, learned_ffn_max_change=ffn_change,
    sampling_checkpoints_valid=True, no_ODE_logs=True,
    checkpoint_sha256={folder: sha(checkpoint(folder, 3)) for folder in ('uninterrupted', 'resumed')}))

baseline_proofs = []
for label in ('swiglu_budget', 'swiglu'):
    cfg, config = smoke_config(label)
    for version, src in (('old', prior/'source'), ('new', source)):
        command = [sys.executable, '-m', 'synthetic.train', '--config', str(config), '--dataset', 'rotated_product_bimodal_16d',
                   '--model', 'swiglu', '--seed', '0', '--device', 'cpu', '--output-root', str(out/(label+'-'+version))]
        result = subprocess.run(command, cwd=src, capture_output=True, text=True)
        (out/(label+'-'+version+'.log')).write_text(result.stdout+result.stderr)
        assert result.returncode == 0, result.stderr
    subpath = Path(cfg['experiment_id'])/'rotated_product_bimodal_16d/swiglu/seed000/checkpoints/step_0000003.pt'
    a = torch.load(out/(label+'-old')/subpath, weights_only=False, map_location='cpu')
    b = torch.load(out/(label+'-new')/subpath, weights_only=False, map_location='cpu')
    for key in state_keys: equal(a[key], b[key])
    baseline_proofs.append(dict(label=label, state='PASS', fulltrainer_bit_identical=True, steps=3))
write(out/'historical-baseline-trainer.json', dict(state='PASS', arms=baseline_proofs))

verify()
for ref in references.values(): assert sha(ref['checkpoint']) == ref['sha256']
assert list((root/'output').iterdir()) == []
dependency = dict(state='PASS', python=sys.version, platform=platform.platform(),
                  torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__, psutil=psutil.__version__,
                  requirements_sha256=sha(source/'requirements-synthetic.txt'),
                  actual_psutil_heartbeat=True, memory_available_gib=psutil.virtual_memory().available/2**30)
write(out/'dependency-check.json', dependency)
summary = dict(state='PASS', relevant_tests=test_count, full_width_initialization_cases=len(paired),
    common_backbone_initializations=len(initial), frozen_default_models=8, default_updates_per_model=3,
    default_trajectories='BIT_IDENTICAL', prior_forced_models=3, prior_state_clock_models=3,
    all_six_prior_model_3update_trajectories='BIT_IDENTICAL', glu_fulltrainer_resume='BIT_IDENTICAL',
    no_ODE_logs=True, sampling_checkpoints_valid=True, both_old_SwiGLU_fulltrainers='BIT_IDENTICAL',
    actual_psutil_heartbeat=True, all_runner_workers_intercepted=True, source_files_verified=len(manifest['all_files']),
    prior_source_unchanged=True, reference_hashes_unchanged=True, historical_source=str(prior/'source'),
    historical_source_files=len(oldmanifest['all_files']), explicit_predecessor_and_manifest_verified=True,
    formal_cells=9, formal_training_started=False, formal_outputs_absent=True, preflight_script_sha256=sha(__file__))
write(out/'summary.json', summary)
for filename in ('summary.json', 'full-width-initialization.json', 'all-arm-initialization.json',
                 'glu-trainer-resume.json', 'historical-baseline-trainer.json',
                 'historical-model-trajectories.json', 'dependency-check.json'):
    shutil.copy2(out/filename, root/'preflight'/filename)
launch = [sys.executable, str(source/'scripts/run_architecture_study.py'), '--plan',
          str(source/'configs/synthetic/architecture_search/sigmoid_glu_queue_20261025_m1.json'),
          '--data-root', '/Users/mr.z/ode-workspace-prep/data/synthetic/v1',
          '--output', str(root/'output/sigmoid_glu_queue_20261025_m1')]
# Expose ready LAST; never mutate this receipt after parent launch.
write(root/'ready.json', dict(state='PREFLIGHT_PASS_AWAITING_PARENT_REVIEW', formal_training_started=False,
    formal_cells=9, runtime=str(source), launch_command=launch, cwd=str(source),
    environment=dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES=''),
    preflight_summary_sha256=sha(out/'summary.json'), source_manifest_sha256=sha(root/'source-manifest.json'),
    source_delta_sha256=sha(root/'source-delta.patch'), dependency_check_sha256=sha(out/'dependency-check.json')))
print(json.dumps(summary))
