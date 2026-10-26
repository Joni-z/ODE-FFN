"""Stage24 harness structure only; no frozen source or execution authorization yet.

Root must hand off stable protocol, source/config/test hashes and explicitly
authorize freezing/preflight before PREPARATION_ONLY is removed. This file
does not launch formal workers. No current receipt claims a check has run.
"""
from argparse import Namespace
from pathlib import Path
import hashlib
import csv
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from unittest.mock import patch

import numpy as np
import psutil
import scipy
import torch
from torch.nn import functional as F

PREPARATION_ONLY = False
ROOT = Path(__file__).resolve().parents[1]
PRIOR = Path('/Users/mr.z/ode-workspace-prep/architecture-search/affine_forcing_20261025_m1')
PARENT_MANIFEST_SHA = '47e833e60e63c5fffb820ce5939f15ee9f595ba5c5fdc6e5a906ea39fa0dbb5c'
STATE_KEYS = ('model', 'optimizer', 'runtime_buffers', 'data_rng', 'torch_rng', 'numpy_rng', 'python_rng')
DEFAULT_REFERENCE_EXECUTIONS = 8


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def verify_distinct_ancestors(source_manifest):
    """The real Stage23 parent is explicit; never derive it by text replacement."""
    assert ROOT.name == 'augmented_state_20261025_m1'
    assert PRIOR.name == 'affine_forcing_20261025_m1' and PRIOR != ROOT
    assert (ROOT/'source').resolve() != (PRIOR/'source').resolve()
    assert sha(PRIOR/'source-manifest.json') == PARENT_MANIFEST_SHA
    parent = json.loads((PRIOR/'source-manifest.json').read_text())
    assert len(parent['all_files']) == 595
    assert source_manifest['parent_source'] == str(PRIOR/'source')
    assert source_manifest['parent_manifest_sha256'] == PARENT_MANIFEST_SHA
    for directory, manifest in ((PRIOR/'source', parent), (ROOT/'source', source_manifest)):
        paths = {str(p.relative_to(directory)) for p in directory.rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts}
        assert paths == set(manifest['all_files'])
        for name, digest in manifest['all_files'].items():
            assert sha(directory/name) == digest, name
    return parent


def legacy_extra_cases():
    cases = [('ode_forced_gate', dict(forced_gate_solver=solver, forced_gate_clock=clock))
             for solver, clock in (('exact', 'static'), ('k2', 'adaptive'), ('exact', 'adaptive'))]
    cases += [('ode_state_clock', dict(state_clock_solver=solver)) for solver in ('euler', 'frozen', 'midpoint')]
    cases += [('glu', dict(ffn_hidden=128)),
              ('ode_vector', dict(gate_hidden=32, gate_reference_hidden=64,
                                  orders=1, ode_solve_init='constant_paired'))]
    cases += [('ode_vector', dict(gate_hidden=32, gate_reference_hidden=64, orders=1,
                                  ode_solve_init='constant_paired', ode_forcing_mode=mode))
              for mode in ('static', 'conditional')]
    assert len(cases) == 10
    return cases


def legacy_trajectories(VelocityNet, destination):
    """Additional ten executions; the separate native checker supplies eight.

    There are 18 executions but 17 unique effective settings: the explicit
    constant-paired vector map repeats one native default-reference case.
    """
    spec = importlib.util.spec_from_file_location('stage23_frozen_models', PRIOR/'source/synthetic/models.py')
    old_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_module)
    rows = []
    for kind, options in legacy_extra_cases():
        torch.manual_seed(0)
        old = old_module.VelocityNet(16, kind, **options)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(0)
        new = VelocityNet(16, kind, **options)
        equal(rng, torch.get_rng_state())
        generator = torch.Generator().manual_seed(100000)
        with torch.no_grad():
            head = torch.randn(new.output.weight.shape, generator=generator)
            old.output.weight.copy_(head)
            new.output.weight.copy_(head)
        optimizers = [torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.)
                      for model in (old, new)]
        for _ in range(3):
            x = torch.randn(31, 16, generator=generator)
            t = torch.rand(31, 1, generator=generator)
            target = torch.randn(31, 16, generator=generator)
            equal(old(x, t), new(x, t))
            for model, optimizer in zip((old, new), optimizers):
                optimizer.zero_grad()
                (model(x, t)-target).square().mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            equal(old.state_dict(), new.state_dict())
            equal(dict(old.named_buffers()), dict(new.named_buffers()))
            equal(optimizers[0].state_dict(), optimizers[1].state_dict())
        rows.append(dict(kind=kind, options=options, steps=3, state='BIT_IDENTICAL'))
    write(destination, dict(state='PASS', historical_source=str(PRIOR/'source'),
                            extra_execution_count=10, default_reference_execution_count=8,
                            total_execution_count=18, distinct_effective_settings=17, rows=rows))


def actual_macs(model, dim):
    count, linear = [0], F.linear
    parameter = next(model.parameters())
    def instrument(x, weight, bias=None):
        count[0] += (x.numel()//x.shape[-1])*weight.numel()
        return linear(x, weight, bias)
    with patch.object(F, 'linear', instrument), torch.no_grad():
        model(torch.zeros(1, dim, dtype=parameter.dtype),
              torch.zeros(1, 1, dtype=parameter.dtype))
    return count[0]


def smoke_config(config, label, destination):
    cfg = dict(config)
    cfg.update(experiment_id=label+'_smoke', steps=3, batch_size=16, eval_interval=1,
               sample_interval=3, checkpoint_interval=1, sample_count=32,
               sliced_directions=4, heun_steps=4)
    path = destination/(label+'-smoke.json')
    write(path, cfg)
    return cfg, path


def resumed_trainer(run, config, label, destination):
    """Actual three updates versus two plus resume; all saved RNG/buffers paired."""
    cfg, path = smoke_config(config, label, destination)
    kind = cfg['models'][0]
    def options(folder, steps=None, resume=None):
        return Namespace(config=str(path), dataset='rotated_product_bimodal_16d', model=kind,
                         seed=0, device='cpu', output_root=str(destination/label/folder),
                         steps=steps, dry_run=False, resume=resume)
    def checkpoint(folder, step):
        return destination/label/folder/cfg['experiment_id']/f'rotated_product_bimodal_16d/{kind}/seed000/checkpoints'/f'step_{step:07d}.pt'
    run(options('uninterrupted'))
    run(options('resumed', 2))
    run(options('resumed', 3, str(checkpoint('resumed', 2))))
    a = torch.load(checkpoint('uninterrupted', 3), weights_only=False, map_location='cpu')
    b = torch.load(checkpoint('resumed', 3), weights_only=False, map_location='cpu')
    for key in STATE_KEYS:
        equal(a[key], b[key])
    metrics = [json.loads(line) for line in (checkpoint('uninterrupted', 0).parent.parent/'metrics.jsonl').read_text().splitlines()]
    assert metrics[-1]['sample']['heun_steps'] == 4 and metrics[-1]['sample']['nfe'] == 8
    assert np.isfinite(metrics[-1]['sample']['sliced_w2'])
    return cfg, a, metrics, {folder: sha(checkpoint(folder, 3)) for folder in ('uninterrupted', 'resumed')}


def real_monitor_heartbeat(monitor, directory):
    directory.mkdir()
    stop = threading.Event()
    thread = threading.Thread(target=monitor, args=(directory, stop, dict(
        state='PREFLIGHT', phase='monitor-check', completed=0, active=[])), daemon=True)
    thread.start()
    for _ in range(50):
        if (directory/'monitor.json').exists():
            break
        time.sleep(.1)
    assert thread.is_alive() and (directory/'telemetry.jsonl').is_file()
    stop.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert json.loads((directory/'monitor.json').read_text())['pid'] == os.getpid()


def main():
    if PREPARATION_ONLY:
        raise SystemExit('PREPARATION ONLY: await explicit root handoff and final reviewed source/config/test hashes. No preflight has run.')
    source, out = ROOT/'source', ROOT/'preflight/attempt3'
    assert not (ROOT/'ready.json').exists()
    assert not list((ROOT/'output').iterdir())
    manifest = json.loads((ROOT/'source-manifest.json').read_text())
    verify_distinct_ancestors(manifest)
    assert len(manifest['all_files']) == 604 and len(manifest['overlays']) == 13
    review = source/'docs/research-architecture-search/augmented-state-code-review.md'
    assert sha(review) == 'c4d75cb539f0208bdf88c1427b9014433d7282d782ff6ac9ca040a2288d0173f'
    out.mkdir()
    print(json.dumps(dict(phase='START', pid=os.getpid(), source=str(source), parent=str(PRIOR/'source'))), flush=True)
    sys.path.insert(0, str(source)); torch.set_num_threads(1)
    commands = [
        ['-m','unittest','tests.test_augmented_state','tests.test_affine_forcing','tests.test_sigmoid_glu',
         'tests.test_state_clock','tests.test_state_clock_integration','tests.test_forced_gate',
         'tests.test_forced_gate_integration','tests.test_diagonal_flow','tests.test_diagonal_integration',
         'tests.test_shared_features','tests.test_controller_tangent','tests.test_common_backbone','-v'],
        ['scripts/check_rate_order_reference.py','--reference-file',str(PRIOR/'source/synthetic/models.py')]]
    for i,args in enumerate(commands):
        result = subprocess.run([sys.executable,*args],cwd=source,capture_output=True,text=True)
        log = result.stdout+result.stderr; (out/f'check-{i}.log').write_text(log)
        assert result.returncode == 0,(i,log)
        if i == 0:
            test_count = int(re.search(r'Ran (\d+) tests?',log).group(1)); assert test_count == 72
    from synthetic.models import VelocityNet
    from synthetic.initialization import copy_common_backbone
    from synthetic.train import run, restore_model
    from synthetic.diagnostics import collect, spectrum
    from scripts.run_clock_study import monitor
    import scripts.run_architecture_study as runner
    for module in ('synthetic.models','synthetic.train','synthetic.augmented','synthetic.diagnostics','scripts.run_architecture_study'):
        assert Path(sys.modules[module].__file__).resolve().is_relative_to(source)
    labels = ['vector_parent','lowrank_budget','lowrank_narrow','lowrank_augmented','swiglu','silu128']
    lowlabels = labels[1:4]
    budgets = {'vector_parent':(83664,81984),'lowrank_budget':(83664,81984),
               'lowrank_narrow':(66702,65088),'lowrank_augmented':(83022,81216),
               'swiglu':(83598,81792),'silu128':(83664,81984)}
    template = json.loads((source/'configs/synthetic/gates_20261021/ode_original.json').read_text())
    initial, references, configs = [], {}, {}
    for suffix in ('v1','product16'):
        plan_path = source/f'configs/synthetic/architecture_search/augmented_state_{suffix}_20261025_m1.json'
        plan = json.loads(plan_path.read_text())
        assert [a['label'] for a in plan['arms']] == labels
        assert (plan['seed'],plan['steps'],plan['device'],plan['workers']) == (0,5000,'cpu',2)
        for dataset in plan['datasets']:
            dim = 16 if dataset.startswith('rotated') else 2
            prior_ode_rng = None
            for arm in plan['arms']:
                ref = arm['backbone_initialization'][dataset]
                assert sha(ref['checkpoint']) == ref['sha256']; references[dataset] = ref
                saved = torch.load(ref['checkpoint'],weights_only=False,map_location='cpu')
                options = dict(template['model_options']); options.update(arm['options'])
                torch.manual_seed(0); model = VelocityNet(dim,arm['model'],**options)
                native = {k:v.clone() for k,v in model.state_dict().items()}; rng = torch.get_rng_state().clone()
                if arm['label'] == 'vector_parent': prior_ode_rng = rng
                if arm['label'] in lowlabels: equal(prior_ode_rng,rng)
                receipt = copy_common_backbone(model,Path(ref['checkpoint']),ref['sha256'])
                equal(rng,torch.get_rng_state())
                for name,value in model.state_dict().items(): equal(value,native[name] if '.ffn.' in name else saved['model'][name])
                equal(dict(model.named_buffers()),saved['runtime_buffers'])
                assert not list(model.state_clock_layers())
                if arm['label'] in lowlabels:
                    assert len(list(model.lowrank_layers())) == 6 and not list(model.ode_layers())
                    for _,layer in model.lowrank_layers():
                        assert layer.rank == options['lowrank_rank'] and layer.gate_hidden == options['lowrank_gate_hidden']
                        assert layer.state_width == options['lowrank_state_width']
                        assert torch.count_nonzero(layer.t_func[-1].weight) == torch.count_nonzero(layer.t_func[-1].bias) == 0
                elif arm['label'] == 'vector_parent': assert len(list(model.ode_layers())) == 6
                else: assert not list(model.ode_layers()) and not list(model.lowrank_layers())
                macs = actual_macs(model,dim); assert macs == runner.macs(model,dim)
                p,m = budgets[arm['label']]
                if dim == 2: p -= 1806; m -= 1792
                assert (model.parameter_count(),macs) == (p,m)
                initial.append(dict(dataset=dataset,label=arm['label'],parameters=p,forward_macs=m,native_ffn_and_rng_unchanged=True,**receipt))
        data = Path('/Users/mr.z/ode-workspace-prep/data/synthetic')/('v1' if suffix=='v1' else 'product16_v1')
        try:
            with patch.object(runner.subprocess,'run',side_effect=RuntimeError('PREFLIGHT_WORKER_DISABLED')):
                runner.run(plan_path,data,out/f'runner-{suffix}')
        except RuntimeError as error: assert str(error) == 'PREFLIGHT_WORKER_DISABLED'
        else: raise AssertionError('worker interception not reached')
        for arm in plan['arms']:
            cfg = json.loads((out/f'runner-{suffix}/configs'/f"{arm['label']}.json").read_text())
            configs[suffix,arm['label']] = cfg
            assert cfg['backbone_initialization'] == arm['backbone_initialization']
            assert cfg['models'] == [arm['model']] and cfg['seeds'] == [0] and cfg['steps'] == 5000
            for key,value in arm['options'].items(): assert cfg['model_options'][key] == value
        assert not (out/f'runner-{suffix}/runs').exists()
    assert len(initial) == 18
    write(out/'all-arm-initialization.json',dict(state='PASS',arms=initial,references=references))

    def build(label,dim,suffix,dtype=torch.float32):
        cfg = configs[suffix,label]; torch.manual_seed(0)
        model = VelocityNet(dim,cfg['models'][0],**cfg['model_options']).to(dtype)
        dataset = 'rotated_product_bimodal_16d' if dim==16 else 'gmm_ring8_2d'
        ref = references[dataset]; before = torch.get_rng_state().clone()
        copy_common_backbone(model,Path(ref['checkpoint']),ref['sha256'])
        equal(before,torch.get_rng_state())
        return model,before

    def core(name,value):
        if any(name.endswith(s) for s in ('.ode1.left','.ode1.t_func.2.weight','.ode1.t_func.2.bias')): return value[:64]
        if any(name.endswith(s) for s in ('.ode2.right','.ode2.t_func.0.weight')): return value[...,:64]
        return value

    paired=[]
    for dtype,atol,rtol in ((torch.float32,3e-6,5e-5),(torch.float64,3e-13,3e-12)):
        for dim,suffix in ((2,'v1'),(16,'product16')):
            models={}; rngs=[]
            for label in labels[:4]:
                model,rng=build(label,dim,suffix,dtype);models[label]=model;rngs.append(rng)
            for rng in rngs[1:]: equal(rng,rngs[0])
            parent=models['vector_parent'];narrow=models['lowrank_narrow'];expanded=models['lowrank_augmented']
            for model in models.values():
                for name,value in parent.state_dict().items():
                    if '.ffn.' not in name: equal(model.state_dict()[name],value)
                equal(dict(model.named_buffers()),dict(parent.named_buffers()))
            for name,value in narrow.state_dict().items(): equal(core(name,expanded.state_dict()[name]),value)
            truncation=[]
            for label in lowlabels:
                for (name,layer),(_,reference) in zip(models[label].lowrank_layers(),parent.ode_layers()):
                    u,s,vh=torch.linalg.svd(reference.proj.weight.detach().cpu().double(),full_matrices=False)
                    q=layer.rank;target=(u[:,:q]*s[:q])@vh[:q]
                    product=layer.left[:64].double()@layer.right[:,:64].double()
                    torch.testing.assert_close(product,target,atol=2e-7,rtol=2e-5)
                    equal(layer.t_func[0].weight[:,:64],reference.t_func[0].weight[:layer.gate_hidden])
                    equal(layer.t_func[0].bias,reference.t_func[0].bias[:layer.gate_hidden])
                    if layer.role=='lift' and layer.state_width==128: equal(layer.left[64:],layer.left[:64])
                    if layer.role=='project' and layer.state_width==128:
                        assert torch.count_nonzero(layer.right[:,64:])==0 and torch.count_nonzero(layer.t_func[0].weight[:,64:])==0
                    truncation.append(dict(label=label,layer=name,rank=q,maximum_product_residual=float((product-target).detach().abs().max())))
            generator=torch.Generator().manual_seed(0)
            x=torch.randn(1,dim,generator=generator,dtype=dtype);t=torch.rand(1,1,generator=generator,dtype=dtype)
            paramaps=[dict(m.named_parameters()) for m in (narrow,expanded)]
            predictions=[m(x,t) for m in (narrow,expanded)];gradient_rows=[[],[]]
            for y in predictions: assert torch.count_nonzero(y)==0
            for index in range(dim):
                for j,(p,y) in enumerate(zip(paramaps,predictions)):
                    grad=dict(zip(p,torch.autograd.grad(y.flatten()[index],tuple(p.values()),retain_graph=True)))
                    for name,value in grad.items():
                        if not name.startswith('output.'): assert torch.count_nonzero(value)==0
                    gradient_rows[j].append(torch.cat([value.flatten() for value in grad.values()]))
            kernels=[torch.stack(rows)@torch.stack(rows).T for rows in gradient_rows]
            torch.testing.assert_close(kernels[0],kernels[1],atol=atol,rtol=rtol)
            head=torch.randn(dim,64,generator=generator,dtype=dtype)*.2
            with torch.no_grad():
                narrow.output.weight.copy_(head);expanded.output.weight.copy_(head)
            predictions=[m(x,t) for m in (narrow,expanded)]
            torch.testing.assert_close(predictions[0],predictions[1],atol=atol,rtol=rtol)
            jacobians=[torch.autograd.functional.jacobian(lambda z:m(z,t),x) for m in (narrow,expanded)]
            torch.testing.assert_close(jacobians[0],jacobians[1],atol=atol,rtol=rtol)
            grads=[dict(zip(p,torch.autograd.grad(y.flatten()[0],tuple(p.values())))) for p,y in zip(paramaps,predictions)]
            for name,value in grads[0].items(): torch.testing.assert_close(value,core(name,grads[1][name]),atol=atol,rtol=rtol)
            live=sum(float(g[:,64:].double().square().sum()) for n,g in grads[1].items() if n.endswith('.ode2.right'))
            assert live>1e-9
            response_difference=sum(float(g.double().square().sum()) for g in grads[1].values())-sum(float(g.double().square().sum()) for g in grads[0].values())
            assert response_difference>0
            if dtype==torch.float64:
                np.testing.assert_allclose(response_difference,live,atol=2e-11,rtol=2e-5)
            paired.append(dict(dtype=str(dtype),dim=dim,common_tensors_and_RNG_exact=True,
                native_zero_head_kernel_paired=True,nonzero_head_functions_and_input_Jacobians_paired_with_tolerance=True,
                nonzero_head_extra_receiver_response_squared=live,nonzero_head_full_response_difference=response_difference,
                nonzero_head_full_NTK_not_paired=True,forward_difference=float((predictions[0]-predictions[1]).detach().abs().max()),
                response_energy_identity_checked_in_float64_only=(dtype==torch.float64),
                input_Jacobian_difference=float((jacobians[0]-jacobians[1]).abs().max()),atol=atol,rtol=rtol,reference_SVD=truncation))
    write(out/'full-width-initialization.json',dict(state='PASS',rows=paired,
        scope='Original random P supplies truncated core, not teacher weights. Narrow/expanded functions pair within GEMM tolerance; nonzero-head parameter tangents differ.'))

    calls=[]
    for label in lowlabels:
        model,_=build(label,16,'product16');layers=dict(model.lowrank_layers());pointers={}
        counts={name:dict(left=0,right=0,controller=0) for name in layers};handles=[]
        for name,layer in layers.items():
            pointers[layer.left.data_ptr()]=(name,'left');pointers[layer.right.data_ptr()]=(name,'right')
            def hook(module,args,value,name=name): counts[name]['controller']+=1
            handles.append(layer.t_func.register_forward_hook(hook))
        linear=F.linear;mac=[0]
        def counted(x,weight,bias=None):
            mac[0]+=(x.numel()//x.shape[-1])*weight.numel()
            if weight.data_ptr() in pointers:
                name,role=pointers[weight.data_ptr()];counts[name][role]+=1
            return linear(x,weight,bias)
        with patch.object(F,'linear',counted),torch.no_grad(): model(torch.zeros(1,16),torch.zeros(1,1))
        for h in handles: h.remove()
        assert all(c==dict(left=1,right=1,controller=1) for c in counts.values())
        assert mac[0]==runner.macs(model,16)==budgets[label][1]
        calls.append(dict(label=label,calls=counts,actual_dense_macs=mac[0],parameters=model.parameter_count()))
    write(out/'executed-call-counts.json',dict(state='PASS',rows=calls))
    legacy_trajectories(VelocityNet,out/'historical-model-trajectories.json')
    real_monitor_heartbeat(monitor,out/'monitor-heartbeat')
    print(json.dumps(dict(phase='INITIALIZATION_AND_HISTORY_PASS',pid=os.getpid())),flush=True)

    smokes=[]
    for label in lowlabels:
        cfg,cp,metrics,digests=resumed_trainer(run,configs['product16',label],label,out)
        cp0=torch.load(out/label/'uninterrupted'/cfg['experiment_id']/'rotated_product_bimodal_16d/ode_lowrank/seed000/checkpoints/step_0000000.pt',weights_only=False,map_location='cpu')
        controllers=[v for k,v in cp['model'].items() if k.endswith('.t_func.2.weight')]
        assert len(controllers)==6 and all(v.norm()>0 for v in controllers)
        changes={k:float((v-cp0['model'][k]).norm()) for k,v in cp['model'].items() if k.endswith(('.left','.right'))}
        assert len(changes)==12 and all(v>0 for v in changes.values())
        extra=[]
        if label=='lowrank_augmented':
            for i in range(3):
                prefix=f'blocks.{i}.ffn.'
                keys=[('ode1.left',slice(64,None),slice(None)),('ode2.right',slice(None),slice(64,None)),
                      ('ode1.t_func.2.weight',slice(64,None),slice(None)),('ode2.t_func.0.weight',slice(None),slice(64,None))]
                for key,a,b in keys:
                    name=prefix+key; delta=float((cp['model'][name][a,b]-cp0['model'][name][a,b]).norm())
                    assert delta>0,(name,delta);extra.append(dict(name=name,changed_norm=delta))
        for index in (0,-1):
            assert len(metrics[index]['ode_layers'])==6
            json.dumps(metrics[index]['ode_layers'],allow_nan=False)
            for record in metrics[index]['ode_layers']:
                assert record['operator_type']=='pruned_lowrank_K1'
                s=record['activation'];assert .2<=s['s_min']<=s['s_max']<=1.
                assert 'compressed_at_probe_mean_duration' in record and 'probe-mean' in record['interpretation']
        probe=VelocityNet(16,'ode_lowrank',**cfg['model_options']);restore_model(probe,cp)
        generator=torch.Generator().manual_seed(0);x=torch.randn(11,16,generator=generator);t=torch.rand(11,1,generator=generator)
        inputs={};hooks=[]
        for name,layer in probe.lowrank_layers():
            hooks.append(layer.register_forward_pre_hook(lambda module,args,name=name:inputs.__setitem__(name,args[0].detach().clone())))
        try:records=collect(probe,x,t)
        finally:
            for h in hooks:h.remove()
        assert len(records)==len(inputs)==6
        for record in records:
            layer=dict(probe.lowrank_layers())[record['layer']];h=inputs[record['layer']]
            with torch.no_grad():
                duration=layer.step_size(h);carrier=F.linear(F.linear(h,layer.right),layer.left)
                denom=h.square().mean().sqrt();expected=float((duration*carrier).square().mean().sqrt()/denom)
                np.testing.assert_allclose(record['activation']['term1_rms_over_input'],expected,atol=2e-7,rtol=2e-6)
                if layer.role=='lift':
                    U=layer.left;V=F.pad(layer.right,(0,layer.state_width-64));rates=duration.mean(0)
                else:
                    U=F.pad(layer.left,(0,0,0,layer.state_width-64));V=layer.right
                    rates=F.pad(duration.mean(0),(0,layer.state_width-64),value=1.)
                full=U@V;compressed=V@(rates[:,None]*U)
                torch.testing.assert_close(full,layer.full_operator(),atol=2e-7,rtol=2e-6)
                torch.testing.assert_close(compressed,layer.compressed_operator(duration.mean(0)),atol=2e-7,rtol=2e-6)
                equal(record['raw'],spectrum(layer.full_operator()))
                equal(record['compressed_at_probe_mean_duration'],spectrum(layer.compressed_operator(duration.mean(0))))
        from scripts.check_gate_geometry import jacobian,gate_context
        probe.double();xx=x[:2].double();tt=t[:2].double()
        native=jacobian(probe,xx,tt);captured_gates={}
        with gate_context(probe,captured_gates,capture=True):probe(xx,tt)
        assert not captured_gates
        with gate_context(probe,captured_gates):anchored=jacobian(probe,xx,tt)
        equal(native[0],anchored[0]);equal(native[1],anchored[1])
        smokes.append(dict(label=label,state='PASS',resume_bit_identical=True,
            learned_factor_changes=changes,learned_controller_norms=[float(v.norm()) for v in controllers],
            learned_extra_branch_changes=extra,actual_hidden_records_checked=6,
            native_anchored_equal_no_ablation_claim=True,final_layer_records=metrics[-1]['ode_layers'],
            sampling_checkpoints_valid=True,checkpoint_sha256=digests))
    write(out/'lowrank-trainer-resume.json',dict(state='PASS',rows=smokes))

    historical_trainers=[]
    for label in ('swiglu','silu128'):
        cfg,path=smoke_config(configs['product16',label],label,out);kind=cfg['models'][0]
        for version,src in (('old',PRIOR/'source'),('new',source)):
            command=[sys.executable,'-m','synthetic.train','--config',str(path),'--dataset','rotated_product_bimodal_16d',
                '--model',kind,'--seed','0','--device','cpu','--output-root',str(out/(label+'-'+version))]
            result=subprocess.run(command,cwd=src,capture_output=True,text=True)
            (out/(label+'-'+version+'.log')).write_text(result.stdout+result.stderr);assert result.returncode==0,result.stderr
        subpath=Path(cfg['experiment_id'])/f'rotated_product_bimodal_16d/{kind}/seed000/checkpoints/step_0000003.pt'
        a=torch.load(out/(label+'-old')/subpath,weights_only=False,map_location='cpu')
        b=torch.load(out/(label+'-new')/subpath,weights_only=False,map_location='cpu')
        for key in STATE_KEYS:equal(a[key],b[key])
        historical_trainers.append(dict(label=label,state='PASS',steps=3,fulltrainer_bit_identical=True))
    write(out/'historical-baseline-trainer.json',dict(state='PASS',arms=historical_trainers,historical_source=str(PRIOR/'source')))

    # These are historical outcomes, independently pinned from initialization references.
    repo=Path('/Users/mr.z/ODE-FFN-report-work');archive=repo/'docs/research-architecture-search/analysis-affine-forcing'
    archive_manifest=archive/'artifact-sha256.json'
    assert sha(archive_manifest)=='b83e112e76204cb0c2293f40e927c9a5a473c30ab69a17ff75d26c2c13e3ab5a'
    archive_files=json.loads(archive_manifest.read_text())['files']
    for name,digest in archive_files.items():assert sha(archive/name)==digest
    pins={str(archive_manifest):sha(archive_manifest),str(PRIOR/'source-manifest.json'):PARENT_MANIFEST_SHA}
    comparison=archive/'comparison.csv';pins[str(comparison)]=sha(comparison)
    rows=list(csv.DictReader(comparison.open()));eligible=[];cp_count=0;array_count=0
    for dataset in references:
        suffix='product16' if dataset.startswith('rotated') else 'v1';study=PRIOR/'output'/f'affine_forcing_{suffix}_20261025_m1'
        for label in ('swiglu_budget','sigmoid_glu'):
            config=study/'configs'/f'{label}.json';pins[str(config)]=sha(config);cfg=json.loads(config.read_text());kind=cfg['models'][0]
            run_dir=study/'runs'/cfg['experiment_id']/dataset/kind/'seed000'
            status=run_dir/'COMPLETE';pins[str(status)]=sha(status);assert status.is_file()
            metric=run_dir/'metrics.jsonl';pins[str(metric)]=sha(metric);end=json.loads(metric.read_text().splitlines()[-1]);assert end['step']==5000
            data=Path(cfg['data_root']);dm=data/'manifest.json';pins[str(dm)]=sha(dm)
            record=next(row for row in rows if row['dataset']==dataset and row['label']==label)
            assert float(record['validation_oracle_mse'])==end['validation_oracle_mse']
            assert float(record['validation_sw2'])==end['sample']['sliced_w2']
            diag=study/'diagnostics'/dataset/label;summary_path=diag/'summary.json';pins[str(summary_path)]=sha(summary_path);diag_summary=json.loads(summary_path.read_text())
            for step in (0,1000,2000,3000,4000,5000):
                path=run_dir/'checkpoints'/f'step_{step:07d}.pt';pins[str(path)]=sha(path)
                checkpoint=torch.load(path,weights_only=False,map_location='cpu');cp_count+=1
                assert checkpoint['step']==step and checkpoint['config_sha256']==sha(config)
                assert checkpoint['dataset_manifest_sha256']==sha(dm)
                if step==5000:assert diag_summary['checkpoint_sha256']==sha(path)
            assert diag_summary['config_sha256']==sha(config) and diag_summary['manifest_sha256']==sha(dm)
            for filename in ('train.npz','validation.npz','probe.npz'):
                path=data/dataset/filename;pins[str(path)]=sha(path)
                assert json.loads(dm.read_text())['files'][dataset+'/'+filename]['sha256']==pins[str(path)]
            assert diag_summary['targets_sha256']==pins[str(data/dataset/'validation.npz')]
            for filename,digest in diag_summary['arrays_sha256'].items():
                path=diag/filename;pins[str(path)]=sha(path);assert pins[str(path)]==digest;array_count+=1
            assert float(record['field_mse'])==diag_summary['field_mse'] and float(record['jacobian_error'])==diag_summary['jacobian_error']
            eligible.append(dict(dataset=dataset,label=label,historical_record=record,checkpoint=diag_summary['checkpoint_sha256'],config_sha256=sha(config)))
    assert cp_count==36 and array_count==30 and len(eligible)==6
    write(out/'historical-frontier-bindings.json',dict(state='PASS',source=str(PRIOR/'source'),source_files=595,
        rows=eligible,checkpoints=cp_count,arrays=array_count,input_sha256=pins,
        scope='Historical SwiGLU96/GLU128 metric frontier; no new training cells or independent repeats.'))

    verify_distinct_ancestors(manifest)
    for ref in references.values():assert sha(ref['checkpoint'])==ref['sha256']
    for path,digest in pins.items():assert sha(path)==digest
    assert not list((ROOT/'output').iterdir())
    dependency=dict(state='PASS',python=sys.version,platform=platform.platform(),torch=torch.__version__,numpy=np.__version__,
        scipy=scipy.__version__,psutil=psutil.__version__,requirements_sha256=sha(source/'requirements-synthetic.txt'),
        actual_psutil_heartbeat=True,memory_available_gib=psutil.virtual_memory().available/2**30)
    write(out/'dependency-check.json',dependency)
    summary=dict(state='PASS',authoritative_attempt=3,
        earlier_attempts=[dict(attempt=1,session=85736,exit_code=1,scope='Harness-only overstrict float32 difference-of-norm identity; preserved script/log. F64 identity and unchanged F32 component tolerances are authoritative.'),
                          dict(attempt=2,session=19879,exit_code=1,scope='Harness-only three-entry padding tuple in independent reconstruction; frozen helper uses correct four-entry padding. Preserved script/log.')],
        relevant_tests=test_count,formal_cells=18,common_backbone_initializations=len(initial),
        full_width_initialization_cases=len(paired),historical_execution_count=18,historical_distinct_effective_settings=17,
        historical_3update_trajectories='BIT_IDENTICAL',three_lowrank_fulltrainer_resumes='BIT_IDENTICAL',
        learned_factors_controllers_extra_branches_active=True,both_generic_fulltrainers='BIT_IDENTICAL',
        historical_frontier_checkpoints=cp_count,historical_frontier_arrays=array_count,
        actual_psutil_heartbeat=True,all_runner_workers_intercepted=True,native_spectral_and_hidden_capture_checked=True,
        source_files_verified=604,historical_source=str(PRIOR/'source'),historical_source_files=595,
        explicit_predecessor_and_manifest_verified=True,prior_source_unchanged=True,reference_hashes_unchanged=True,
        formal_training_started=False,formal_outputs_absent=True,preflight_script_sha256=sha(__file__))
    write(out/'summary.json',summary)
    for filename in ('summary.json','full-width-initialization.json','all-arm-initialization.json','lowrank-trainer-resume.json',
                     'historical-baseline-trainer.json','executed-call-counts.json','historical-model-trajectories.json',
                     'historical-frontier-bindings.json','dependency-check.json'):
        shutil.copy2(out/filename,ROOT/'preflight'/filename)
    launch=[sys.executable,str(source/'scripts/run_architecture_study.py'),'--plan',
        str(source/'configs/synthetic/architecture_search/augmented_state_queue_20261025_m1.json'),
        '--data-root','/Users/mr.z/ode-workspace-prep/data/synthetic/v1',
        '--output',str(ROOT/'output/augmented_state_queue_20261025_m1')]
    # All environment/dependency/summary writes precede ready. Never edit ready after exposure.
    write(ROOT/'ready.json',dict(state='PREFLIGHT_PASS_AWAITING_PARENT_REVIEW',formal_training_started=False,formal_cells=18,
        runtime=str(source),launch_command=launch,cwd=str(source),
        environment=dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES=''),
        preflight_summary_sha256=sha(out/'summary.json'),source_manifest_sha256=sha(ROOT/'source-manifest.json'),
        source_delta_sha256=sha(ROOT/'source-delta.patch'),dependency_check_sha256=sha(out/'dependency-check.json'),
        historical_frontier_bindings_sha256=sha(out/'historical-frontier-bindings.json'),code_review_sha256=sha(review)))
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
