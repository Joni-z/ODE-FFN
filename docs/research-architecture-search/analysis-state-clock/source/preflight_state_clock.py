"""Registered original state-clock preflight. Never launches a formal worker."""
from argparse import Namespace
from pathlib import Path
import hashlib,json,os,platform,subprocess,sys,threading,time
from unittest.mock import patch
import numpy as np
import psutil
import torch

root=Path(__file__).resolve().parents[1];source=root/'source';out=root/'preflight'/'attempt1';out.mkdir()
prior=Path('/Users/mr.z/ode-workspace-prep/architecture-search/forced_gate_20261024_m1')
assert prior != root
sys.path.insert(0,str(source));torch.set_num_threads(1)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
manifest=json.loads((root/'source-manifest.json').read_text());oldmanifest=json.loads((prior/'source-manifest.json').read_text())
assert manifest['parent_source']==str(prior/'source')
assert len(oldmanifest['all_files'])==571
def verify():
    for name,digest in manifest['all_files'].items():assert sha(source/name)==digest,name
    for name,digest in oldmanifest['all_files'].items():assert sha(prior/'source'/name)==digest,name
def equal(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,atol=0,rtol=0)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for key in a:equal(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b
verify()
commands=[['-m','unittest','tests.test_state_clock','tests.test_state_clock_integration','tests.test_forced_gate','tests.test_forced_gate_integration','tests.test_diagonal_flow','tests.test_diagonal_integration','tests.test_shared_features','tests.test_controller_tangent','tests.test_common_backbone','-v'],
          ['scripts/check_rate_order_reference.py','--reference-file',str(prior/'source/synthetic/models.py')]]
for i,args in enumerate(commands):
    result=subprocess.run([sys.executable,*args],cwd=source,capture_output=True,text=True)
    (out/f'check-{i}.log').write_text(result.stdout+result.stderr);assert result.returncode==0,(i,result.stderr)

from synthetic.models import VelocityNet
from synthetic.initialization import copy_common_backbone
from synthetic.train import run
from scripts.check_gate_geometry import jacobian
from scripts.run_clock_study import monitor
import scripts.run_architecture_study as runner

def actual_macs(model,dim):
    # The formal runner uses float32; dtype-aware preflight also probes float64.
    from torch.nn import functional as F
    count=[0];linear=F.linear;parameter=next(model.parameters())
    def instrument(x,weight,bias=None):
        count[0]+=(x.numel()//x.shape[-1])*weight.numel()
        return linear(x,weight,bias)
    with patch.object(F,'linear',instrument),torch.no_grad():
        model(torch.zeros(1,dim,dtype=parameter.dtype),torch.zeros(1,1,dtype=parameter.dtype))
    return count[0]

template=json.loads((source/'configs/synthetic/gates_20261021/ode_original.json').read_text())
initial=[];references={};configs={}
labels=['state_euler','state_frozen','state_midpoint','swiglu_budget','swiglu','swiglu_compute']
for suffix in ('v1','product16'):
    path=source/f'configs/synthetic/architecture_search/state_clock_{suffix}_20261025_m1.json';plan=json.loads(path.read_text())
    assert [a['label'] for a in plan['arms']]==labels
    assert (plan['seed'],plan['steps'],plan['device'],plan['workers'])==(0,5000,'cpu',2)
    for dataset in plan['datasets']:
        dim=16 if dataset.startswith('rotated') else 2
        for arm in plan['arms']:
            ref=arm['backbone_initialization'][dataset];assert sha(ref['checkpoint'])==ref['sha256'];references[dataset]=ref
            saved=torch.load(ref['checkpoint'],weights_only=False,map_location='cpu')
            options=dict(template['model_options']);options.update(arm['options'])
            torch.manual_seed(0);model=VelocityNet(dim,arm['model'],**options)
            native={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
            receipt=copy_common_backbone(model,Path(ref['checkpoint']),ref['sha256']);equal(rng,torch.get_rng_state())
            for name,value in model.state_dict().items():equal(value,native[name] if '.ffn.' in name else saved['model'][name])
            equal(dict(model.named_buffers()),saved['runtime_buffers'])
            for _,layer in model.state_clock_layers():
                assert torch.count_nonzero(layer.t_func[-1].weight)==0 and torch.count_nonzero(layer.t_func[-1].bias)==0
                assert torch.count_nonzero(layer.proj.weight)>0
            initial.append(dict(dataset=dataset,label=arm['label'],parameters=model.parameter_count(),forward_macs=actual_macs(model,dim),
                native_ffn_and_rng_unchanged=True,**receipt))
    data=Path('/Users/mr.z/ode-workspace-prep/data/synthetic')/('v1' if suffix=='v1' else 'product16_v1')
    try:
        with patch.object(runner.subprocess,'run',side_effect=RuntimeError('PREFLIGHT_WORKER_DISABLED')):runner.run(path,data,out/f'runner-{suffix}')
    except RuntimeError as error:assert str(error)=='PREFLIGHT_WORKER_DISABLED'
    else:raise AssertionError('worker interception not reached')
    for arm in plan['arms']:
        cfg=json.loads((out/f'runner-{suffix}/configs'/f"{arm['label']}.json").read_text());configs[suffix,arm['label']]=cfg
        assert cfg['backbone_initialization']==arm['backbone_initialization']
        assert cfg['models']==[arm['model']] and cfg['seeds']==[0] and cfg['steps']==5000
        for key,value in arm['options'].items():assert cfg['model_options'][key]==value
    assert not (out/f'runner-{suffix}/runs').exists()
write(out/'all-arm-initialization.json',dict(state='PASS',arms=initial,references=references))

# Execute and independently count native P/controller calls and dense MACs.
stage_counts=[]
from torch.nn import functional as F
for label in labels[:3]:
    cfg=configs['product16',label];torch.manual_seed(0)
    model=VelocityNet(16,cfg['models'][0],**cfg['model_options'])
    layer_counts={name:dict(projection=0,controller=0) for name,_ in model.state_clock_layers()}
    pointers={layer.proj.weight.data_ptr():name for name,layer in model.state_clock_layers()}
    handles=[]
    for name,layer in model.state_clock_layers():
        def counted_controller(module,args,result,name=name):layer_counts[name]['controller']+=1
        handles.append(layer.t_func.register_forward_hook(counted_controller))
    count=[0];linear=F.linear
    def instrument(x,weight,bias=None):
        count[0]+=(x.numel()//x.shape[-1])*weight.numel()
        if weight.data_ptr() in pointers:layer_counts[pointers[weight.data_ptr()]]['projection']+=1
        return linear(x,weight,bias)
    with patch.object(F,'linear',instrument),torch.no_grad():model(torch.zeros(1,16),torch.zeros(1,1))
    for handle in handles:handle.remove()
    expected_P=1 if label=='state_euler' else 2;expected_controller=2 if label=='state_midpoint' else 1
    assert all(v==dict(projection=expected_P,controller=expected_controller) for v in layer_counts.values())
    assert count[0]==runner.macs(model,16)
    stage_counts.append(dict(label=label,per_layer_calls=layer_counts,actual_macs=count[0],parameters=model.parameter_count()))
write(out/'executed-stage-counts.json',dict(state='PASS',rows=stage_counts))

# Native full width: constant-clock legacy, frozen/midpoint function and tangent scope.
paired=[]
for dtype in (torch.float32,torch.float64):
    for dim,dataset,suffix in ((2,'gmm_ring8_2d','v1'),(16,'rotated_product_bimodal_16d','product16')):
        ref=references[dataset];models=[];rngs=[];features={}
        generator=torch.Generator().manual_seed(0)
        x=torch.randn(19,dim,generator=generator,dtype=dtype);t=torch.rand(19,1,generator=generator,dtype=dtype)
        head=torch.randn(dim,64,generator=generator,dtype=dtype)*.2;target=torch.randn(19,dim,generator=generator,dtype=dtype)
        for label in labels[:3]:
            cfg=configs[suffix,label];torch.manual_seed(0)
            model=VelocityNet(dim,cfg['models'][0],**cfg['model_options']).to(dtype)
            rngs.append(torch.get_rng_state().clone());copy_common_backbone(model,Path(ref['checkpoint']),ref['sha256']);equal(rngs[-1],torch.get_rng_state())
            def capture(module,args,result,label=label):features[label]=result.detach().clone()
            hook=model.norm.register_forward_hook(capture)
            (model(x,t)-target).square().sum().backward();hook.remove();models.append(model)
        frozen,dynamic=models[1:]
        equal(features['state_frozen'],features['state_midpoint'])
        # Zero output weights annihilate every backbone derivative; the affine
        # output Jacobian depends only on these equal captured features.
        for name,parameter in frozen.named_parameters():
            equal(parameter.grad,dict(dynamic.named_parameters())[name].grad)
            if not name.startswith('output.'):assert torch.count_nonzero(parameter.grad)==0
        zero_kernel=features['state_frozen']@features['state_frozen'].T+1
        equal(zero_kernel,features['state_midpoint']@features['state_midpoint'].T+1)
        for model in models:
            model.zero_grad()
            with torch.no_grad():model.output.weight.copy_(head)
            (model(x,t)-target).square().sum().backward()
        equal(frozen(x,t),dynamic(x,t))
        jerr=float((jacobian(frozen,x,t)[1]-jacobian(dynamic,x,t)[1]).abs().max())
        tolerance=3e-6 if dtype==torch.float32 else 3e-13
        assert jerr<=tolerance
        controller_difference=0.;P_difference=0.
        for name,parameter in frozen.named_parameters():
            other=dict(dynamic.named_parameters())[name]
            if '.t_func.2.weight' in name:controller_difference=max(controller_difference,float((parameter.grad-other.grad).abs().max()))
            if '.proj.weight' in name:
                torch.testing.assert_close(parameter.grad,other.grad,atol=tolerance,rtol=tolerance)
                P_difference=max(P_difference,float((parameter.grad-other.grad).abs().max()))
        assert controller_difference>1e-8
        for label,model,rng in zip(labels[:3],models,rngs):
            equal(rng,rngs[0]);equal(model.state_dict(),models[0].state_dict())
            legacy_proof=None
            if label!='state_midpoint':
                cfg=configs[suffix,label];opts=dict(cfg['model_options']);opts.pop('state_clock_solver');opts['orders']=1 if label=='state_euler' else 2
                torch.manual_seed(0);legacy=VelocityNet(dim,'ode_vector',**opts).to(dtype)
                equal(torch.get_rng_state(),rng)
                with torch.no_grad():
                    for _,layer in legacy.ode_layers():layer.t_func[-1].weight.zero_();layer.t_func[-1].bias.zero_()
                copy_common_backbone(legacy,Path(ref['checkpoint']),ref['sha256'])
                with torch.no_grad():legacy.output.weight.copy_(head)
                equal(legacy.state_dict(),model.state_dict());equal(legacy(x,t),model(x,t));(legacy(x,t)-target).square().sum().backward()
                for name,parameter in legacy.named_parameters():equal(parameter.grad,dict(model.named_parameters())[name].grad)
                legacy_proof='BIT_IDENTICAL_FORWARD_STATE_RNG_GRADIENTS'
            paired.append(dict(dtype=str(dtype),dim=dim,label=label,native_P_controller_state_and_RNG_equal=True,
                legacy_constant_clock=legacy_proof,frozen_midpoint_forward_bit_equal=True,
                frozen_midpoint_input_J_max_difference=jerr,frozen_midpoint_P_gradient_max_difference=P_difference,
                zero_head_NTK_pair='frozen_midpoint_only',nonzero_head_controller_gradient_difference=controller_difference,
                parameters=model.parameter_count(),dense_macs=actual_macs(model,dim)))
write(out/'full-width-initialization.json',dict(state='PASS',rows=paired,interpretation='Euler is not function/NTK paired with frozen or midpoint. Frozen/midpoint have equal initial functions and P/input derivatives, but different exposed controller tangents.'))

# Old forced-gate default states/optimizer paths remain unchanged against Stage20.
import importlib.util
spec=importlib.util.spec_from_file_location('stage20_frozen_models',prior/'source/synthetic/models.py')
old_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(old_module)
forced_proofs=[]
for solver,clock in (('exact','static'),('k2','adaptive'),('exact','adaptive')):
    opts=dict(forced_gate_solver=solver,forced_gate_clock=clock)
    torch.manual_seed(0);old_model=old_module.VelocityNet(16,'ode_forced_gate',**opts);old_rng=torch.get_rng_state().clone()
    torch.manual_seed(0);new_model=VelocityNet(16,'ode_forced_gate',**opts);equal(old_rng,torch.get_rng_state())
    generator=torch.Generator().manual_seed(100000)
    with torch.no_grad():
        head=torch.randn(new_model.output.weight.shape,generator=generator);old_model.output.weight.copy_(head);new_model.output.weight.copy_(head)
    optimizers=[torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=0.) for m in (old_model,new_model)]
    for _ in range(3):
        x=torch.randn(31,16,generator=generator);t=torch.rand(31,1,generator=generator);target=torch.randn(31,16,generator=generator)
        equal(old_model(x,t),new_model(x,t))
        for m,opt in zip((old_model,new_model),optimizers):
            opt.zero_grad();(m(x,t)-target).square().mean().backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        equal(old_model.state_dict(),new_model.state_dict());equal(optimizers[0].state_dict(),optimizers[1].state_dict())
    forced_proofs.append(dict(solver=solver,clock=clock,steps=3,state='BIT_IDENTICAL'))
write(out/'forced-reference-check.json',dict(state='PASS',historical_source=str(prior/'source'),arms=forced_proofs))

# Real monitor heartbeat, followed by clean shutdown before smoke training.
heartbeat=out/'monitor-heartbeat';heartbeat.mkdir();stop=threading.Event()
watcher=threading.Thread(target=monitor,args=(heartbeat,stop,dict(state='PREFLIGHT',phase='monitor-check',completed=0,active=[])),daemon=True);watcher.start()
for _ in range(50):
    if (heartbeat/'monitor.json').exists():break
    time.sleep(.1)
assert watcher.is_alive() and (heartbeat/'telemetry.jsonl').is_file()
stop.set();watcher.join(timeout=2);assert not watcher.is_alive()
assert json.loads((heartbeat/'monitor.json').read_text())['pid']==os.getpid()

smokes=[]
for label in labels[:3]:
    cfg=dict(configs['product16',label]);cfg.update(experiment_id=label+'_resume_smoke',steps=3,batch_size=16,eval_interval=1,
        sample_interval=3,checkpoint_interval=1,sample_count=32,sliced_directions=4,heun_steps=4)
    config=out/(label+'-smoke.json');write(config,cfg)
    def options(folder,steps=None,resume=None):return Namespace(config=str(config),dataset='rotated_product_bimodal_16d',model='ode_state_clock',seed=0,device='cpu',output_root=str(out/label/folder),steps=steps,dry_run=False,resume=resume)
    def checkpoint(folder,step):return out/label/folder/cfg['experiment_id']/'rotated_product_bimodal_16d/ode_state_clock/seed000/checkpoints'/f'step_{step:07d}.pt'
    run(options('uninterrupted'));run(options('resumed',2));run(options('resumed',3,str(checkpoint('resumed',2))))
    a=torch.load(checkpoint('uninterrupted',3),weights_only=False,map_location='cpu');b=torch.load(checkpoint('resumed',3),weights_only=False,map_location='cpu')
    for key in ('model','optimizer','runtime_buffers','data_rng','torch_rng','numpy_rng','python_rng'):equal(a[key],b[key])
    weights=[v for k,v in a['model'].items() if '.t_func.2.weight' in k];assert len(weights)==6 and all(v.norm()>0 for v in weights)
    metrics=[json.loads(line) for line in (checkpoint('uninterrupted',0).parent.parent/'metrics.jsonl').read_text().splitlines()]
    assert metrics[-1]['sample']['heun_steps']==4 and metrics[-1]['sample']['nfe']==8 and np.isfinite(metrics[-1]['sample']['sliced_w2'])
    assert len(metrics[-1]['ode_layers'])==6
    for index in (0,-1):
        for entry in metrics[index]['ode_layers']:
            assert entry['operator_type']=='state_clock_vector_field' and entry['solver']==cfg['model_options']['state_clock_solver']
            stats=entry['activation'];assert stats['scope']=='internal_state_clock_probes_only'
            assert stats['midpoint_clock_is_executed_stage']==(label=='state_midpoint')
            assert all(np.isfinite(v) for v in stats.values() if isinstance(v,(int,float)))
            for key in ('initial_clock_mean','midpoint_clock_probe_mean'):assert .2<=stats[key]<=1.
            if index==0:
                assert stats['gate_change_rms']==stats['clock_taylor_probe_rms']==stats['midpoint_correction_probe_rms']==0.
    smokes.append(dict(label=label,state='PASS',resume_bit_identical=True,learned_controller_norms=[float(v.norm()) for v in weights],
        finite_live_stage_statistics=True,sampling_checkpoints_valid=True,
        checkpoint_sha256={folder:sha(checkpoint(folder,3)) for folder in ('uninterrupted','resumed')}))
write(out/'state-clock-trainer-resume.json',dict(state='PASS',arms=smokes))

# Both unchanged SwiGLU budgets must retain complete historical trainer states.
baseline_proofs=[]
for label in ('swiglu_budget','swiglu'):
    cfg=dict(configs['product16',label]);cfg.update(experiment_id=label+'_default_smoke',steps=3,batch_size=16,eval_interval=1,
        sample_interval=3,checkpoint_interval=1,sample_count=32,sliced_directions=4,heun_steps=4)
    config=out/(label+'-baseline-smoke.json');write(config,cfg)
    for version,src in (('old',prior/'source'),('new',source)):
        command=[sys.executable,'-m','synthetic.train','--config',str(config),'--dataset','rotated_product_bimodal_16d',
            '--model','swiglu','--seed','0','--device','cpu','--output-root',str(out/(label+'-'+version))]
        result=subprocess.run(command,cwd=src,capture_output=True,text=True);(out/(label+'-'+version+'.log')).write_text(result.stdout+result.stderr)
        assert result.returncode==0,result.stderr
    subpath=Path(cfg['experiment_id'])/'rotated_product_bimodal_16d/swiglu/seed000/checkpoints/step_0000003.pt'
    a=torch.load(out/(label+'-old')/subpath,weights_only=False,map_location='cpu');b=torch.load(out/(label+'-new')/subpath,weights_only=False,map_location='cpu')
    for key in ('model','optimizer','runtime_buffers','data_rng','torch_rng','numpy_rng','python_rng'):equal(a[key],b[key])
    baseline_proofs.append(dict(label=label,state='PASS',fulltrainer_bit_identical=True,steps=3))
write(out/'historical-baseline-trainer.json',dict(state='PASS',arms=baseline_proofs))
verify()
for ref in references.values():assert sha(ref['checkpoint'])==ref['sha256']
assert list((root/'output').iterdir())==[]
summary=dict(state='PASS',relevant_tests=54,full_width_initialization_cases=len(paired),common_backbone_initializations=len(initial),
    frozen_default_models=8,default_updates_per_model=3,default_trajectories='BIT_IDENTICAL',
    all_three_state_clock_arms_fulltrainer_resume='BIT_IDENTICAL',learned_controllers_nonzero=True,state_clock_P_stats_sampling_checkpoints_valid=True,
    both_old_SwiGLU_fulltrainers='BIT_IDENTICAL',actual_psutil_heartbeat=True,all_runner_workers_intercepted=True,
    source_files_verified=len(manifest['all_files']),prior_source_unchanged=True,reference_hashes_unchanged=True,historical_source=str(prior/'source'),historical_source_files=len(oldmanifest['all_files']),
    explicit_predecessor_and_manifest_verified=True,prior_forced_models=3,prior_forced_updates=3,prior_forced_trajectories="BIT_IDENTICAL",
    formal_cells=18,formal_training_started=False,formal_outputs_absent=True,preflight_script_sha256=sha(__file__))
write(out/'summary.json',summary)
launch=[sys.executable,str(source/'scripts/run_architecture_study.py'),'--plan',str(source/'configs/synthetic/architecture_search/state_clock_queue_20261025_m1.json'),
    '--data-root','/Users/mr.z/ode-workspace-prep/data/synthetic/v1','--output',str(root/'output/state_clock_queue_20261025_m1')]
write(root/'ready.json',dict(state='PREFLIGHT_PASS_AWAITING_PARENT_REVIEW',formal_training_started=False,formal_cells=18,runtime=str(source),launch_command=launch,cwd=str(source),
    environment=dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES=''),
    python=sys.version,platform=platform.platform(),psutil=psutil.__version__,memory_available_gib=psutil.virtual_memory().available/2**30,
    preflight_summary_sha256=sha(out/'summary.json'),source_manifest_sha256=sha(root/'source-manifest.json'),source_delta_sha256=sha(root/'source-delta.patch')))
# Stable receipt locations point to the successful attempt only.
import shutil
for filename in ('summary.json','full-width-initialization.json','all-arm-initialization.json',
                 'state-clock-trainer-resume.json','historical-baseline-trainer.json','forced-reference-check.json','executed-stage-counts.json'):
    shutil.copy2(out/filename,root/'preflight'/filename)
print(json.dumps(summary))
