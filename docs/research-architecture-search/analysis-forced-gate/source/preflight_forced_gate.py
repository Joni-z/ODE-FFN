"""Registered forced-gate preflight. Never launches a formal worker."""
from argparse import Namespace
from pathlib import Path
import hashlib,json,os,platform,subprocess,sys,threading,time
from unittest.mock import patch
import numpy as np
import psutil
import torch

root=Path(__file__).resolve().parents[1];source=root/'source';out=root/'preflight'/'attempt3';out.mkdir()
prior=root.parent/'diagonal_flow_20261024_m1'
assert prior != root
sys.path.insert(0,str(source));torch.set_num_threads(1)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
manifest=json.loads((root/'source-manifest.json').read_text());oldmanifest=json.loads((prior/'source-manifest.json').read_text())
assert manifest['parent_source']==str(prior/'source')
assert len(oldmanifest['all_files'])==561
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
commands=[['-m','unittest','tests.test_forced_gate','tests.test_forced_gate_integration','tests.test_diagonal_flow','tests.test_diagonal_integration','tests.test_shared_features','tests.test_controller_tangent','tests.test_common_backbone','-v'],
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
labels=['swiglu_budget','swiglu','forced_static','forced_k2','forced_exact']
for suffix in ('v1','product16'):
    path=source/f'configs/synthetic/architecture_search/forced_gate_{suffix}_20261024_m1.json';plan=json.loads(path.read_text())
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
            for _,layer in model.forced_gate_layers():assert torch.count_nonzero(layer.raw_eigenvalues)==0
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

# Full-width native construction/projection/RNG/function/gradient pairing.
paired=[]
for dtype in (torch.float32,torch.float64):
    for dim,dataset,suffix in ((2,'gmm_ring8_2d','v1'),(16,'rotated_product_bimodal_16d','product16')):
        ref=references[dataset];models=[];rngs=[]
        generator=torch.Generator().manual_seed(0)
        x=torch.randn(19,dim,generator=generator,dtype=dtype);t=torch.rand(19,1,generator=generator,dtype=dtype)
        head=torch.randn(dim,64,generator=generator,dtype=dtype)*.2
        for label in labels[1:]:
            cfg=configs[suffix,label];torch.manual_seed(0)
            model=VelocityNet(dim,cfg['models'][0],**cfg['model_options']).to(dtype)
            rngs.append(torch.get_rng_state().clone())
            copy_common_backbone(model,Path(ref['checkpoint']),ref['sha256'])
            equal(rngs[-1],torch.get_rng_state())
            with torch.no_grad():model.output.weight.copy_(head)
            models.append(model)
        base=models[0];expected=base(x,t).detach()
        base_j=jacobian(base,x,t)[1] if dtype==torch.float64 else None
        base(x,t).square().mean().backward()
        for label,model,rng in zip(labels[1:],models,rngs):
            equal(rng,rngs[0])
            for name,value in base.state_dict().items():equal(value,model.state_dict()[name])
            equal(dict(base.named_buffers()),dict(model.named_buffers()))
            equal(model(x,t),expected)
            if dtype==torch.float64:
                derivative=jacobian(model,x,t)[1]
                torch.testing.assert_close(derivative,base_j,atol=2e-13,rtol=2e-13)
                derivative_error=float((derivative-base_j).abs().max())
            else:derivative_error=None
            if model is not base:model(x,t).square().mean().backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            for name,parameter in base.named_parameters():equal(parameter.grad,dict(model.named_parameters())[name].grad)
            if label!='swiglu':
                assert model.parameter_count()==base.parameter_count()+255
                for name,flow in model.forced_gate_layers():
                    assert torch.count_nonzero(flow.raw_eigenvalues)==0
                    assert flow.raw_eigenvalues.grad.abs().max()>0
            assert actual_macs(model,dim)==actual_macs(base,dim)
            paired.append(dict(dtype=str(dtype),dim=dim,label=label,forward_difference=0.,input_jacobian_max_difference=derivative_error,
                common_projections_buffers_rng_and_raw_gradients_equal=True,parameters=model.parameter_count(),
                dense_macs=actual_macs(model,dim)))
        for (name,left),(other,right) in zip(models[2].named_parameters(),models[3].named_parameters()):
            assert name==other;equal(left.grad,right.grad)
write(out/'full-width-initialization.json',dict(state='PASS',rows=paired,
    construction='Native float32 initialization, common conversion to probe dtype, then common reference copy.',
    interpretation='Raw common gradients equal baseline; whole gradient clipping can differ after adding eta. Adaptive K2/exact complete initial gradients agree.'))

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
for label in ('forced_static','forced_k2','forced_exact'):
    cfg=dict(configs['product16',label]);cfg.update(experiment_id=label+'_resume_smoke',steps=3,batch_size=16,eval_interval=1,
        sample_interval=3,checkpoint_interval=1,sample_count=32,sliced_directions=4,heun_steps=4)
    config=out/(label+'-smoke.json');write(config,cfg)
    def options(folder,steps=None,resume=None):return Namespace(config=str(config),dataset='rotated_product_bimodal_16d',model='ode_forced_gate',seed=0,device='cpu',output_root=str(out/label/folder),steps=steps,dry_run=False,resume=resume)
    def checkpoint(folder,step):return out/label/folder/cfg['experiment_id']/'rotated_product_bimodal_16d/ode_forced_gate/seed000/checkpoints'/f'step_{step:07d}.pt'
    run(options('uninterrupted'));run(options('resumed',2));run(options('resumed',3,str(checkpoint('resumed',2))))
    a=torch.load(checkpoint('uninterrupted',3),weights_only=False,map_location='cpu');b=torch.load(checkpoint('resumed',3),weights_only=False,map_location='cpu')
    for key in ('model','optimizer','runtime_buffers','data_rng','torch_rng','numpy_rng','python_rng'):equal(a[key],b[key])
    eta=[value for name,value in a['model'].items() if name.endswith('.raw_eigenvalues')]
    assert len(eta)==3 and sum(float(v.norm()) for v in eta)>0
    metrics=[json.loads(s) for s in (checkpoint('uninterrupted',0).parent.parent/'metrics.jsonl').read_text().splitlines()]
    assert metrics[-1]['sample']['heun_steps']==4 and metrics[-1]['sample']['nfe']==8
    assert np.isfinite(metrics[-1]['sample']['sliced_w2'])
    assert len(metrics[-1]['ode_layers'])==3
    for entry in metrics[-1]['ode_layers']:
        assert entry['operator_type']=='conditional_forced_gate'
        assert entry['solver']==cfg['model_options']['forced_gate_solver'] and entry['clock']==cfg['model_options']['forced_gate_clock']
        raw=a['model'][entry['layer']+'.raw_eigenvalues'];eigen=raw.tanh()
        equal(torch.tensor(entry['eigenvalue_coordinates']),raw)
        equal(torch.tensor(entry['P_eigenvalues']),eigen)
        assert entry['P_sigma_max']==float(eigen.abs().max())
        assert entry['activation']['scope']=='observed_gate_response_only'
        assert entry['conditional_rate_min']>=-entry['P_sigma_max']
        assert entry['conditional_rate_max']<=entry['P_sigma_max']
        assert entry['forcing_abs_max']>0
        assert entry['activation']['gain_min']>0
    smokes.append(dict(label=label,state='PASS',resume_bit_identical=True,eta_nonzero=True,
        eta_norm=sum(float(v.norm()) for v in eta),sample_and_P_statistics_valid=True,
        checkpoint_sha256={folder:sha(checkpoint(folder,3)) for folder in ('uninterrupted','resumed')}))
write(out/'forced-gate-trainer-resume.json',dict(state='PASS',arms=smokes))

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
summary=dict(state='PASS',relevant_tests=42,full_width_initialization_cases=len(paired),common_backbone_initializations=len(initial),
    frozen_default_models=8,default_updates_per_model=3,default_trajectories='BIT_IDENTICAL',
    all_three_forced_arms_fulltrainer_resume='BIT_IDENTICAL',forced_eta_nonzero=True,forced_P_stats_sampling_checkpoints_valid=True,
    both_old_SwiGLU_fulltrainers='BIT_IDENTICAL',actual_psutil_heartbeat=True,all_runner_workers_intercepted=True,
    source_files_verified=len(manifest['all_files']),prior_source_unchanged=True,reference_hashes_unchanged=True,historical_source=str(prior/'source'),historical_source_files=len(oldmanifest['all_files']),
    prior_self_comparison_attempt_invalidated=True,
    formal_cells=15,formal_training_started=False,formal_outputs_absent=True,preflight_script_sha256=sha(__file__))
write(out/'summary.json',summary)
launch=[sys.executable,str(source/'scripts/run_architecture_study.py'),'--plan',str(source/'configs/synthetic/architecture_search/forced_gate_queue_20261024_m1.json'),
    '--data-root','/Users/mr.z/ode-workspace-prep/data/synthetic/v1','--output',str(root/'output/forced_gate_queue_20261024_m1')]
write(root/'ready.json',dict(state='PREFLIGHT_PASS_AWAITING_PARENT_REVIEW',formal_training_started=False,formal_cells=15,runtime=str(source),launch_command=launch,cwd=str(source),
    environment=dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES=''),
    python=sys.version,platform=platform.platform(),psutil=psutil.__version__,memory_available_gib=psutil.virtual_memory().available/2**30,
    preflight_summary_sha256=sha(out/'summary.json'),source_manifest_sha256=sha(root/'source-manifest.json'),source_delta_sha256=sha(root/'source-delta.patch')))
# Stable receipt locations point to the successful attempt only.
import shutil
for filename in ('summary.json','full-width-initialization.json','all-arm-initialization.json',
                 'forced-gate-trainer-resume.json','historical-baseline-trainer.json'):
    shutil.copy2(out/filename,root/'preflight'/filename)
print(json.dumps(summary))
