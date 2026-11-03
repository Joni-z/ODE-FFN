"""Frozen outer-transform sensitivity; native objective and sampler unchanged."""
import argparse,csv,json,math,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cycle11'))
from trial11 import base,build_model

def flip(model):
    for _,layer in model.ode_layers():
        assert layer.gate_transform in ['sigmoid','affine']
        layer.gate_transform='affine' if layer.gate_transform=='sigmoid' else 'sigmoid'

def tensors(model):return {**{f'p/{n}':v.detach().clone() for n,v in model.named_parameters()},**{f'b/{n}':v.detach().clone() for n,v in model.named_buffers()}}
def unchanged(model,before):
    after=tensors(model);assert before.keys()==after.keys()
    assert all(torch.equal(before[n],after[n]) for n in before)

def self_test(device):
    torch.manual_seed(0);rows=[]
    g=torch.linspace(-200,200,401,device=device,dtype=torch.double,requires_grad=True)
    sig=.2+.8*torch.sigmoid(g/10);aff=.6+.02*g
    slope=torch.autograd.grad((aff-sig).sum(),g)[0]
    assert torch.allclose(slope,.02*torch.tanh(g/20).square(),atol=1e-12,rtol=1e-12)
    assert torch.allclose(sig,.6+.4*torch.tanh(g/20),atol=1e-12,rtol=1e-12)
    small=g.detach().new_tensor([.5,1.,2.]);delta=.02*small-.4*torch.tanh(small/20)
    assert torch.allclose(delta,small.pow(3)/60000,atol=0,rtol=.005)
    for family in ['vector_pd_k2','pd_factorized_linear','pd_mlp_affine','pd_factorized_affine']:
        torch.manual_seed(0);model=build_model(16,family).to(device).double().eval()
        layer=next(model.ode_layers())[1];actual=layer.duration_from_logit(g)
        expected=aff if layer.gate_transform=='affine' else sig
        assert torch.allclose(actual,expected,atol=1e-12,rtol=1e-12)
        actual_slope=torch.autograd.grad(actual.sum(),g)[0]
        expected_slope=torch.full_like(g,.02) if layer.gate_transform=='affine' else .02*(1-torch.tanh(g/20).square())
        assert torch.allclose(actual_slope,expected_slope,atol=1e-12,rtol=1e-12)

        with torch.no_grad():
            # A deliberately nonzero fixture readout avoids the native zero-head tautology.
            model.output.weight.copy_(torch.linspace(-.05,.05,model.output.weight.numel(),device=device,dtype=torch.double).reshape_as(model.output.weight))
            x=torch.randn(16,16,device=device,dtype=torch.double);t=torch.linspace(.01,.99,16,device=device,dtype=torch.double)[:,None];before=tensors(model);original=model(x,t).clone();modes=[l.gate_transform for _,l in model.ode_layers()]
            flip(model);alternative=model(x,t);assert bool(torch.isfinite(alternative).all());assert not torch.equal(original,alternative)
            flip(model);unchanged(model,before);assert torch.equal(original,model(x,t)) and modes==[l.gate_transform for _,l in model.ode_layers()]
        rows.append(dict(family=family,nonzero_readout=True,nontrivial_swap=True,round_trip_prediction_exact=True,tensors_buffers_unchanged=True))
    return dict(passed=True,device=str(device),torch=torch.__version__,scalar_formulas_derivatives=True,rows=rows)

def load(case,workspace,device):
    folder=workspace/case['run_dir']/'output'/case['experiment_id']/case['dataset']/case['id']/'seed000';checkpoint=folder/'checkpoints/step_0020000.pt'
    assert (folder/'ADAPTER_COMPLETE').exists() and base.digest(checkpoint)==case['checkpoint_sha256']
    cfg=json.loads((folder/'config.json').read_text());state=torch.load(checkpoint,map_location=device,weights_only=False);manifest_file=Path(cfg['data_root'])/'manifest.json'
    assert state['step']==cfg['steps']==20000 and cfg['sample_count']==4096 and cfg['heun_steps']==32 and cfg['seeds']==[0]
    assert state['config_sha256']==base.digest(folder/'config.json') and state['dataset_manifest_sha256']==base.digest(manifest_file)
    assert cfg['cell']['family']==case['family'] and cfg['cell']['dataset']==case['dataset'] and cfg['cell']['data_version']==case['data_version']
    base.verify_data(Path(cfg['data_root']).parent,cfg['cell']);spec=json.loads(manifest_file.read_text())['datasets'][case['dataset']]
    fixed=base.train.fixed_eval_data(Path(cfg['data_root'])/case['dataset'],base.train.load_distribution(spec),spec['dim'],0,device,4096,256,'test')
    torch.manual_seed(0);model=build_model(spec['dim'],case['family']).to(device).eval();base.train.restore_model(model,state)
    assert len(list(model.ode_layers()))==6
    assert all(l.orders==2 and l.rate_placement=='before_projection' and l.gate_transform==case['native_transform'] and l.spectral_cap is None and l.operator_mode=='multiplicative' and l.solve_return=='map' and l.forcing_mode=='none' for _,l in model.ode_layers())
    binding=dict(case,config_sha256=state['config_sha256'],manifest_sha256=state['dataset_manifest_sha256'],checkpoint_path=str(checkpoint),config_path=str(folder/'config.json'))
    return folder,model,fixed,state,binding

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--device',default='cuda');p.add_argument('--host');p.add_argument('--workspace');p.add_argument('--out',required=True);p.add_argument('--self-test',action='store_true');p.add_argument('--witness',action='store_true');p.add_argument('--detach',action='store_true');a=p.parse_args()
    base.verify_sources();torch.set_num_threads(1);device=torch.device(a.device);out=Path(a.out);cohort_file=Path(__file__).with_name('cases.json');cohort=json.loads(cohort_file.read_text());source=Path(a.source).resolve();assert source==base.HERE
    for file,digest in cohort['source_csvs'].items():assert base.digest(source.parents[1]/file)==digest
    assert len(cohort['cases'])==len({r['id'] for r in cohort['cases']})==24
    tables={f:list(csv.DictReader((source.parents[1]/f).open())) for f in cohort['source_csvs']}
    for case in cohort['cases']:
        match=next(r for r in tables[case['source_csv']] if r['id']==case['id'] and r['checkpoint_sha256']==case['checkpoint_sha256'])
        assert all(case[k]==match[k] for k in ['id','dataset','data_version','host','family','optimizer'])
        assert case['projection_lr']==float(match['projection_lr']) and case['native_field_mse']==float(match['replayed_field_mse']) and case['native_test_sw2']==float(match['numpy_test_sw2'])

    if a.detach:
        assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True);command=[sys.executable,'-u',str(Path(__file__).resolve())]+[v for v in sys.argv[1:] if v!='--detach']
        flags=(subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_BREAKAWAY_FROM_JOB) if os.name=='nt' else 0
        with out.with_suffix('.log').open('xb') as log:
            proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=flags,start_new_session=os.name!='nt')
        base.dump(out.with_suffix('.pid.json'),dict(pid=proc.pid,command=command));print(json.dumps(dict(pid=proc.pid,out=str(out))));return
    if a.self_test:
        result=self_test(device);result.update(cohort_records_verified=24,source_csvs_verified=len(tables),producer_sha256=base.digest(__file__),cohort_sha256=base.digest(cohort_file));base.dump(out,result);print(json.dumps(result));return
    assert a.host in ['145','lenovo'] and a.workspace;workspace=Path(a.workspace);cases=[c for c in cohort['cases'] if c['host']==a.host]
    assert len(cases)==12
    path_bindings=[]
    if a.witness:
        for case in cases:
            folder=workspace/case['run_dir']/'output'/case['experiment_id']/case['dataset']/case['id']/'seed000';file=folder/'checkpoints/step_0020000.pt';cfg=json.loads((folder/'config.json').read_text())
            assert (folder/'ADAPTER_COMPLETE').exists() and base.digest(file)==case['checkpoint_sha256']
            assert cfg['cell']['id']==case['id'] and cfg['cell']['data_version']==case['data_version'] and cfg['steps']==20000
            base.verify_data(Path(cfg['data_root']).parent,cfg['cell'])
            path_bindings.append(dict(cell=case['id'],checkpoint_sha256=case['checkpoint_sha256'],config_sha256=base.digest(folder/'config.json'),manifest_sha256=base.digest(Path(cfg['data_root'])/'manifest.json')))
        assert len(path_bindings)==12
    if a.witness:cases=[c for c in cases if c['optimizer']=='adamw' and c['projection_lr']==.0003];assert len(cases)==4
    out.mkdir(parents=True,exist_ok=False);rows=[];bindings=[];started=time.time();kernel=self_test(device) if a.witness else None
    for case in cases:
        folder,model,fixed,state,binding=load(case,workspace,device);before=tensors(model);px,pt=fixed['probe_x'][:16],fixed['probe_t'][:16]
        with torch.no_grad():native_probe=model(px,pt).clone();assert bool(torch.isfinite(native_probe).all())
        if a.witness:
            with torch.no_grad():
                original_rollout=base.train.generated_samples(model,fixed['sample_source'][:64],32).clone();assert bool(torch.isfinite(original_rollout).all());flip(model);alternative=model(px,pt);flip(model);restored=model(px,pt);restored_rollout=base.train.generated_samples(model,fixed['sample_source'][:64],32)
                assert torch.equal(native_probe,restored) and torch.equal(original_rollout,restored_rollout)
            unchanged(model,before);rows.append(dict(cell=case['id'],family=case['family'],native_probe_finite=True,swapped_probe_finite=bool(torch.isfinite(alternative).all()),round_trip_probe_exact=True,round_trip_native_heun64_exact=True,tensors_buffers_unchanged=True));bindings.append(binding)
        else:
            with np.load(folder/'final_test_examples.npz',allow_pickle=False) as saved:archived=saved['generated'].copy();assert np.array_equal(saved['target'],fixed['sample_target'].cpu().numpy())
            arrays=dict(target=fixed['sample_target'].cpu().numpy(),directions=fixed['directions'].cpu().numpy(),native_probe=native_probe.cpu().numpy());case_rows=[];native_prediction=None
            for action in ['native','swap']:
                if action=='swap':flip(model)
                with torch.no_grad():
                    prediction=torch.cat([model(x,t) for x,t in zip(fixed['x'].split(1024),fixed['t'].split(1024))]).double();finite_field=bool(torch.isfinite(prediction).all());field=float((prediction-fixed['oracle'].double()).square().mean()) if finite_field else None
                    if action=='native':native_prediction=prediction.clone()
                    displacement=float((prediction-native_prediction).square().mean()) if finite_field else None
                    generated=base.train.generated_samples(model,fixed['sample_source'],32);fraction=float(torch.isfinite(generated).all(1).float().mean());finite_generated=bool(torch.isfinite(generated).all())
                    sw2=base.train.sliced_w2(generated,fixed['sample_target'],fixed['directions']) if finite_generated else None
                    if sw2 is not None and not math.isfinite(sw2):sw2=None
                    probe_finite=bool(torch.isfinite(model(fixed['probe_x'],fixed['probe_t'])).all())
                    activation_error=None
                    try:
                        activation=base.train.collect(model,fixed['probe_x'],fixed['probe_t']) if probe_finite else None
                        json.dumps(activation,allow_nan=False)
                    except Exception as exc:
                        if action=='native':raise
                        activation=None;activation_error=f'{type(exc).__name__}: {exc}'
                arrays[action+'_generated']=generated.cpu().numpy()
                row=dict(cell=case['id'],dataset=case['dataset'],family=case['family'],optimizer=case['optimizer'],projection_lr=case['projection_lr'],action=action,transform=next(model.ode_layers())[1].gate_transform,validation_field_mse=field,test_sw2=sw2,finite_field=finite_field,finite_generated_fraction=fraction,sampling_metric_finite=sw2 is not None,probe_finite=probe_finite,field_prediction_displacement_mse=displacement,activation=activation,activation_error=activation_error)
                if action=='native':
                    assert finite_field and finite_generated and sw2 is not None and probe_finite
                    row.update(generated_replay_max_error=float(np.max(np.abs(arrays['native_generated']-archived))),field_replay_error=abs(field-case['native_field_mse']),sw2_replay_error=abs(sw2-case['native_test_sw2']))
                    assert np.array_equal(arrays['native_generated'],archived) and row['field_replay_error']<2e-6 and row['sw2_replay_error']<2e-6
                case_rows.append(row)
            flip(model)
            with torch.no_grad():restored=model(px,pt);assert torch.equal(native_probe,restored)
            unchanged(model,before);arrays['restored_probe']=restored.cpu().numpy();file=out/(case['id']+'.npz');np.savez_compressed(file,**arrays)
            binding.update(array_file=file.name,array_sha256=base.digest(file),round_trip_probe_exact=True,tensors_buffers_unchanged=True);bindings.append(binding);rows+=case_rows
        resources=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15) if device.type=='cuda' else None
        progress=dict(unix_time=time.time(),started=started,pid=os.getpid(),host=a.host,completed_checkpoints=len(bindings),total_checkpoints=len(cases),recorded_evaluations=len(rows),complete=len(bindings)==len(cases),gpu=resources.stdout.strip() if resources else None,last_cell=case['id'])
        base.dump(out/'progress.json',progress)
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(progress)+'\n')
        result=dict(passed=True,complete=False,witness=a.witness,kernel=kernel,path_bindings=path_bindings,cohort_records_verified=24,source_csvs_verified=len(tables),host=a.host,device=str(device),torch=torch.__version__,rows=rows,bindings=bindings,producer_sha256=base.digest(__file__),cohort_sha256=base.digest(cohort_file),native_source_manifest_sha256=base.digest(source/'source_provenance.json'),scope='Frozen attribute swaps only; no training, tensor update or model selection.')
        base.dump(out/'results.json',result);print(json.dumps(progress),flush=True);del model,state,fixed
    assert len(bindings)==(4 if a.witness else 12) and len(rows)==(4 if a.witness else 24)
    result['complete']=True;base.dump(out/'results.json',result)
    base.dump(out/'finished.json',dict(passed=True,pid=os.getpid(),checkpoints=len(bindings),evaluations=len(rows),results_sha256=base.digest(out/'results.json')))

if __name__=='__main__':main()
