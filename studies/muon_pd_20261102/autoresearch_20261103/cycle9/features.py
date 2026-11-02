"""Frozen native-feature energy accounting; no training or altered predictor."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch

def summarize(P,h,d,orders,native_output=None):
    assert P.shape==(64,64) and h.shape[1]==64 and d.shape[0]==len(h)
    q=h*d;r1=q@P.T;r2=.5*(r1@P.T)*d if orders==2 else np.zeros_like(r1)
    H=float(np.mean(np.sum(h*h,axis=1)));assert H>1e-20
    M=q.T@q/len(q);S=P@M@P.T;B=P.T@P;N=M/H
    energy=lambda x:float(np.mean(np.sum(x*x,axis=1)))
    E1,E2,Einc=energy(r1),energy(r2),energy(r1+r2)
    cross=float(2*np.mean(np.sum(r1*r2,axis=1)))
    relative=lambda a,b:abs(a-b)/max(abs(a),abs(b),1e-30)
    trace_error=relative(E1,float(np.trace(S)));series_error=relative(Einc,E1+E2+cross)
    assert trace_error<1e-10 and series_error<1e-10
    ps=np.linalg.svd(P,compute_uv=False);qs=np.linalg.svd(q/np.sqrt(len(q)),compute_uv=False);rs=np.linalg.svd(r1/np.sqrt(len(q)),compute_uv=False)
    norm=float(np.sum(q*q)/len(q));mean=q.mean(0);mean_increment=r1.mean(0)
    C=np.geomspace(.5,2,64);rescaled_q=q*C[None,:];rescaled_p=P/C[None,:];rescaled_r=rescaled_q@rescaled_p.T
    coordinate_error=float(np.linalg.norm(rescaled_r-r1)/max(np.linalg.norm(r1),1e-30));assert coordinate_error<1e-10
    out=dict(input_energy=H,feature_energy=norm,increment_energy=E1,normalized_increment_energy=E1/H,first_term_rms_over_input=float(np.sqrt(E1/H)),p_sigma_max=float(ps[0]),p_stable_rank=float(np.sum(ps**2)/ps[0]**2),feature_stable_rank=float(np.sum(qs**2)/qs[0]**2),increment_stable_rank=float(np.sum(rs**2)/rs[0]**2),feature_mean_energy_fraction=float(mean@mean/norm),increment_mean_energy_fraction=float(mean_increment@mean_increment/E1),feature_singular_values=qs.tolist(),increment_singular_values=rs.tolist(),p_singular_values=ps.tolist(),second_term_energy=E2,term_cross_energy=cross,full_increment_energy=Einc,term_cancellation_factor=Einc/(E1+E2),term_alignment_cosine=cross/(2*np.sqrt(E1*E2)) if E2 else None,trace_energy_relative_error=trace_error,series_energy_relative_error=series_error,diagonal_coordinate_prediction_relative_error=coordinate_error)
    if native_output is not None:
        denominator=max(np.sqrt(energy(native_output)),np.sqrt(H),np.sqrt(E1),np.sqrt(E2),1e-30)
        error=float(np.sqrt(energy(native_output-h-r1-r2))/denominator);assert error<5e-6
        out['native_layer_reconstruction_relative_rms_error']=error
    return out,dict(P=P,h=h,d=d,q=q,M=M,N=N,B=B,r1=r1,r2=r2,native_output=native_output if native_output is not None else h+r1+r2)

def capture(model,x,t):
    # Same incoming-hidden hook pattern as native synthetic.diagnostics.collect.
    incoming={};outgoing={};hooks=[]
    for name,layer in model.ode_layers():
        def before(module,args,key=name):incoming[key]=(args[0].detach().clone(),args[1].detach().clone() if len(args)>1 and args[1] is not None else None)
        def after(module,args,output,key=name):outgoing[key]=output.detach().clone()
        hooks.extend([layer.register_forward_pre_hook(before),layer.register_forward_hook(after)])
    try:
        with torch.no_grad():hooked=model(x,t).clone()
    finally:
        for hook in hooks:hook.remove()
    with torch.no_grad():assert torch.equal(hooked,model(x,t))
    return incoming,outgoing

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--device',default='cuda');p.add_argument('--host');p.add_argument('--cycle8-output');p.add_argument('--cycle6-output');p.add_argument('--out',required=True);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    source=Path(a.source).resolve();sys.path.insert(0,str(source/'autoresearch_20261103/cycle8'));from trial8 import base,build_model
    base.verify_sources();torch.set_num_threads(1);device=torch.device(a.device);out=Path(a.out)
    cases_file=Path(__file__).with_name('cases.json');cases=json.loads(cases_file.read_text())['cases']
    assert len(cases)==len({c['id'] for c in cases})==14
    assert all(sum(c['host']==host for c in cases)==7 for host in ['145','lenovo'])
    assert all(c['reference_id'] is None or c['reference_id'] in {v['id'] for v in cases} for c in cases)
    if a.self_test:
        tests=[]
        for family in ['scalar_k2','pd_direct_raw_k1']:
            torch.manual_seed(0);model=build_model(16,family).to(device);model.eval();rng=torch.Generator(device=device).manual_seed(0);x=torch.randn(32,16,device=device,generator=rng);t=torch.rand(32,1,device=device,generator=rng)
            inputs,outputs=capture(model,x,t)
            for name,layer in model.ode_layers():
                h,context=inputs[name]
                with torch.no_grad():d=layer.step_size(h,context)
                stats,_=summarize(layer.operator().detach().double().cpu().numpy(),h.double().cpu().numpy(),d.double().cpu().numpy(),layer.orders,outputs[name].double().cpu().numpy())
                tests.append(dict(family=family,layer=name,trace_error=stats['trace_energy_relative_error'],series_error=stats['series_energy_relative_error'],native_reconstruction_error=stats['native_layer_reconstruction_relative_rms_error']))
        # Nontrivial symmetric parameter/feature accounting identity.
        rng=np.random.default_rng(0);P0,P1,q0,q1=[rng.standard_normal((64,64)) for _ in range(4)];B0=P0.T@P0;B1=P1.T@P1;M0=q0.T@q0;M1=q1.T@q1
        left=np.trace(B1@M1)-np.trace(B0@M0);right=np.trace((B1-B0)@(M1+M0)/2)+np.trace((B1+B0)@(M1-M0)/2)
        assert abs(left-right)/max(abs(left),1.)<1e-10
        base.dump(out,dict(passed=True,device=a.device,torch=torch.__version__,hooked_prediction_exact=True,layers=tests,symmetric_decomposition_relative_error=float(abs(left-right)/max(abs(left),1.)),analysis_source_sha256=base.digest(__file__),cases_sha256=base.digest(cases_file)));return
    assert a.host in ['145','lenovo'] and a.cycle8_output and a.cycle6_output
    rows=[];bindings=[];arrays={};out.mkdir(parents=True,exist_ok=False)
    for case in [c for c in cases if c['host']==a.host]:
        root=Path(a.cycle8_output if case['group']=='cycle8' else a.cycle6_output);folder=root/case['experiment_id']/case['dataset']/case['id']/'seed000';cfg=json.loads((folder/'config.json').read_text());checkpoint=folder/f'checkpoints/step_{case["step"]:07d}.pt'
        assert (folder/'ADAPTER_COMPLETE').exists() and base.digest(checkpoint)==case['checkpoint_sha256'] and base.digest(folder/'config.json')==case['config_sha256']
        state=torch.load(checkpoint,map_location=device,weights_only=False);manifest_file=Path(cfg['data_root'])/'manifest.json'
        assert state['step']==cfg['steps']==case['step'] and state['config_sha256']==case['config_sha256'] and state['dataset_manifest_sha256']==base.digest(manifest_file)==case['manifest_sha256']
        assert cfg['cell']['family']==case['family'] and cfg['cell']['dataset']==case['dataset'] and cfg['seeds']==[0]
        base.verify_data(Path(cfg['data_root']).parent,cfg['cell'])
        spec=json.loads(manifest_file.read_text())['datasets'][case['dataset']]
        # Verbatim native fixed_eval_data probe slicing, without unrelated sampling.
        with np.load(Path(cfg['data_root'])/case['dataset']/'probe.npz',allow_pickle=False) as probe:
            px=probe['xt'][:256].copy();pt=probe['t'][:256].copy()
        assert len(px)==256
        probe_hash=hashlib.sha256(px.tobytes()+pt.tobytes()).hexdigest()
        torch.manual_seed(0);model=build_model(spec['dim'],case['family']).to(device);base.train.restore_model(model,state);model.eval()
        inputs,outputs=capture(model,torch.from_numpy(px).to(device),torch.from_numpy(pt).to(device))
        native=json.loads((folder/'metrics.jsonl').read_text().splitlines()[-1]);archived={r['layer']:r for r in native['ode_layers']};assert native['step']==case['step'] and len(inputs)==len(archived)==6
        payload={};local=[]
        for index,(name,layer) in enumerate(model.ode_layers()):
            assert layer.spectral_cap is None and layer.operator_mode=='multiplicative' and layer.forcing_mode=='none' and layer.solve_return=='map'
            h,context=inputs[name]
            with torch.no_grad():duration=layer.step_size(h,context)
            stats,raw=summarize(layer.operator().detach().double().cpu().numpy(),h.double().cpu().numpy(),duration.double().cpu().numpy(),layer.orders,outputs[name].double().cpu().numpy())
            expected=archived[name]['activation']['term1_rms_over_input'];error=abs(stats['first_term_rms_over_input']-expected)/max(abs(expected),1e-30);assert error<5e-6
            stats.update(cell=case['id'],dataset=case['dataset'],family=case['family'],relaxation=case['relaxation'],step=case['step'],layer=name,index=index,orders=layer.orders,archived_term1_relative_error=error,probe_sha256=probe_hash)
            local.append(stats);arrays[(case['id'],name)]=raw
            payload.update({f'layer{index}_{key}':value for key,value in raw.items()})
        file=out/(case['id']+'.npz');np.savez_compressed(file,**payload)
        bindings.append(dict(case,probe_sha256=probe_hash,probe_count=256,time_min=float(pt.min()),time_max=float(pt.max()),distinct_times=len(np.unique(pt)),array_file=file.name,array_sha256=base.digest(file)))
        rows+=local;print(json.dumps(dict(cell=case['id'],layers=6,hooked_prediction_exact=True)),flush=True);del model,state
    pairs=[]
    for case in [c for c in cases if c['host']==a.host and c.get('reference_id')]:
        for layer in [r['layer'] for r in rows if r['cell']==case['id']]:
            x=arrays[(case['reference_id'],layer)];y=arrays[(case['id'],layer)];B0,B1,N0,N1=x['B'],y['B'],x['N'],y['N'];left=float(np.trace(B1@N1)-np.trace(B0@N0));parameter=float(np.trace((B1-B0)@(N1+N0)/2));feature=float(np.trace((B1+B0)@(N1-N0)/2));scale=max(abs(float(np.trace(B0@N0))),abs(float(np.trace(B1@N1))),1e-30);error=abs(left-parameter-feature)/scale;assert error<1e-10
            pairs.append(dict(cell=case['id'],reference=case['reference_id'],layer=layer,normalized_energy_change=left,parameter_accounting_term=parameter,feature_accounting_term=feature,relative_identity_error=error))
    assert len(bindings)==7 and len(rows)==42 and len(pairs)==30
    base.dump(out/'results.json',dict(passed=True,host=a.host,device=a.device,torch=torch.__version__,rows=rows,pairs=pairs,bindings=bindings,cases_sha256=base.digest(cases_file),analysis_source_sha256=base.digest(__file__),native_manifest_sha256=base.digest(source/'source_provenance.json'),scope='Frozen native probe accounting; no training, inverse moment, new metric selection or causal claim.'))

if __name__=='__main__':main()
