"""Full stream/teacher replay and independent checkpoint/endpoint audit."""
import argparse,hashlib,os,time
from pathlib import Path
import numpy as np
import psutil
import torch
from teacher_support import TEACHERS,STUDENTS,TIMES,SNAPSHOTS,OLD,BANKS,DATASET,load,write,sha,setup,teacher,student,signature,targets,verify_cold,assert_equal
from explicit_forward import forward

def hash_arrays(previous,*arrays):
    h=hashlib.sha256(bytes.fromhex(previous))
    for a in arrays:
        a=np.ascontiguousarray(a);h.update(str((a.shape,str(a.dtype))).encode('ascii'));h.update(a.tobytes())
    return h.hexdigest()
def manual_values(model,x,t,family):
    out=[]
    with torch.no_grad():
        for i in range(0,len(x),1024):out.append(forward(model,torch.from_numpy(x[i:i+1024].copy()),torch.from_numpy(t[i:i+1024].copy()),family).numpy())
    return np.concatenate(out)
def native_values(model,x,t):
    out=[]
    with torch.no_grad():
        for i in range(0,len(x),1024):out.append(model(torch.from_numpy(x[i:i+1024].copy()),torch.from_numpy(t[i:i+1024].copy())).numpy())
    return np.concatenate(out)
def independent_jac(model,x,t,family):
    x=torch.from_numpy(x.copy()).double().requires_grad_(True);t=torch.from_numpy(t.copy()).double()
    value=forward(model,x,t,family);jac=torch.stack([torch.autograd.grad(value[:,i].sum(),x,retain_graph=i<15)[0] for i in range(16)],1)
    return value.detach().numpy(),jac.detach().numpy()

def main(root):
    torch.set_num_threads(1);setup();started=time.monotonic();out=root/'output';assert load(out/'status.json')['state']=='COMPLETE'
    audit_source=Path(__file__).parent
    for name,digest in load(audit_source/'audit-source-sha256.json').items():assert sha(audit_source/name)==digest,name
    dest=root/'independent-audit';dest.mkdir(exist_ok=False);peak=0.
    def progress(**extra):
        nonlocal peak
        rss=psutil.Process().memory_info().rss/2**20;peak=max(peak,rss)
        d=dict(pid=os.getpid(),elapsed_seconds=time.monotonic()-started,rss_mib=rss,peak_sampled_rss_mib=peak,**extra)
        write(dest/'status.json',dict(state='RUNNING',**d));print(d,flush=True)
    bindings=load(out/'input-bindings.json')
    for path,digest in bindings.items():assert sha(path)==digest,path
    source=root/'source'
    for name,digest in load(source/'source-sha256.json').items():assert sha(source/name)==digest,name
    cache=out/'cache';meta=load(cache/'metadata.json')
    for name,digest in meta['files_sha256'].items():assert sha(cache/name)==digest,name
    maps={name:np.load(cache/(name+'.npy'),mmap_mode='r') for name in ['x','t']+['label-'+t for t in TEACHERS]}
    teachers={};teacher_sigs={}
    for name in TEACHERS:teachers[name],_=teacher(name);teacher_sigs[name]=signature(teachers[name]);assert teacher_sigs[name]==meta['teacher_signatures'][name]
    bank,_=targets();rng=torch.Generator().manual_seed(100000);source_states=torch.load(cache/'source-input-rng.pt',map_location='cpu',weights_only=False)
    empty=hashlib.sha256(b'').hexdigest();original_digest=empty;input_digest=empty;label_digest={t:empty for t in TEACHERS};replay_label={t:empty for t in TEACHERS}
    from synthetic.population import fingerprint_batch
    for i in range(5000):
        target=bank[torch.randint(len(bank),(512,),generator=rng)];noise=torch.randn(target.shape,generator=rng);t=torch.rand((512,1),generator=rng);x=(1-t)*noise+t*target
        np.testing.assert_array_equal(x.numpy(),maps['x'][i]);np.testing.assert_array_equal(t.numpy(),maps['t'][i])
        original_digest=fingerprint_batch(original_digest,target,noise,t);input_digest=hash_arrays(input_digest,x.numpy(),t.numpy())
        for name,m in teachers.items():
            with torch.no_grad():label=m(x,t).numpy()
            np.testing.assert_array_equal(label,maps['label-'+name][i]);h=hashlib.sha256(bytes.fromhex(label_digest[name]));h.update(label.tobytes());label_digest[name]=h.hexdigest()
            replay_label[name]=hash_arrays(replay_label[name],label)
        if str(i+1) in meta['snapshots']:
            s=meta['snapshots'][str(i+1)];assert original_digest==s['input_digest'] and input_digest==s['input_replay_digest']
            assert label_digest==s['label_digests'] and replay_label==s['label_replay_digests'];assert torch.equal(rng.get_state(),source_states[i+1])
        if (i+1)%1000==0:progress(phase='all cached training inputs and teacher labels',step=i+1)
    for name in TEACHERS:
        checkpoint=torch.load(meta['teachers'][name],map_location='cpu',weights_only=False)
        assert original_digest==checkpoint['input_stream_digest'] and torch.equal(rng.get_state(),checkpoint['data_rng'])
        assert signature(teachers[name])==teacher_sigs[name] and all(p.grad is None for p in teachers[name].parameters())
    with np.load(BANKS/'field.npz') as bankfield:eval_x,eval_t,oracle=bankfield['x'],bankfield['t'],bankfield['oracle'].astype(np.float64)
    teacher_values={};teacher_panels={}
    from synthetic.train import restore_model
    from scripts.check_gate_geometry import jacobian
    for name,m in teachers.items():
        actual=manual_values(m,eval_x,eval_t,name)
        with np.load(cache/'evaluation'/name/'field.npz') as saved:np.testing.assert_array_equal(actual,saved['prediction']);teacher_values[name]=saved['prediction'].astype(np.float64)
        exact=student(name);restore_model(exact,torch.load(meta['teachers'][name],map_location='cpu',weights_only=False));exact.eval()
        np.testing.assert_array_equal(native_values(exact,eval_x,eval_t),actual)
        teacher_panels[name]={}
        for clock in TIMES:
            with np.load(cache/'evaluation'/name/f'panel_{clock:.1f}.npz') as saved:teacher_panels[name][clock]={k:saved[k] for k in saved.files}
            with np.load(BANKS/f'panel_{clock:.1f}.npz') as bank_panel:xx,tt=bank_panel['x'],bank_panel['t']
            native_value,native_jac=jacobian(m,torch.from_numpy(xx),torch.from_numpy(tt))
            np.testing.assert_array_equal(native_value.numpy(),teacher_panels[name][clock]['prediction']);np.testing.assert_array_equal(native_jac.numpy(),teacher_panels[name][clock]['jacobian'])
    results=[];checkpoint_count=0;field_replays=0;panel_replays=0;matrix_records=[];max_double_value=0.;max_double_jac=0.;curves=[]
    for name in TEACHERS:
        for family in STUDENTS:
            progress(phase='checkpoint and explicit endpoint replay',teacher=name,student=family)
            folder=out/'runs'/(name+'--'+family);assert load(folder/'status.json')['state']=='COMPLETE';result=load(folder/'result.json');provenance=load(folder/'provenance.json')
            assert provenance['seed']==0 and provenance['threads']==1 and provenance['cache_metadata_sha256']==sha(cache/'metadata.json')
            assert result['actual_replay_digests']==dict(input=input_digest,label=replay_label[name])
            assert result['input_stream_digest']==original_digest and result['teacher_label_digest']==label_digest[name]
            cold=student(family);verify_cold(cold,family);cold_buffers={k:v.detach().clone() for k,v in cold.named_buffers()}
            for step in SNAPSHOTS:
                state=torch.load(folder/f'step_{step:07d}.pt',map_location='cpu',weights_only=False);assert state['step']==step and state['cache_position']==step
                stream=meta['snapshots'][str(step)];assert state['input_stream_digest']==stream['input_digest'] and state['teacher_label_digest']==stream['label_digests'][name]
                assert state['actual_replay_digests']==dict(input=stream['input_replay_digest'],label=stream['label_replay_digests'][name]);assert_equal(state['runtime_buffers'],cold_buffers)
                if step==0:assert_equal(state['model'],cold.state_dict());assert not state['optimizer']['state']
                else:
                    assert len(state['optimizer']['state'])==len(list(cold.parameters()))
                    assert all(int(v['step'])==step for v in state['optimizer']['state'].values())
                restore_model(cold,state);cold.eval();checkpoint_count+=1
                actual=manual_values(cold,eval_x,eval_t,family)
                with np.load(folder/f'field_{step:07d}.npz') as saved:np.testing.assert_array_equal(actual,saved['prediction']);field_replays+=1
                err=actual.astype(np.float64)-teacher_values[name];energy=float(np.mean(teacher_values[name]**2));risk=float(np.mean(err**2))
                curves.append(dict(teacher=name,student=family,step=step,teacher_mse=risk,relative_teacher_mse=risk/energy))
                if step==5000:
                    np.testing.assert_allclose([risk,risk/energy],[result['teacher_mse'],result['relative_teacher_mse']],atol=1e-14,rtol=1e-12)
                    assert signature(cold)==result['final_signature']
                spectra=load(folder/f'spectrum_{step:07d}.json')
                for layer_name,layer in cold.raw.ode_layers():
                    p=layer.proj.weight.detach().double().numpy();singular=np.linalg.svd(p,compute_uv=False);stored=spectra[layer_name]
                    np.testing.assert_allclose(singular,stored['singular_values'],atol=1e-12,rtol=1e-12)
                    # Compare eigenvalue multisets by polynomial invariants,
                    # avoiding an arbitrary ordering of conjugate pairs.
                    eig=np.array(stored['eigenvalue_real'])+1j*np.array(stored['eigenvalue_imag'])
                    for power in [1,2,3]:np.testing.assert_allclose(np.sum(eig**power),np.trace(np.linalg.matrix_power(p,power)),atol=1e-10,rtol=1e-10)
                    matrix_records.append(dict(teacher=name,student=family,step=step,layer=layer_name,spectral_norm=float(singular[0]),frobenius_norm=float(np.linalg.norm(p)),
                        symmetric_minimum=float(np.linalg.eigvalsh((p+p.T)*.5)[0])))
            state=torch.load(folder/'step_0005000.pt',map_location='cpu',weights_only=False);cold=student(family);restore_model(cold,state);cold.eval()
            for parameter in cold.parameters():parameter.requires_grad_(False)
            native_double=student(family);restore_model(native_double,state);native_double=native_double.double().eval()
            for parameter in native_double.parameters():parameter.requires_grad_(False)
            for clock in TIMES:
                with np.load(BANKS/f'panel_{clock:.1f}.npz') as b:x,t=b['x'],b['t']
                native_value,native_jac=jacobian(cold,torch.from_numpy(x),torch.from_numpy(t))
                with np.load(folder/f'panel_{clock:.1f}.npz') as saved:np.testing.assert_array_equal(native_value.numpy(),saved['prediction']);np.testing.assert_array_equal(native_jac.numpy(),saved['jacobian'])
                tp=teacher_panels[name][clock];risk=float(np.mean((native_value.numpy().astype(np.float64)-tp['prediction'].astype(np.float64))**2))
                je=float(np.mean(np.sum((native_jac.numpy().astype(np.float64)-tp['jacobian'].astype(np.float64))**2,axis=(1,2))))
                row=next(r for r in result['panels'] if r['time']==clock);np.testing.assert_allclose([risk,je],[row['teacher_mse'],row['jacobian_error']],atol=1e-14,rtol=1e-12)
                manual_value,manual_jac=independent_jac(native_double,x,t,family)
                expected_value,expected_jac=jacobian(native_double,torch.from_numpy(x).double(),torch.from_numpy(t).double())
                np.testing.assert_allclose(manual_value,expected_value.numpy(),atol=1e-10,rtol=1e-10);np.testing.assert_allclose(manual_jac,expected_jac.numpy(),atol=1e-10,rtol=1e-10)
                max_double_value=max(max_double_value,float(np.max(np.abs(manual_value-expected_value.numpy()))));max_double_jac=max(max_double_jac,float(np.max(np.abs(manual_jac-expected_jac.numpy()))));panel_replays+=1
            results.append(result)
    original=next(r for r in results if r['teacher']=='ode_k2' and r['student']=='ode_k2');generic=[r for r in results if r['teacher']=='ode_k2' and r['student']!='ode_k2']
    pattern=original['relative_teacher_mse']>.05 and all(r['relative_teacher_mse']<=.05 and r['teacher_mse']<=.5*original['teacher_mse'] and
        r['mean_panel_teacher_mse']<=.5*original['mean_panel_teacher_mse'] and r['mean_panel_jacobian_error']<=original['mean_panel_jacobian_error'] for r in generic)
    decision=dict(registered_learning_difficulty_pattern=bool(pattern),architecture_selected=False,ordinary_FM_confirmation=False,
        interpretation='finite-budget ODE fitting difficulty on a realizable ODE target' if pattern else 'registered learning-difficulty pattern not established')
    for path,digest in bindings.items():assert sha(path)==digest,path
    for name,digest in meta['files_sha256'].items():assert sha(cache/name)==digest,name
    for name,digest in load(source/'source-sha256.json').items():assert sha(source/name)==digest,name
    for name,digest in load(audit_source/'audit-source-sha256.json').items():assert sha(audit_source/name)==digest,name
    write(dest/'results.json',results);write(dest/'learning-curves.json',curves);write(dest/'P-trajectories.json',matrix_records);write(dest/'decision.json',decision)
    result=dict(status='PASS',pid=os.getpid(),elapsed_seconds=time.monotonic()-started,peak_sampled_rss_mib=peak,training_cache_steps_replayed=5000,
        live_teacher_label_batches_bitwise=10000,checkpoints_verified=checkpoint_count,complete_saved_fields_bitwise=field_replays,native_endpoint_panels_bitwise=panel_replays,
        independent_double_forward_max_error=max_double_value,independent_double_jacobian_max_error=max_double_jac,P_checkpoint_records=len(matrix_records),decision=decision)
    write(dest/'audit.json',result);write(dest/'status.json',dict(state='COMPLETE',**result));print(result,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
