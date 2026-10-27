"""Independent checkpoint, stream, panel, P telemetry and manual Heun audit."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
a=ap.parse_args();root=a.root.resolve();out=a.output.resolve();out.mkdir(exist_ok=False)
sys.path.insert(0,str(root/'source'));import study

torch.set_num_threads(1);started=time.monotonic();study.verify(root,True)
assert study.load(root/'output/status.json')['state']=='TRAINING_COMPLETE'
original={str(p):study.sha(p) for p in (root/'output').rglob('*') if p.is_file()}
s=study.load(root/'settings.json');data=Path(s['data']);val=study.bank(root/'banks/validation.npz')
with np.load(data/'split.npz') as f:train=torch.from_numpy(f['train'])
rng=torch.Generator().manual_seed(100000);digest=hashlib.sha256(b'').hexdigest();digests={0:digest};states={0:rng.get_state().clone()}
for step in range(1,5001):
    ids=train[torch.randint(45000,(64,),generator=rng)]
    noise=torch.randn(64,3,32,32,generator=rng);time_tensor=torch.randn(64,generator=rng).mul(.8).sub(.8).sigmoid()
    h=hashlib.sha256(bytes.fromhex(digest))
    for value in (ids,noise,time_tensor):h.update(value.numpy().tobytes())
    digest=h.hexdigest()
    if step%500==0:digests[step]=digest;states[step]=rng.get_state().clone()
comparisons=[];curves=[];counts=dict(checkpoints=0,panels=0,P_records=0,clock_rows=0,sample_batches=0,stream_batches=5000)


def forward(model,b):
    result=[]
    with torch.no_grad():
        for start in range(0,len(b['x']),64):
            x=torch.from_numpy(b['x'][start:start+64]);t=torch.from_numpy(b['t'][start:start+64])
            result.append(model(x,t,torch.zeros(len(x),dtype=torch.int64)).numpy())
    return np.concatenate(result)
def calculate(pred,b):
    x=b['x'].astype(np.float64);target=b['target'].astype(np.float64);pred=pred.astype(np.float64)
    den=b['delta'].astype(np.float64)[:,None,None,None]
    # Explicit velocity difference, independent of producer's clean residual.
    vp=(pred-x)/den;vt=(target-x)/den
    return dict(weighted_mse=float(np.mean((vp-vt)**2)),clean_mse=float(np.mean((pred-target)**2)),prediction_rms=float(np.sqrt(np.mean(pred**2))))
def close(x,y):np.testing.assert_allclose(x,y,atol=1e-10,rtol=1e-8)
def sample(model,source):
    result=[];h=1./32
    with torch.no_grad():
        for offset in range(0,len(source),64):
            state=torch.from_numpy(source[offset:offset+64]).clone();label=torch.zeros(len(state),dtype=torch.long)
            for k in range(32):
                time0=torch.full((len(state),),k*h);time1=torch.full_like(time0,(k+1)*h)
                v0=(model(state,time0,label)-state)/(1-time0[:,None,None,None]).clamp_min(.05)
                proposed=state+h*v0
                v1=(model(proposed,time1,label)-proposed)/(1-time1[:,None,None,None]).clamp_min(.05)
                state=state+(.5*h)*(v0+v1)
            result.append(state.numpy())
    return np.concatenate(result)

for arm in study.ARMS:
    folder=root/'output/runs'/arm;model=study.model_for(root,arm);cold={k:v.clone() for k,v in model.state_dict().items()};cold_buffers={k:v.clone() for k,v in model.named_buffers()}
    cold_rng=torch.get_rng_state().clone();records=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in records]==list(range(0,5001,500))
    for step in range(0,5001,500):
        st=torch.load(folder/f'step_{step:07d}.pt',weights_only=False)
        assert st['step']==step and st['input_stream_digest']==digests[step]
        assert torch.equal(st['data_rng'],states[step]) and torch.equal(st['torch_rng'],cold_rng)
        if step==0:
            study.equal(st['model'],cold);study.equal(st['runtime_buffers'],cold_buffers);assert not st['optimizer']['state']
        else:
            ids=st['optimizer']['param_groups'][0]['params'];assert len(ids)==len(list(model.parameters()))
            for param,index in zip(model.parameters(),ids):
                if not param.requires_grad:assert index not in st['optimizer']['state'];continue
                moment=st['optimizer']['state'][index];assert int(moment['step'])==step
                for key in ('exp_avg','exp_avg_sq'):assert moment[key].shape==param.shape and torch.isfinite(moment[key]).all()
        study.restore_model(model,st);model.eval()
        for name,value in model.named_buffers():torch.testing.assert_close(value,st['runtime_buffers'][name],atol=0,rtol=0)
        pred=forward(model,val)
        with np.load(folder/f'validation_{step:07d}.npz') as f:np.testing.assert_array_equal(pred,f['prediction'])
        m=calculate(pred,val);stored=next(r for r in records if r['step']==step)
        for k,v in m.items():close(v,stored[k])
        curves.append(dict(family=arm,step=step,**m));counts['checkpoints']+=1
        if step not in study.SNAPSHOTS:continue
        p_records=study.load(folder/f'P_{step:07d}.json')
        for record in p_records:
            _,j,key=record['layer'].split('.');layer=getattr(model.blocks[int(j)].mlp,key)
            p=layer.proj.weight.detach().double().numpy();eig=np.linalg.eigvals(p)
            close(np.linalg.svd(p,compute_uv=False),record['singular_values']);close(np.linalg.norm(p),record['frobenius'])
            close(np.sort_complex(eig),np.sort_complex(np.array(record['eigenvalues_real'])+1j*np.array(record['eigenvalues_imag'])))
            counts['P_records']+=1
        saved_clocks=study.load(folder/f'clocks_{step:07d}.json')
        for clock in (.1,.3,.5,.7,.9):
            b=study.bank(root/'banks'/f'panel_{clock:.1f}.npz');captured=[];hooks=[]
            for j,block in enumerate(model.blocks):
                if not hasattr(block.mlp,'ode1'):continue
                for name in ('ode1','ode2'):
                    def hook(layer,inputs,key=f'blocks.{j}.{name}'):captured.append((key,layer,inputs[0].detach().clone()))
                    hooks.append(getattr(block.mlp,name).register_forward_pre_hook(hook))
            pred=forward(model,b)
            for hook in hooks:hook.remove()
            with np.load(folder/f'panel_{step:07d}_{clock:.1f}.npz') as f:np.testing.assert_array_equal(pred,f['prediction'])
            counts['panels']+=1
            for name,layer,value in captured:
                row=next(r for r in saved_clocks if r['layer']==name and r['time']==clock)
                flat=value.reshape(-1,64);np.testing.assert_array_equal(flat[:32].numpy(),np.array(row['probes'],dtype=np.float32))
                with torch.no_grad():
                    pre=torch.nn.functional.linear(flat,layer.t_func[0].weight,layer.t_func[0].bias)
                    logit=torch.nn.functional.linear(torch.nn.functional.silu(pre),layer.t_func[2].weight,layer.t_func[2].bias)
                    duration=torch.sigmoid(logit/10)*.8+.2;term=flat;total=flat;parts=[]
                    for k in range(layer.orders):term=torch.nn.functional.linear(term,layer.proj.weight)*duration/(k+1);parts.append(term);total=total+term
                    rms=lambda x:float(x.square().mean().sqrt())
                    values=dict(s_mean=float(duration.mean()),s_std=float(duration.std(unbiased=False)),s_min=float(duration.min()),s_max=float(duration.max()),
                        input_rms=rms(flat),output_rms=rms(total),term1_rms=rms(parts[0]),term2_rms=rms(parts[1]) if len(parts)>1 else 0.)
                for key,v in values.items():close(v,row[key])
                counts['clock_rows']+=1
    generated=sample(model,np.load(root/'banks/sampling-source.npy'))
    np.testing.assert_array_equal(generated,np.load(folder/'generated.npy'));counts['sample_batches']+=1
    result=study.load(folder/'result.json');final=calculate(forward(model,val),val)
    for key,v in final.items():close(v,result[key])
    close(float(np.mean(np.abs(generated)>1)),result['out_of_range_fraction'])
    comparisons.append(dict(family=arm,**final,out_of_range_fraction=result['out_of_range_fraction'],cell_seconds=result['cell_seconds']))
    print(json.dumps(dict(audited_family=arm,**counts,elapsed=time.monotonic()-started)),flush=True)
assert original=={str(p):study.sha(p) for p in (root/'output').rglob('*') if p.is_file()}
study.verify(root,True)
for name,rows in [('comparison',comparisons),('curves',curves)]:
    with (out/f'{name}.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
study.write(out/'audit.json',dict(status='PASS',**counts,formal_files_unchanged=len(original),seconds=time.monotonic()-started,
    scope='Checkpoint predictions/buffers/moments/RNG/stream and literal telemetry/sample replay; train-window losses are retained raw logs, not full retraining.'))
print(json.dumps(study.load(out/'audit.json')),flush=True)
