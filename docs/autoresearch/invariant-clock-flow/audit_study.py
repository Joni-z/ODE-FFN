"""All-checkpoint and sampler replay plus alternative complete-Jacobian checks."""
import copy,json,time
from pathlib import Path
import numpy as np
import torch
import twist_study as g
from manual_jacobian import network_jacobian

def prediction(model,x,t):
    model.eval()
    with torch.no_grad():return np.concatenate([model(torch.from_numpy(x[i:i+1024]),torch.from_numpy(t[i:i+1024])).numpy() for i in range(0,len(x),1024)])

def heun(model,source):
    x=torch.from_numpy(source.copy());dt=1/32;model.eval()
    with torch.no_grad():
        for i in range(32):
            t=torch.full((len(x),1),i*dt,dtype=x.dtype);velocity=model(x,t);trial=x+dt*velocity
            next_t=torch.full_like(t,(i+1)*dt);x=x+0.5*dt*(velocity+model(trial,next_t))
    return x.numpy()

def audit(root):
    _,n=g.setup(root);g.verify(root);assert g.load(root/'output/status.json')['state']=='COMPLETE'
    out=root/'audit';out.mkdir();tick=time.monotonic();new=[];historical=[];checkpoints=panels=anchors=matrices=0;chain_max=0.
    immutable={str(p):g.sha(p) for p in (root/'output').rglob('*') if p.is_file()}
    for e in g.load(root/'settings.json')['laws']:
        law=e['law'];banks=Path(e['banks']);bank=g.train_bank(e,n);rng=torch.Generator().manual_seed(100000);digest=n.EMPTY;stream={0:(digest,rng.get_state().clone())}
        for step in range(1,5001):
            _,t,a,b=n.draw(bank,rng);digest=n.legacy.fingerprint_batch(digest,a,b,t)
            if step in g.STEPS:stream[step]=(digest,rng.get_state().clone())
        g.write(out/f'{law}-stream.json',dict(steps=5000,digests={str(k):v[0] for k,v in stream.items()}))
        with np.load(banks/'field.npz') as f:field={k:f[k] for k in f.files}
        for arm in g.ARMS:
            folder=root/'output/runs'/law/arm;result=g.load(folder/'result.json');assert g.load(folder/'status.json')['state']=='COMPLETE'
            assert g.load(folder/'draw-update-timing.json')['update_calls']==5000
            for step in g.STEPS:
                state=torch.load(folder/f'step_{step:07d}.pt',weights_only=False)
                assert state['step']==step and state['input_stream_digest']==stream[step][0] and torch.equal(state['data_rng'],stream[step][1])
                for value in state['optimizer']['state'].values():assert int(value['step'])==step
                model=g.build(root,law,arm);n.restore_model(model,state);actual=prediction(model,field['x'],field['t'])
                with np.load(folder/f'field_{step:07d}.npz') as f:assert actual.tobytes()==f['prediction'].tobytes()
                saved=g.load(folder/f'generator_{step:07d}.json');values,kinds=g.matrix_values(model)
                with np.load(folder/f'matrices_{step:07d}.npz') as f:
                    assert set(f.files)==set(values)
                    for name,v in values.items():assert v.tobytes()==f[name].tobytes()
                for row in saved['matrices']:
                    a=values[row['layer']].astype(np.float64);singular=np.linalg.svd(a,compute_uv=False);eigen=np.linalg.eigvals(a)
                    wanted=(float(singular.max()),float(singular.min()),float(eigen.real.max()),float(np.abs(eigen).max()))
                    assert wanted==tuple(row[k] for k in ('sigma_max','sigma_min','eigen_max_real','eigen_max_absolute'))
                    assert row['singular_CV']==(float(singular.std()/singular.mean()) if singular.mean()>0 else None)
                    assert row['kind']==kinds[row['layer']];matrices+=1
                checkpoints+=1
            with np.load(folder/'diagnostics/field.npz') as f:assert actual.tobytes()==f['prediction'].tobytes()
            oracle=float(np.mean((actual-field['oracle'])**2));assert oracle==result['oracle_mse']
            errors=[];promoted=copy.deepcopy(model).double().eval()
            for time_value in (.1,.3,.5,.7,.9):
                with np.load(banks/f'panel_{time_value:.1f}.npz') as f:panel={k:f[k] for k in f.files}
                p,j=n.legacy.jacobian(model,torch.from_numpy(panel['x']),torch.from_numpy(panel['t']));p,j=p.numpy(),j.numpy()
                with np.load(folder/f'diagnostics/panel_{time_value:.1f}.npz') as f:assert p.tobytes()==f['prediction'].tobytes() and j.tobytes()==f['jacobian'].tobytes()
                errors.append(float(np.mean(np.sum((j-panel['oracle_jacobian'])**2,axis=(1,2)))))
                for i in range(2):
                    x=torch.from_numpy(panel['x'][i]).double();t=torch.from_numpy(panel['t'][i]).double()
                    explicit=network_jacobian(promoted,x,t)
                    automatic=torch.autograd.functional.jacobian(lambda value:promoted(value[None],t[None])[0],x,vectorize=True)
                    torch.testing.assert_close(explicit,automatic,rtol=3e-9,atol=3e-9)
                    chain_max=max(chain_max,float((explicit-automatic).abs().max()));anchors+=1
                panels+=1
            assert float(np.mean(errors))==result['jacobian_error']
            with np.load(banks/'sampling.npz') as f:sample={k:f[k] for k in f.files}
            generated=heun(model,sample['source'])
            with np.load(folder/'diagnostics/samples.npz') as f:assert generated.tobytes()==f['generated'].tobytes()
            # Independent metric expression with native float32 projection arithmetic.
            left=torch.from_numpy(generated)@torch.from_numpy(sample['directions']).T;right=torch.from_numpy(sample['target'])@torch.from_numpy(sample['directions']).T
            sw=float(torch.sqrt((left.sort(0).values-right.sort(0).values).square().mean()))
            assert sw==result['sw2'];new.append(dict(law=law,arm=arm,oracle_mse=oracle,jacobian_error=result['jacobian_error'],sw2=sw))
            g.write(out/'progress.json',dict(new_endpoints=len(new),historical_endpoints=len(historical),checkpoints=checkpoints,panels=panels,manual_anchors=anchors))
        for family in g.HISTORY:
            model,folder,_=g.historical(root,law,family);actual=prediction(model,field['x'],field['t']);truth=g.load(folder/'diagnostics/metrics.json')
            with np.load(folder/'diagnostics/field.npz') as f:assert actual.tobytes()==f['prediction'].tobytes()
            assert float(np.mean((actual-field['oracle'])**2))==truth['oracle_mse']
            for tv in (.1,.3,.5,.7,.9):
                with np.load(banks/f'panel_{tv:.1f}.npz') as f:p,j=n.legacy.jacobian(model,torch.from_numpy(f['x']),torch.from_numpy(f['t']))
                with np.load(folder/f'diagnostics/panel_{tv:.1f}.npz') as f:assert p.numpy().tobytes()==f['prediction'].tobytes() and j.numpy().tobytes()==f['jacobian'].tobytes()
            with np.load(banks/'sampling.npz') as f:generated=heun(model,f['source'])
            with np.load(folder/'diagnostics/samples.npz') as f:assert generated.tobytes()==f['generated'].tobytes()
            historical.append(dict(law=law,arm=family,oracle_mse=truth['oracle_mse'],jacobian_error=truth['jacobian_error'],sw2=truth['sw2'],**n.compute_budget(model)))
    assert (len(new),len(historical),checkpoints,panels,anchors,matrices)==(21,15,84,105,210,432)
    for path,digest in immutable.items():assert g.sha(path)==digest
    g.verify(root);g.write(out/'result.json',dict(status='PASS',new_endpoints=new,historical_endpoints=historical,new_checkpoint_predictions_bitwise=84,
        new_native_Jacobian_panels_bitwise=105,new_manual_Heun_batches_bitwise=21,reused_endpoint_replays=15,manual_double_chain_anchors=210,
        manual_chain_max_absolute_error=chain_max,matrix_summaries=432,full_optimizer_trajectory_rerun=False,formal_files_unchanged=len(immutable),seconds=time.monotonic()-tick))
    print(json.dumps(dict(status='PASS',new_endpoints=21,historical_endpoints=15,checkpoints=84,manual_chain_anchors=210)),flush=True)
