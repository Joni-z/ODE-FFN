"""Independent checkpoint, stream, matrix, sampler and feature verification."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
import study


def same(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()


def predictions(model,bank):
    answers=[]
    with torch.no_grad():
        for start in range(0,len(bank['x']),64):
            x=torch.from_numpy(bank['x'][start:start+64]);t=torch.from_numpy(bank['t'][start:start+64])
            answers.append(model(x,t,torch.zeros(len(x),dtype=torch.long)).numpy())
    return np.concatenate(answers)


def fitting(pred,bank):
    d=pred.astype(np.float64)-bank['target'].astype(np.float64)
    weight=bank['delta'].astype(np.float64)[:,None,None,None]
    return dict(weighted_mse=float(np.mean((d/weight)**2)),clean_mse=float(np.mean(d*d)),
                prediction_rms=float(np.sqrt(np.mean(pred.astype(np.float64)**2))))


def manual_heun(model,source):
    # Independent expression already verified in cycle20; powers of two make
    # these multiplications equivalent to the producer's divisions in float32.
    answers=[];h=1./128
    with torch.no_grad():
        for offset in range(0,len(source),64):
            z=torch.from_numpy(np.array(source[offset:offset+64]));labels=torch.zeros(len(z),dtype=torch.long)
            for j in range(128):
                t=torch.full((len(z),),j*h);nt=torch.full_like(t,(j+1)*h)
                first=(model(z,t,labels)-z)/(1-t[:,None,None,None]).clamp_min(.05)
                trial=z+h*first
                second=(model(trial,nt,labels)-trial)/(1-nt[:,None,None,None]).clamp_min(.05)
                z=z+.5*h*(first+second)
            answers.append(z.numpy())
    return np.concatenate(answers)


def checkpoint(root,arm,step,deep=True):
    old=study.imported('native');folder=root/'output/runs'/arm
    state=torch.load(folder/f'step_{step:07d}.pt',weights_only=False,map_location='cpu')
    assert state['step']==step
    model=study.construct(arm);old.restore_model(model,state);model.eval()
    group=state['optimizer']['param_groups'];assert len(group)==1
    assert (group[0]['lr'],tuple(group[0]['betas']),group[0]['weight_decay'])==(1e-4,(.9,.99),0.)
    parameters=list(model.parameters());assert len(parameters)==len(group[0]['params'])
    for parameter,index in zip(parameters,group[0]['params']):
        if not parameter.requires_grad:
            assert index not in state['optimizer']['state'];continue
        moment=state['optimizer']['state'][index];assert int(moment['step'])==step
        for name in ('exp_avg','exp_avg_sq'):
            assert moment[name].shape==parameter.shape and torch.isfinite(moment[name]).all()
    source=study.initial(arm)
    assert torch.equal(state['torch_rng'],source['torch_rng'])
    if step==5000:old.equal(state,source)
    bank=old.bank(study.PARENTS['native']/'banks/validation.npz');pred=predictions(model,bank)
    with np.load(folder/f'validation_{step:07d}.npz') as saved:same(pred,saved['prediction'])
    metric=fitting(pred,bank);record=next(json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines() if json.loads(line)['step']==step)
    for name,value in metric.items():np.testing.assert_allclose(value,record[name],rtol=1e-8,atol=1e-10)
    result=dict(arm=arm,step=step,**metric,Adam_step=step,native_predictions_bitwise=True)
    if not deep:return result
    actual_weights=dict(model.named_parameters())
    history=study.load(folder/f'matrix-history-{step:07d}.json')
    for row in history:
        name=f"blocks.{row['block']}.mlp.{row['name']}";weight=actual_weights[name].detach().double().numpy()
        np.testing.assert_allclose(np.linalg.svd(weight,compute_uv=False),row['singular_values'],rtol=1e-8,atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(weight),row['frobenius'],rtol=1e-8,atol=1e-10)
    Ps=study.load(folder/f'P_{step:07d}.json')
    for row in Ps:
        name=row['layer'].replace('.ode','.mlp.ode')+'.proj.weight'
        weight=actual_weights[name].detach().double().numpy()
        np.testing.assert_allclose(np.linalg.svd(weight,compute_uv=False),row['singular_values'],rtol=1e-8,atol=1e-10)
        expected=np.asarray(row['eigenvalues_real'])+1j*np.asarray(row['eigenvalues_imag'])
        np.testing.assert_allclose(np.sort_complex(np.linalg.eigvals(weight)),np.sort_complex(expected),rtol=1e-8,atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(weight),row['frobenius'],rtol=1e-8,atol=1e-10)
    clock_rows=study.load(folder/f'clocks_{step:07d}.json');clock_count=0
    for value in (.1,.3,.5,.7,.9):
        panel=old.bank(study.PARENTS['native']/'banks'/f'panel_{value:.1f}.npz')
        captured={};hooks=[]
        for j,block in enumerate(model.blocks):
            if not hasattr(block.mlp,'ode1'):continue
            for key in ('ode1','ode2'):
                def hook(layer,args,name=f'blocks.{j}.{key}'):
                    captured[name]=(layer,args[0].detach().reshape(-1,64).numpy().copy())
                hooks.append(getattr(block.mlp,key).register_forward_pre_hook(hook))
        observed=predictions(model,panel)
        for hook in hooks:hook.remove()
        with np.load(folder/f'panel_{step:07d}_{value:.1f}.npz') as saved:same(observed,saved['prediction'])
        for name,(layer,values) in captured.items():
            row=next(r for r in clock_rows if r['layer']==name and r['time']==value)
            same(values[:32],np.asarray(row['probes'],dtype=np.float32))
            cast=lambda v:v.detach().double().numpy()
            x=values.astype(np.float64);hidden=x@cast(layer.t_func[0].weight).T+cast(layer.t_func[0].bias)
            hidden=hidden/(1+np.exp(-hidden));logit=hidden@cast(layer.t_func[2].weight).T+cast(layer.t_func[2].bias)
            s=.2+.8/(1+np.exp(-logit/10));p=cast(layer.proj.weight);first=(x@p.T)*s;second=(first@p.T)*s/2
            output=x+first+(second if layer.orders==2 else 0);rms=lambda v:float(np.sqrt(np.mean(v*v)))
            expected=dict(s_mean=s.mean(),s_std=s.std(),s_min=s.min(),s_max=s.max(),input_rms=rms(x),output_rms=rms(output),term1_rms=rms(first),term2_rms=rms(second) if layer.orders==2 else 0.)
            for key,v in expected.items():np.testing.assert_allclose(v,row[key],rtol=5e-5,atol=1e-5)
            clock_count+=1
    assert clock_count==len(clock_rows)
    result.update(matrix_records=len(history),P_records=len(Ps),clock_records=clock_count,panels=5)
    return result


def stream_states():
    first=study.initial('ode_k1');rng=torch.Generator();rng.set_state(first['data_rng'])
    data=Path(study.load(study.PARENTS['native']/'settings.json')['data'])
    with np.load(data/'split.npz') as bank:indices=torch.from_numpy(bank['train'])
    digest=first['input_stream_digest'];states={5000:(rng.get_state().clone(),digest)}
    import hashlib
    for step in range(5001,20001):
        ids=indices[torch.randint(len(indices),(64,),generator=rng)]
        noise=torch.randn((64,3,32,32),generator=rng);t=torch.sigmoid(.8*torch.randn(64,generator=rng)-.8)
        h=hashlib.sha256(bytes.fromhex(digest))
        for array in (ids,noise,t):h.update(array.numpy().tobytes())
        digest=h.hexdigest()
        if step%1000==0:states[step]=(rng.get_state().clone(),digest)
    return states


def training(root,out):
    summary=study.load(root/'output/summary.json');assert summary['state']=='COMPLETE' and len(summary['exits'])==7
    assert all(row['exit_code']==0 for row in summary['exits'])
    states=stream_states();rows=[];sampling=[]
    for arm in study.ARMS:
        folder=root/'output/runs'/arm
        records=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
        assert [r['step'] for r in records]==list(range(5000,20001,1000))
        for step in range(5000,20001,1000):
            rows.append(checkpoint(root,arm,step,deep=step in (5000,10000,15000,20000)))
            state=torch.load(folder/f'step_{step:07d}.pt',weights_only=False,map_location='cpu')
            assert torch.equal(state['data_rng'],states[step][0]) and state['input_stream_digest']==states[step][1]
        for step in (10000,20000):
            model=study.construct(arm);study.imported('native').restore_model(model,torch.load(folder/f'step_{step:07d}.pt',weights_only=False,map_location='cpu'));model.eval()
            generated=manual_heun(model,np.load(study.PARENTS['native']/'banks/sampling-source.npy'))
            same(generated,np.load(folder/f'generated-{step:07d}.npy'))
            sampling.append(dict(arm=arm,step=step,images=len(generated),bitwise=True))
        print(json.dumps(dict(arm=arm,verified_checkpoints=len(rows),verified_sampling_banks=len(sampling))),flush=True)
    return dict(rows=rows,sampling=sampling,independent_stream_batches=15000)


def feature_phase(root,out):
    sys.path.insert(0,str(study.PARENTS['native']/'source'))
    import image_metrics as metrics
    from metric_core import uint8_images
    import scipy.linalg
    producer=study.load(root/'metric-output/result.json');assert producer['status']=='PASS' and len(producer['rows'])==14
    assert {(r['arm'],r['step']) for r in producer['rows']}=={(a,s) for a in study.ARMS for s in (10000,20000)}
    extractor=metrics.extractor(study.PARENTS['native'])
    ref=metrics.features(extractor,np.load(study.PARENTS['native']/'banks/fid-reference.uint8.npy'))
    same(ref,np.load(study.PARENTS['native']/'metric-inputs/reference-features.npy'))
    rows=[]
    for formal in producer['rows']:
        arm,step=formal['arm'],formal['step'];raw_path=root/'output/runs'/arm/f'generated-{step:07d}.npy';raw=np.load(raw_path)
        feature_path=root/'metric-output'/f'{arm}-{step:07d}-features.npy'
        assert study.sha(raw_path)==formal['sample_sha256'] and study.sha(feature_path)==formal['feature_sha256']
        record=next(json.loads(line) for line in (root/'output/runs'/arm/'metrics.jsonl').read_text().splitlines() if json.loads(line)['step']==step)
        for key in ('weighted_mse','clean_mse','prediction_rms'):assert formal[key]==record[key]
        assert formal['out_of_range_fraction']==float(np.mean(np.abs(raw)>1))
        replay=metrics.features(extractor,uint8_images(raw));saved=np.load(root/'metric-output'/f'{arm}-{step:07d}-features.npy');same(replay,saved)
        a=saved.astype(np.float64);b=ref.astype(np.float64);ac=a-a.mean(0);bc=b-b.mean(0)
        singular=scipy.linalg.svd(ac@bc.T,compute_uv=False,lapack_driver='gesvd')
        value=float(np.sum((a.mean(0)-b.mean(0))**2)+np.sum(ac*ac)/(len(a)-1)+np.sum(bc*bc)/(len(b)-1)-2*singular.sum()/np.sqrt((len(a)-1)*(len(b)-1)))
        assert abs(value-formal['fid'])<1e-8
        rows.append(dict(arm=arm,step=step,fid=value,feature_rows=len(saved),feature_replay_bitwise=True,difference=value-formal['fid']))
        print(json.dumps(rows[-1]),flush=True)
    return dict(rows=rows,reference_features_bitwise=2048,generated_features_bitwise=14336)


def main(root,phase,arm=None,step=None,destination=None):
    torch.set_num_threads(1);study.verify(root);assert study.sha(study.__file__)==study.sha(root/'source/study.py')
    if phase!='snapshot':assert study.load(root/'metric-output/result.json')['status']=='PASS','Complete both production phases before the final audit'
    out=destination or root/('audit-snapshot' if phase=='snapshot' else 'audit-'+phase);out.mkdir(exist_ok=False)
    before={str(p):study.sha(p) for folder in (root/'output',root/'metric-output') if folder.exists() for p in folder.rglob('*') if p.is_file()} if phase!='snapshot' else {}
    started=time.monotonic()
    result=dict(rows=[checkpoint(root,arm,step)],scope='One checkpoint only; cohort audit incomplete') if phase=='snapshot' else training(root,out) if phase=='training' else feature_phase(root,out)
    for path,digest in before.items():assert study.sha(path)==digest,path
    study.verify(root);study.write(out/'result.json',dict(status='PASS',phase=phase,**result,seconds=time.monotonic()-started,formal_files_unchanged=len(before),auditor_source_sha256=study.sha(__file__)))
    print(json.dumps(dict(status='PASS',phase=phase,seconds=time.monotonic()-started)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=('snapshot','training','features'),required=True);p.add_argument('--arm',choices=study.ARMS);p.add_argument('--step',type=int);p.add_argument('--output',type=Path);a=p.parse_args();main(a.root,a.phase,a.arm,a.step,a.output)
