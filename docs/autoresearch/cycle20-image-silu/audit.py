"""Checkpoint/stream/matrix/Heun audit, with SciPy confined to FID phase."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,csv,hashlib,sys,time
from pathlib import Path
import numpy as np
import torch
import study

def same(a,b):assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def predict(model,b):
    rows=[]
    with torch.no_grad():
        for i in range(0,len(b['x']),64):rows.append(model(torch.from_numpy(b['x'][i:i+64]),torch.from_numpy(b['t'][i:i+64]),torch.zeros(min(64,len(b['x'])-i),dtype=torch.long)).numpy())
    return np.concatenate(rows)
def metrics(pred,b):
    x=b['x'].astype(np.float64);target=b['target'].astype(np.float64);pred=pred.astype(np.float64);delta=b['delta'].astype(np.float64)[:,None,None,None]
    vp=(pred-x)/delta;vt=(target-x)/delta
    return dict(weighted_mse=float(np.mean((vp-vt)**2)),clean_mse=float(np.mean((pred-target)**2)),prediction_rms=float(np.sqrt(np.mean(pred**2))))
def heun(model,source):
    result=[];h=1./128
    with torch.no_grad():
        for offset in range(0,len(source),64):
            state=torch.from_numpy(np.array(source[offset:offset+64]));labels=torch.zeros(len(state),dtype=torch.long)
            for i in range(128):
                t=torch.full((len(state),),i*h);nt=torch.full_like(t,(i+1)*h)
                a=(model(state,t,labels)-state)/(1-t[:,None,None,None]).clamp_min(.05);trial=state+h*a
                b=(model(trial,nt,labels)-trial)/(1-nt[:,None,None,None]).clamp_min(.05);state=state+.5*h*(a+b)
            result.append(state.numpy())
    return np.concatenate(result)
def independent_fid(x,y):
    import scipy.linalg
    x=x.astype(np.float64);y=y.astype(np.float64);a=x-x.mean(0);b=y-y.mean(0);sv=scipy.linalg.svd(a@b.T,compute_uv=False,lapack_driver='gesvd')
    return float(np.sum((x.mean(0)-y.mean(0))**2)+np.sum(a*a)/(len(a)-1)+np.sum(b*b)/(len(b)-1)-2*sv.sum()/np.sqrt((len(a)-1)*(len(b)-1)))

def training(root,out):
    old=study.imported(root,'image_parent');p=study.paths(root)['image_parent'];data=Path(old.load(p/'settings.json')['data'])
    with np.load(data/'split.npz') as f:indices=torch.from_numpy(f['train'])
    rng=torch.Generator().manual_seed(100000);digest=hashlib.sha256(b'').hexdigest();states={0:rng.get_state().clone()};digests={0:digest}
    for step in range(1,5001):
        ids=indices[torch.randint(45000,(64,),generator=rng)];noise=torch.randn(64,3,32,32,generator=rng);t=(.8*torch.randn(64,generator=rng)-.8).sigmoid()
        h=hashlib.sha256(bytes.fromhex(digest))
        for value in (ids,noise,t):h.update(value.numpy().tobytes())
        digest=h.hexdigest()
        if step%500==0:states[step]=rng.get_state().clone();digests[step]=digest
    # Native reference snapshots must see this same input stream at every step.
    for step in range(0,5001,500):
        st=torch.load(p/'output/runs/ode_k1'/f'step_{step:07d}.pt',weights_only=False)
        assert st['input_stream_digest']==digests[step] and torch.equal(st['data_rng'],states[step])
    val=old.bank(p/'banks/validation.npz');rows=[];curves=[];counts=dict(checkpoints=0,time_panels=0,matrix_records=0,generated_images=0,stream_batches=5000)
    for arm in study.ARMS:
        folder=root/'output/runs'/arm;model=study.model_for(root,arm);cold={k:v.clone() for k,v in model.state_dict().items()};buffers={k:v.clone() for k,v in model.named_buffers()};cold_rng=torch.get_rng_state().clone()
        import json
        records=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()];assert [r['step'] for r in records]==list(range(0,5001,500))
        for step in range(0,5001,500):
            st=torch.load(folder/f'step_{step:07d}.pt',weights_only=False)
            assert st['step']==step and st['input_stream_digest']==digests[step] and torch.equal(st['data_rng'],states[step]) and torch.equal(st['torch_rng'],cold_rng)
            if step==0:old.equal(st['model'],cold);old.equal(st['runtime_buffers'],buffers);assert not st['optimizer']['state']
            else:
                ids=st['optimizer']['param_groups'][0]['params'];assert len(ids)==len(list(model.parameters()))
                for parameter,index in zip(model.parameters(),ids):
                    if not parameter.requires_grad:assert index not in st['optimizer']['state'];continue
                    moment=st['optimizer']['state'][index];assert int(moment['step'])==step
                    for key in ('exp_avg','exp_avg_sq'):assert moment[key].shape==parameter.shape and torch.isfinite(moment[key]).all()
            old.restore_model(model,st);model.eval();pred=predict(model,val)
            with np.load(folder/f'validation_{step:07d}.npz') as saved:same(pred,saved['prediction'])
            m=metrics(pred,val);record=next(r for r in records if r['step']==step)
            for key,value in m.items():np.testing.assert_allclose(value,record[key],atol=1e-10,rtol=1e-8)
            curves.append(dict(family=arm,step=step,**m));counts['checkpoints']+=1
            if step not in study.SNAPSHOTS:continue
            history=study.load(folder/f'matrices_{step:07d}.json');assert len(history)==6
            for row in history:
                _,j,name=row['layer'].split('.');weight=getattr(model.blocks[int(j)].mlp,name).weight.detach().double().numpy();assert list(weight.shape)==row['shape']
                sv=np.linalg.svd(weight,compute_uv=False);np.testing.assert_allclose(sv,row['singular_values'],atol=1e-10,rtol=1e-8)
                np.testing.assert_allclose([np.linalg.norm(weight),sv[0],sv.std()/sv.mean()],[row['frobenius'],row['sigma_max'],row['singular_CV']],atol=1e-10,rtol=1e-8)
                if weight.shape[0]==weight.shape[1]:np.testing.assert_allclose(np.sort_complex(np.linalg.eigvals(weight)),np.sort_complex(np.asarray(row['eigen_real'])+1j*np.asarray(row['eigen_imag'])),atol=1e-10,rtol=1e-8)
                else:assert 'eigen_real' not in row and 'eigen_imag' not in row
                counts['matrix_records']+=1
            for value in (.1,.3,.5,.7,.9):
                b=old.bank(p/'banks'/f'panel_{value:.1f}.npz');prediction=predict(model,b)
                with np.load(folder/f'panel_{step:07d}_{value:.1f}.npz') as saved:same(prediction,saved['prediction'])
                counts['time_panels']+=1
        generated=heun(model,np.load(p/'banks/sampling-source.npy'));same(generated,np.load(folder/'generated.npy'));counts['generated_images']+=len(generated)
        final=metrics(predict(model,val),val);formal=study.load(folder/'result.json')
        for key,value in final.items():np.testing.assert_allclose(value,formal[key],atol=1e-10,rtol=1e-8)
        assert float(np.mean(np.abs(generated)>1))==formal['out_of_range_fraction'];rows.append(dict(family=arm,**final,out_of_range_fraction=formal['out_of_range_fraction'],seconds=formal['seconds']))
        print(__import__('json').dumps(dict(family=arm,**counts)),flush=True)
    for name,values in [('comparison',rows),('curves',curves)]:
        with (out/f'{name}.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    return counts

def features(root,out):
    p=study.paths(root)['image_parent'];sys.path.insert(0,str(p/'source'));import image_metrics as m
    from metric_core import uint8_images
    extractor=m.extractor(p);reference=m.features(extractor,np.load(p/'banks/fid-reference.uint8.npy'));same(reference,np.load(p/'metric-inputs/reference-features.npy'))
    rows=[];formal=study.load(root/'metric-output/result.json')['rows'];assert len(formal)==9
    for row in formal:
        arm=row['family'];saved=np.load(study.feature_path(root,arm))
        if arm in study.ARMS:
            replay=m.features(extractor,uint8_images(np.load(root/'output/runs'/arm/'generated.npy')));same(replay,saved)
        score=independent_fid(saved,reference);assert abs(score-float(row['fid']))<1e-8
        rows.append(dict(family=arm,fid=score,difference=score-float(row['fid']),feature_provenance='new bitwise replay' if arm in study.ARMS else 'sealed prior audit/bank'))
        print(__import__('json').dumps(rows[-1]),flush=True)
    return dict(rows=rows,new_generated_feature_rows=3072,reference_feature_rows=2048)

def main(root,phase):
    torch.set_num_threads(1);started=time.monotonic();study.verify(root);out=root/f'audit-{phase}';out.mkdir()
    folders=(root/'output',) if phase=='training' else (root/'output',root/'metric-output')
    original={str(f):study.sha(f) for folder in folders if folder.exists() for f in folder.rglob('*') if f.is_file()}
    result=training(root,out) if phase=='training' else features(root,out)
    for file,digest in original.items():assert study.sha(file)==digest,file
    study.verify(root);study.write(out/'result.json',dict(status='PASS',phase=phase,**result,formal_files_unchanged=len(original),seconds=time.monotonic()-started))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--phase',choices=('training','features'),required=True);a=ap.parse_args();main(a.root,a.phase)
