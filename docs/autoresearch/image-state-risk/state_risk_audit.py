"""Complete native prediction replay and direct-distance posterior audit."""
import time
import numpy as np
import torch
import state_risk_study as g
from bank_posterior import posterior

def audit(root):
    g.setup();g.verify(root);result=g.load(root/'output/result.json');assert result['status']=='COMPLETE'
    assert len(result['rows'])==100 and len({(v['arm'],v['domain'],v['index']) for v in result['rows']})==100
    assert len(result['reference_rows'])==20
    out=root/'audit';out.mkdir();tick=time.monotonic();counts=dict(reference_batches=0,prediction_batches=0,per_image_metrics=0);maximum=0.
    immutable={str(p):g.sha(p) for p in (root/'output').rglob('*') if p.is_file()};bank=g.train_bank();models={a:g.model_for(a) for a in g.ARMS}
    with np.load(root/'states.npz') as states,np.load(root/'targets.npz') as targets:
        for domain in g.DOMAINS:
            for j in g.INDICES:
                assert time.monotonic()-tick<1200
                key=f'{domain}_{j}';x=states[key];t=j/128
                independent=posterior(bank,torch.from_numpy(x.reshape(64,-1)).double(),t,direct=True,batch=23)
                with np.load(root/'output/posterior'/f'{key}.npz') as f:reference={k:f[k] for k in f.files}
                for k,value in reference.items():
                    if k=='top_index':assert np.array_equal(independent[k],value)
                    else:
                        np.testing.assert_allclose(independent[k],value,rtol=3e-9,atol=3e-9)
                        maximum=max(maximum,float(np.max(abs(independent[k]-value))))
                target=targets[domain].reshape(64,-1).astype(np.float64) if domain.startswith('paired_') else None
                rr=next(v for v in result['reference_rows'] if v['domain']==domain and v['index']==j)
                assert rr['effective_support_median']==float(np.median(reference['effective_support']))
                assert rr['entropy_mean']==float(np.mean(reference['entropy']))
                assert rr['top_weight_median']==float(np.median(reference['top_weight']))
                assert rr['log_density_per_dimension_mean']==float(np.mean(reference['log_density'])/3072)
                if target is not None:assert rr['paired_target_mse']==float(np.mean((reference['mean']-target)**2))
                counts['reference_batches']+=1
                for arm,model in models.items():
                    with torch.no_grad():p=model(torch.from_numpy(x),torch.full((64,),t),torch.zeros(64,dtype=torch.long)).numpy()
                    assert p.tobytes()==np.load(root/'output/predictions'/f'{arm}_{key}.npy').tobytes()
                    flat=p.reshape(64,-1).astype(np.float64);errors=np.square(flat-reference['mean']).mean(1)
                    values=dict(bank_reference_clean_mse=errors,bank_reference_velocity_mse=errors/(1-t)**2)
                    if target is not None:values['paired_target_mse']=np.square(flat-target).mean(1)
                    row=next(v for v in result['rows'] if v['arm']==arm and v['domain']==domain and v['index']==j)
                    with np.load(root/'output/per-image'/f'{arm}_{key}.npz') as f:
                        assert set(f.files)==set(values)
                        for k,v in values.items():assert np.array_equal(f[k],v);assert row[k]==float(np.mean(v));counts['per_image_metrics']+=len(v)
                    counts['prediction_batches']+=1
                g.write(out/'progress.json',dict(**counts,seconds=time.monotonic()-tick))
    assert counts==dict(reference_batches=20,prediction_batches=100,per_image_metrics=16000)
    for p,digest in immutable.items():assert g.sha(p)==digest
    g.verify(root);g.write(out/'result.json',dict(status='PASS',**counts,direct_distance_reference_max_absolute_error=maximum,formal_files_unchanged=len(immutable),seconds=time.monotonic()-tick))
    print('STATE_RISK_AUDIT_PASS',flush=True)
