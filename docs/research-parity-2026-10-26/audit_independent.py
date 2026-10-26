"""Manual forwards and direct weighted QR in rotated latent coordinates."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import psutil
import torch
from runtime_helpers import ARMS,TIMES,PROBES,load,sha,setup,build
from manual_forward import manual

def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def independent_local(kind,a,b):
    half=lambda u,v:(u-v)*.5
    energy=lambda u:np.einsum('ij,ij->i',u,u)/u.shape[1]
    if kind=='clock':
        e=(a['out']+b['out'])*.5;o=half(a['out'],b['out'])
        return dict(even_value=e.mean(1),even_square=energy(e),odd=energy(o))
    if kind in ['silu','sigmoid']:
        center=(a['pre']+b['pre'])*.5;d=half(a['pre'],b['pre']);o=half(a['out'],b['out'])
        r=dict(center=energy(center),input_odd=energy(d),odd=energy(o))
        if kind=='silu':
            base=.5*d;extra=o-base;limit=.75*np.abs(center)*np.minimum(1.,np.abs(d))
            r.update(base=energy(base),added=energy(extra),bound=energy(limit),bound_violation=np.max(np.abs(extra)-limit,axis=1))
            assert r['bound_violation'].max()<1e-11
        return r
    du=half(a['gate'],b['gate']);dv=half(a['value'],b['value']);o=half(a['out'],b['out'])
    u=torch.from_numpy(du);v=torch.from_numpy(dv)
    activate=torch.sigmoid if kind=='product_glu' else torch.nn.functional.silu
    with torch.no_grad():intrinsic=((activate(u)*v-activate(-u)*(-v))*.5).numpy()
    return dict(odd=energy(o),intrinsic=energy(intrinsic),cross=np.einsum('ij,ij->i',o,intrinsic)/o.shape[1],center_effect=energy(o-intrinsic))

def main(root):
    torch.set_num_threads(1);setup();started=time.monotonic();out=root/'output';audit=root/'independent-audit';audit.mkdir(exist_ok=False)
    bindings=load(out/'input-bindings.json');source=Path(__file__).parent
    for path,digest in bindings.items():assert sha(path)==digest,path
    for name,digest in load(source/'source-sha256.json').items():assert sha(source/name)==digest
    proc=psutil.Process();peak=0.
    def progress(**extra):
        nonlocal peak
        rss=proc.memory_info().rss/2**20;peak=max(peak,rss)
        r=dict(pid=os.getpid(),elapsed_seconds=time.monotonic()-started,rss_mib=rss,peak_sampled_rss_mib=peak,**extra)
        write(audit/'status.json',dict(state='RUNNING',**r));print(json.dumps(r),flush=True)
    rotation=np.load(out/'rotation.npy');weights=np.load(out/'bootstrap-weights.npy');n=weights.shape[1]
    expected=np.vstack([np.ones(n,dtype=np.int64),np.random.default_rng(0).multinomial(n,np.full(n,1/n),size=256)])
    np.testing.assert_array_equal(weights,expected)
    rows=load(out/'local-summary.json');lookup={(r['arm'],r['time'],r['site']):r for r in rows}
    all_weighted=np.load(out/'weighted-local.npz');field_count=0;local_count=0;local_error=0.;weighted_error=0.;replayed_middle=[]
    for arm in ARMS:
        model,_=build(arm);model=model.double()
        for t in TIMES:
            progress(phase='manual forward and activation audit',arm=arm,time=t)
            with np.load(PROBES/'banks'/f'time-{t:.1f}.npz') as bank:x=bank['A']@rotation.T
            fp,ap,meta=manual(model,x,t,arm);fm,am,mm=manual(model,-x,t,arm);assert meta==mm
            with np.load(out/'fields'/arm/f'time-{t:.1f}.npz') as saved:
                np.testing.assert_array_equal(fp,saved['plus']);np.testing.assert_array_equal(fm,saved['minus']);field_count+=2
            with np.load(out/'local'/arm/f'time-{t:.1f}.npz') as saved:
                expected_keys=set()
                for site,info in meta.items():
                    record=lookup[arm,t,site];assert record['kind']==info['kind'] and record['width']==info['width']
                    local=independent_local(info['kind'],ap[site],am[site]);means={}
                    for name,value in local.items():
                        key=site+'__'+name;expected_keys.add(key)
                        np.testing.assert_allclose(value,saved[key],atol=1e-12,rtol=1e-12)
                        local_error=max(local_error,float(np.max(np.abs(value-saved[key]))));local_count+=1
                        if name!='bound_violation':
                            means[name]=np.average(value,weights=weights[0]);np.testing.assert_allclose(means[name],record[name],atol=1e-12,rtol=1e-12)
                            weighted=weights@value/n;wanted=all_weighted[f'{arm}__{t:.1f}__{site}__{name}']
                            np.testing.assert_allclose(weighted,wanted,atol=1e-12,rtol=1e-12)
                            weighted_error=max(weighted_error,float(np.max(np.abs(weighted-wanted))))
                    if arm=='ode_k2_direct' and site.endswith('.middle'):
                        eta=float(np.sqrt(means['added']/means['base']));zeta=float(np.sqrt(means['bound']/means['base']))
                        np.testing.assert_allclose([eta,zeta],[record['eta'],record['zeta']],atol=1e-12,rtol=1e-12)
                        replayed_middle.append((eta,zeta))
                assert expected_keys==set(saved.files)
            del ap,am
        del model
    all_weighted.close()
    numeric_errors={};all_stats=[];fit_count=0;min_rank=16
    names=['total','even','linear_odd','nonlinear_odd','nonlinear_alignment','nonlinear_energy_ratio','oracle_nonlinear_energy','coefficients']
    for t in TIMES:
        with np.load(PROBES/'banks'/f'time-{t:.1f}.npz') as bank:x=bank['A'];oracle=bank['oracle']
        plus=[];minus=[]
        for arm in ARMS:
            with np.load(out/'fields'/arm/f'time-{t:.1f}.npz') as f:plus.append(f['plus']@rotation);minus.append(f['minus']@rotation)
        plus=np.stack(plus);minus=np.stack(minus);odd=(plus-minus)*.5;even=(plus+minus)*.5;m=len(ARMS);d=x.shape[1]
        y=np.concatenate([odd.transpose(1,0,2).reshape(n,m*d),oracle],axis=1)
        with np.load(out/'projections'/f'time-{t:.1f}.npz') as saved:target={key:saved[key] for key in names}
        actual={key:np.empty_like(value) for key,value in target.items()}
        for j,w in enumerate(weights):
            if j%64==0:progress(phase='direct weighted QR',time=t,resample=j)
            sw=np.sqrt(w[:,None]);q,r=np.linalg.qr(x*sw,mode='reduced');rank=np.linalg.matrix_rank(r);min_rank=min(min_rank,int(rank));assert rank==16
            coeff=np.linalg.solve(r,q.T@(y*sw));grouped=coeff.reshape(d,m+1,d)
            fitted=x@coeff;residual=((y-fitted)*sw).reshape(n,m+1,d)
            f_res=residual[:,:m];v_res=residual[:,m];v_energy=np.sum(v_res*v_res)/(n*d)
            f_energy=np.sum(f_res*f_res,axis=(0,2))/(n*d);cross=np.sum(f_res*v_res[:,None,:],axis=(0,2))/(n*d)
            fit_group=fitted.reshape(n,m+1,d);fit_error=(fit_group[:,:m]-fit_group[:,m,None,:])*sw[:,None,:]
            actual['linear_odd'][j]=np.sum(fit_error*fit_error,axis=(0,2))/(n*d)
            difference=f_res-v_res[:,None,:];actual['nonlinear_odd'][j]=np.sum(difference*difference,axis=(0,2))/(n*d)
            actual['even'][j]=np.average(np.mean(even*even,axis=2),weights=w,axis=1)
            pair_error=np.mean(((plus-oracle[None])**2+(minus+oracle[None])**2)*.5,axis=2)
            actual['total'][j]=np.average(pair_error,weights=w,axis=1)
            actual['oracle_nonlinear_energy'][j]=v_energy
            actual['nonlinear_alignment'][j]=cross/v_energy;actual['nonlinear_energy_ratio'][j]=f_energy/v_energy
            physical=np.stack([rotation@grouped[:,k]@rotation.T for k in range(m+1)],axis=1).reshape(d,-1)
            actual['coefficients'][j]=physical;fit_count+=1
        for key in names:
            np.testing.assert_allclose(actual[key],target[key],atol=2e-11,rtol=2e-10,err_msg=f'{t}/{key}')
            numeric_errors[key]=max(numeric_errors.get(key,0.),float(np.max(np.abs(actual[key]-target[key]))))
        np.savez_compressed(audit/f'qr-time-{t:.1f}.npz',**actual)
        all_stats.append({key:actual[key] for key in ['total','even','linear_odd','nonlinear_odd']})
    aggregate={key:np.mean([s[key] for s in all_stats],axis=0) for key in all_stats[0]}
    with np.load(out/'aggregate.npz') as saved:
        for key,value in aggregate.items():np.testing.assert_allclose(value,saved[key],atol=2e-11,rtol=2e-10)
    decision=load(out/'decision.json');gap=aggregate['total'][:,0,None]-aggregate['total'][:,1:4];assert np.all(gap>0)
    actual_contrasts=[]
    for component in ['even','nonlinear_odd']:
        fraction=(aggregate[component][:,0,None]-aggregate[component][:,1:4])/gap
        low,high=np.percentile(fraction[1:],[2.5,97.5],axis=0)
        result='SUPPORTED_MAJORITY' if all(low>=.5) else 'REJECTED_MAJORITY' if all(high<.5) else 'MIXED_OR_INCONCLUSIVE'
        assert result==decision['explanation_decisions'][component]
        for j,arm in enumerate(ARMS[1:4]):
            wanted=next(r for r in decision['contrasts'] if r['component']==component and r['reference']==arm)
            np.testing.assert_allclose([fraction[0,j],low[j],high[j]],[wanted['signed_fraction'],wanted['low95'],wanted['high95']],atol=2e-11,rtol=2e-10)
            actual_contrasts.append(dict(component=component,reference=arm,signed_fraction=float(fraction[0,j]),low95=float(low[j]),high95=float(high[j])))
    small=all(a<=.1 for a,b in replayed_middle);certificate=small and all(b<=.1 for a,b in replayed_middle)
    assert ('UNIFORMLY_SMALL' if small else 'REJECTED_UNIFORMLY_SMALL')==decision['explanation_decisions']['middle_activation_added_odd']
    assert certificate==decision['explanation_decisions']['small_center_certificate'] and len(replayed_middle)==15
    for path,digest in bindings.items():assert sha(path)==digest,path
    for name,digest in load(source/'source-sha256.json').items():assert sha(source/name)==digest
    result=dict(status='PASS',pid=os.getpid(),elapsed_seconds=time.monotonic()-started,peak_sampled_rss_mib=peak,
        field_tensors_bitwise=field_count,local_row_statistics=local_count,local_max_abs_error=local_error,
        weighted_local_max_abs_error=weighted_error,weighted_QR_fits=fit_count,minimum_QR_rank=min_rank,
        maximum_projection_errors=numeric_errors,decisions_reproduced=True,contrasts=actual_contrasts,new_training_runs=0)
    write(audit/'audit.json',result);write(audit/'status.json',dict(state='COMPLETE',**result));print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
