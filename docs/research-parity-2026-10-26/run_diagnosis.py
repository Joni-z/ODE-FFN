"""Frozen-model antithetic diagnostics under the pre-outcome protocol."""
import argparse,json,os,time,traceback
from pathlib import Path
import numpy as np
import psutil
import torch
from runtime_helpers import ARMS,TIMES,MODELS,PROBES,GLU,DATASET,load,sha,setup,build,capture,check_alias,glu_checkpoint
from parity_math import local_stats,projection_stats,self_test

def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def classify(fractions):
    intervals=np.quantile(fractions[1:],[.025,.975],axis=0)
    state=('SUPPORTED_MAJORITY' if np.all(intervals[0]>=.5) else
           'REJECTED_MAJORITY' if np.all(intervals[1]<.5) else 'MIXED_OR_INCONCLUSIVE')
    return state,intervals

def main(root):
    torch.set_num_threads(1);started=time.monotonic();source=Path(__file__).parent
    for name,digest in load(source/'source-sha256.json').items():assert sha(source/name)==digest
    out=root/'output';out.mkdir(exist_ok=False);bindings={};native32=0;native64=0;positive=0;local_rows=[];weighted_local={};max_rss=0.
    def bind(p):bindings[str(Path(p).resolve())]=sha(p)
    def status(state='RUNNING',**more):
        nonlocal max_rss
        proc=psutil.Process();rss=proc.memory_info().rss/2**20;max_rss=max(max_rss,rss)
        record=dict(state=state,pid=os.getpid(),elapsed_seconds=time.monotonic()-started,rss_mib=rss,peak_sampled_rss_mib=max_rss,**more)
        write(out/'status.json',record)
        with (out/'telemetry.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    try:
        provenance=setup();alias=check_alias();tests=self_test()
        for p in [root/'preflight.json',root/'preflight-failure-1.json',source/'source-sha256.json',MODELS/'provenance.json',MODELS/'study.json',MODELS/'status.json',
                  PROBES/'population.json',PROBES/'standardized-draws.npz',PROBES/'audit.json',PROBES/'status.json',GLU/'configs/sigmoid_glu.json',GLU/'configs/swiglu.json',
                  GLU/'diagnostics'/DATASET/'sigmoid_glu/summary.json',Path(alias['historical_checkpoint'])]:bind(p)
        for name in provenance['source_sha256']:bind(Path(provenance['source_root'])/name)
        pop=load(PROBES/'population.json');rotation=np.array(pop['rotation']);mu=np.array(pop['locations']);variance=np.array(pop['variances'])
        np.testing.assert_allclose(rotation.T@rotation,np.eye(16),atol=2e-14)
        banks=[]
        for t in TIMES:
            path=PROBES/'banks'/f'time-{t:.1f}.npz';bind(path)
            with np.load(path) as d:bank={k:d[k] for k in d.files}
            den=(1-t)**2+t*t*variance
            truth=lambda q:(t*variance-(1-t))/den*q+(1-t)*mu/den*np.tanh(t*mu*q/den)
            np.testing.assert_allclose(truth(bank['A']),bank['oracle'],atol=1e-13,rtol=1e-13)
            np.testing.assert_array_equal(truth(-bank['A']),-truth(bank['A']))
            banks.append(bank)
        n=len(banks[0]['A']);rng=np.random.default_rng(0)
        weights=np.vstack([np.ones(n,dtype=np.int64),rng.multinomial(n,np.full(n,1/n),size=256)])
        np.save(out/'bootstrap-weights.npy',weights);np.save(out/'rotation.npy',rotation)
        projections=out/'projections';projections.mkdir();fields=out/'fields';fields.mkdir();locals_root=out/'local';locals_root.mkdir()
        for arm in ARMS:
            model,cp=build(arm);bind(cp);status(phase='restoring immutable model',arm=arm)
            if arm!='sigmoid_glu128':
                bind(cp.parent/'result.json')
                for t in TIMES:
                    p=MODELS/'banks'/DATASET/f'panel_{t:.1f}.npz';s=cp.parent/'diagnostics'/f'panel_{t:.1f}.npz';bind(p);bind(s)
                    with np.load(p) as bank,np.load(s) as saved:
                        with torch.enable_grad():value=model(torch.from_numpy(bank['x']).requires_grad_(True),torch.from_numpy(bank['t'])).detach().numpy()
                        np.testing.assert_array_equal(value,saved['prediction']);native32+=1
            model=model.double();(fields/arm).mkdir();(locals_root/arm).mkdir()
            if arm=='sigmoid_glu128':
                for t in TIMES:
                    p=GLU/'diagnostics'/DATASET/'sigmoid_glu'/f'common-{t:.1f}.npz';bind(p)
                    with np.load(p) as saved:
                        with torch.no_grad():actual=model(torch.from_numpy(saved['x']),torch.full((len(saved['x']),1),t,dtype=torch.float64)).numpy()
                        np.testing.assert_array_equal(actual,saved['prediction']);native64+=1
            for t,bank in zip(TIMES,banks):
                status(phase='paired field and activation extraction',arm=arm,time=t)
                x=bank['A']@rotation.T;target=bank['oracle']@rotation.T
                plus,ap,meta=capture(model,x,t,arm);minus,am,mm=capture(model,-x,t,arm);assert meta==mm
                if arm!='sigmoid_glu128':
                    path=PROBES/'fields'/arm/f'time-{t:.1f}.npz';bind(path)
                    with np.load(path) as saved:np.testing.assert_array_equal(plus@rotation,saved['native']);positive+=1
                assert np.isfinite(plus).all() and np.isfinite(minus).all()
                np.savez_compressed(fields/arm/f'time-{t:.1f}.npz',plus=plus,minus=minus,oracle=target)
                payload={}
                for site,info in meta.items():
                    stats=local_stats(info['kind'],ap[site],am[site]);row=dict(arm=arm,time=t,site=site,**info)
                    for name,values in stats.items():
                        payload[site+'__'+name]=values
                        if name=='bound_violation':row[name]=float(values.max())
                        else:
                            means=weights@values/n;weighted_local[f'{arm}__{t:.1f}__{site}__{name}']=means
                            row[name]=float(means[0])
                    if info['kind']=='silu':
                        row['eta']=float(np.sqrt(row['added']/row['base']));row['zeta']=float(np.sqrt(row['bound']/row['base']))
                    local_rows.append(row)
                np.savez_compressed(locals_root/arm/f'time-{t:.1f}.npz',**payload)
                del ap,am,payload
            del model
        write(out/'local-summary.json',local_rows);np.savez_compressed(out/'weighted-local.npz',**weighted_local)
        metrics=[]
        for t,bank in zip(TIMES,banks):
            status(phase='weighted raw-linear projections',time=t)
            plus=[];minus=[]
            for arm in ARMS:
                with np.load(fields/arm/f'time-{t:.1f}.npz') as d:plus.append(d['plus']);minus.append(d['minus'])
            result=projection_stats(bank['A']@rotation.T,np.stack(plus),np.stack(minus),bank['oracle']@rotation.T,weights)
            np.savez_compressed(projections/f'time-{t:.1f}.npz',**result)
            metrics.append({k:v for k,v in result.items() if k in ['total','even','linear_odd','nonlinear_odd']})
        aggregate={k:np.mean([r[k] for r in metrics],axis=0) for k in metrics[0]}
        np.savez_compressed(out/'aggregate.npz',**aggregate)
        contrasts=[];decisions={};gaps=aggregate['total'][:,0,None]-aggregate['total'][:,1:4]
        assert np.all(gaps>0),'registered fractions require positive primary gaps in every saved resample'
        for name in ['even','nonlinear_odd']:
            fraction=(aggregate[name][:,0,None]-aggregate[name][:,1:4])/gaps
            state,interval=classify(fraction);decisions[name]=state
            for j,arm in enumerate(ARMS[1:4]):
                contrasts.append(dict(component=name,reference=arm,total_excess=float(gaps[0,j]),component_excess=float(aggregate[name][0,0]-aggregate[name][0,j+1]),
                    signed_fraction=float(fraction[0,j]),low95=float(interval[0,j]),high95=float(interval[1,j])))
        middle=[r for r in local_rows if r['arm']=='ode_k2_direct' and r['site'].endswith('.middle')]
        assert len(middle)==15
        small=all(r['eta']<=.10 for r in middle);certificate=small and all(r['zeta']<=.10 for r in middle)
        decisions['middle_activation_added_odd']='UNIFORMLY_SMALL' if small else 'REJECTED_UNIFORMLY_SMALL'
        decisions['small_center_certificate']=certificate
        write(out/'decision.json',dict(status='COMPLETE',explanation_decisions=decisions,contrasts=contrasts,
            primary_middle_cases=middle,architecture_selected=False,new_training_runs=0))
        write(out/'summary.json',dict(arms=list(ARMS),times=list(TIMES),rows=n,bootstrap_resamples=256,bootstrap_seed=0,
            aggregate={arm:{k:float(v[0,j]) for k,v in aggregate.items()} for j,arm in enumerate(ARMS)}))
        write(out/'input-bindings.json',bindings)
        for path,digest in bindings.items():assert sha(path)==digest,path
        for name,digest in load(source/'source-sha256.json').items():assert sha(source/name)==digest
        write(out/'audit.json',dict(status='PASS',self_tests=tests,alias=alias,native_float32_panels=native32,
            native_float64_GLU_panels=native64,old_positive_full_panels=positive,model_time_cases=len(ARMS)*len(TIMES),local_cases=len(local_rows),
            input_bindings=len(bindings),threads=torch.get_num_threads(),new_model_updates=0))
        status(state='COMPLETE',phase='frozen diagnosis complete',decisions=decisions)
    except BaseException:
        write(out/'failure.json',dict(traceback=traceback.format_exc(),pid=os.getpid(),elapsed_seconds=time.monotonic()-started))
        status(state='FAILED',phase='exception');raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
