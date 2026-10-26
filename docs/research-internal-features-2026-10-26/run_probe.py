"""Actual internal features of immutable direct predictors; no model updates."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import psutil
import torch
from probe_math import design,solve,predict,self_test

ARMS=('ode_k2_direct','swiglu128_direct','swiglu192_direct','swiglu384_direct')
TIMES=(.1,.3,.5,.7,.9);DATASET='rotated_product_bimodal_16d'
COMMON=('z','ffn','residual');EXTRA=('clock1','solve1','increment1','middle','clock2','increment2')

def load(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sites(arm):
    local=COMMON+EXTRA if arm=='ode_k2_direct' else COMMON
    return ('input',)+tuple(f'b{i}.{site}' for i in range(3) for site in local)+('post_norm',)

def extract(model,x,t,arm):
    result={site:[] for site in sites(arm)};values=[];hooks=[]
    def save(site,value):result[site].append(value.detach().numpy().copy())
    hooks.append(model.raw.input.register_forward_hook(lambda m,a,y:save('input',y)))
    hooks.append(model.raw.norm.register_forward_hook(lambda m,a,y:save('post_norm',y)))
    for index,block in enumerate(model.raw.blocks):
        prefix=f'b{index}.'
        hooks.append(block.ffn.register_forward_pre_hook(lambda m,a,k=prefix+'z':save(k,a[0])))
        hooks.append(block.ffn.register_forward_hook(lambda m,a,y,k=prefix+'ffn':save(k,y)))
        hooks.append(block.register_forward_hook(lambda m,a,y,k=prefix+'residual':save(k,y)))
        if arm=='ode_k2_direct':
            hooks.append(block.ffn.ode1.t_func[1].register_forward_hook(lambda m,a,y,k=prefix+'clock1':save(k,y)))
            hooks.append(block.ffn.ode2.t_func[1].register_forward_hook(lambda m,a,y,k=prefix+'clock2':save(k,y)))
            hooks.append(block.ffn.ode1.register_forward_hook(lambda m,a,y,k=prefix+'solve1':save(k,y)))
            hooks.append(block.ffn.ode1.register_forward_hook(lambda m,a,y,k=prefix+'increment1':save(k,y-a[0])))
            hooks.append(block.ffn.act.register_forward_hook(lambda m,a,y,k=prefix+'middle':save(k,y)))
            hooks.append(block.ffn.ode2.register_forward_hook(lambda m,a,y,k=prefix+'increment2':save(k,y-a[0])))
    try:
        with torch.no_grad():
            for start in range(0,len(x),256):
                xx=torch.from_numpy(np.ascontiguousarray(x[start:start+256]));tt=torch.full((len(xx),1),t,dtype=xx.dtype)
                values.append(model(xx,tt).numpy())
    finally:
        for hook in hooks:hook.remove()
    result={k:np.concatenate(v) for k,v in result.items()}
    for key,value in result.items():assert value.shape==(len(x),64) and np.isfinite(value).all(),key
    return result,np.concatenate(values)

def main(args):
    torch.set_num_threads(1);out=args.output;out.mkdir(parents=True,exist_ok=False);started=time.monotonic();inputs={};rows=[]
    def bind(p):inputs[str(p.resolve())]=sha(p)
    def progress(state='RUNNING',**extra):
        record=dict(pid=psutil.Process().pid,elapsed_seconds=time.monotonic()-started,rss_mib=psutil.Process().memory_info().rss/2**20,**extra)
        write(out/'status.json',dict(state=state,**record))
        with (out/'telemetry.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    source=Path(__file__).parent
    for name,digest in load(source/'source-sha256.json').items():assert sha(source/name)==digest
    provenance=load(args.models/'provenance.json');old=Path(provenance['source_root'])
    for name,digest in provenance['source_sha256'].items():assert sha(old/name)==digest
    sys.path.insert(0,str(old))
    from synthetic.conservative_readout import build
    from synthetic.train import restore_model
    import synthetic.models as code
    assert Path(code.__file__).resolve().is_relative_to(old)
    for name in ['provenance.json','study.json','status.json']:bind(args.models/name)
    bind(old/'source-sha256.json')
    for name in ['population.json','standardized-draws.npz','audit.json','status.json']:bind(args.probes/name)
    pop=load(args.probes/'population.json');rotation=np.array(pop['rotation']);mu=np.array(pop['locations'])
    with np.load(args.probes/'standardized-draws.npz') as d:pair=(d['sign_A']*mu+.2*d['target_noise_A']-d['source_A'])@rotation.T
    banks=[]
    for t in TIMES:
        path=args.probes/'banks'/f'time-{t:.1f}.npz';bind(path)
        with np.load(path) as d:banks.append({k:d[k] for k in d.files})
    settings=load(args.models/'study.json')['arms'];tests=self_test();native32=0;compatibility=0.;fit_count=0
    def add_rows(arm,site,label,t,pa,ph,target):
        ea=pa-target;eh=ph-target;n=len(pa)
        for fold,sl in enumerate([slice(0,n//2),slice(n//2,n)]):
            total=float(((ea[sl]**2+eh[sl]**2)/2).mean());own=float((ea[sl]*eh[sl]).mean());other=float(((ea[sl]-eh[sl])**2/2).mean())
            np.testing.assert_allclose(total,own+other,atol=1e-12)
            rows.append(dict(arm=arm,site=site,label=label,time=t,evaluation_fold=fold,total=total,own=own,other=other))
    for arm in ARMS:
        progress(phase='restore model',arm=arm);folder=out/arm;folder.mkdir();oldrun=args.models/'runs'/DATASET/arm
        for name in ['step_0005000.pt','result.json']:bind(oldrun/name)
        s=settings[arm];model=build(16,s['family'],s['readout']).eval()
        restore_model(model,torch.load(oldrun/'step_0005000.pt',map_location='cpu',weights_only=False))
        for parameter in model.parameters():parameter.requires_grad_(False)
        for t in TIMES:
            bp=args.models/'banks'/DATASET/f'panel_{t:.1f}.npz';sp=oldrun/'diagnostics'/f'panel_{t:.1f}.npz';bind(bp);bind(sp)
            with np.load(bp) as b,np.load(sp) as saved:
                with torch.enable_grad():actual=model(torch.from_numpy(b['x']).requires_grad_(True),torch.from_numpy(b['t'])).detach().numpy()
                np.testing.assert_array_equal(actual,saved['prediction']);native32+=1
        model=model.double();a_features=[];residual_targets=[]
        for t,bank in zip(TIMES,banks):
            progress(phase='A feature extraction',arm=arm,time=t)
            f,value=extract(model,bank['A']@rotation.T,t,arm);oldfield=args.probes/'fields'/arm/f'time-{t:.1f}.npz';bind(oldfield)
            with np.load(oldfield) as saved:np.testing.assert_array_equal(value@rotation,saved['native'])
            residual_targets.append(np.column_stack([bank['oracle']@rotation.T-value,pair-value]))
            a_features.append(f);np.savez_compressed(folder/f'features_{t:.1f}.npz',**f)
        n=len(pair);mid=n//2;coeff={};fitrecords=[]
        for site in sites(arm):
            folds=[]
            for fold,sl in enumerate([slice(0,mid),slice(mid,n)]):
                x=design(np.concatenate([f[site][sl] for f in a_features]));y=np.concatenate([v[sl] for v in residual_targets])
                c,record=solve(x,y);folds.append(c);fit_count+=1
                fitrecords.append(dict(site=site,training_fold=fold,**record))
            coeff[site]=np.array(folds)
        np.savez_compressed(folder/'coefficients.npz',**coeff);write(folder/'fit-records.json',fitrecords)
        for tindex,(t,bank) in enumerate(zip(TIMES,banks)):
            progress(phase='held-out hybrids',arm=arm,time=t)
            with np.load(args.probes/'fields'/arm/f'time-{t:.1f}.npz') as saved:baseA=saved['native'];baseH=saved['hybrid_diagonal']
            pa={site:baseA[:,None,:]+predict(a_features[tindex][site],coeff[site],rotation) for site in sites(arm)}
            ph={site:np.empty_like(value) for site,value in pa.items()}
            for i in range(16):
                hybrid=bank['B'].copy();hybrid[:,i]=bank['A'][:,i]
                f,value=extract(model,hybrid@rotation.T,t,arm)
                np.testing.assert_array_equal((value@rotation)[:,i],baseH[:,i])
                for site in sites(arm):ph[site][:,:,i]=baseH[:,i,None]+predict(f[site],coeff[site],rotation)[:,:,i]
            previous=args.previous/arm/f'predictions_{t:.1f}.npz';bind(previous)
            with np.load(previous) as oldheads:
                for suffix,value in [('A',pa['post_norm']),('H',ph['post_norm'])]:
                    wanted=oldheads['post_norm_shared_'+suffix]
                    compatibility=max(compatibility,float(np.max(np.abs(value-wanted))))
                    np.testing.assert_allclose(value,wanted,atol=1e-10,rtol=1e-10)
            payload={}
            for site in sites(arm):
                payload[site+'_A']=pa[site];payload[site+'_H']=ph[site]
                for j,label in enumerate(['mean','pair']):add_rows(arm,site,label,t,pa[site][:,j],ph[site][:,j],bank['oracle'])
            add_rows(arm,'native','native',t,baseA,baseH,bank['oracle'])
            np.savez_compressed(folder/f'predictions_{t:.1f}.npz',**payload)
        del model,a_features,residual_targets,pa,ph
    write(out/'rows.json',rows);write(out/'input-bindings.json',inputs)
    for name,digest in inputs.items():assert sha(name)==digest,name
    for name,digest in provenance['source_sha256'].items():assert sha(old/name)==digest
    for name,digest in load(source/'source-sha256.json').items():assert sha(source/name)==digest
    write(out/'audit.json',dict(status='PASS',self_tests=tests,native_float32_panels=native32,fit_count=fit_count,
        feature_site_count=sum(len(sites(arm)) for arm in ARMS),rows=len(rows),final_probe_compatibility_max=compatibility,
        input_bindings=len(inputs),source_sha256=sha(__file__),helper_sha256=sha(source/'probe_math.py')))
    progress(state='COMPLETE',phase='all models complete',rows=len(rows))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--models',type=Path);p.add_argument('--probes',type=Path);p.add_argument('--previous',type=Path);p.add_argument('--output',type=Path);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:print(json.dumps(self_test()))
    else:main(a)
