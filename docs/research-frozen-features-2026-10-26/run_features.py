"""Frozen-backbone linear-readout diagnosis with actual native activations."""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import psutil
import torch
from linear_probe import SITES,SCOPES,LABELS,fit_all,predict,self_test

ARMS=('ode_k2_direct','swiglu128_direct','swiglu192_direct','swiglu384_direct')
TIMES=(.1,.3,.5,.7,.9)
DATASET='rotated_product_bimodal_16d'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

def features(model,x,clock):
    captured={key:[] for key in SITES};outputs=[]
    def save(key,tensor):captured[key].append(tensor.detach().cpu().numpy().copy())
    hooks=[model.raw.input.register_forward_hook(lambda m,a,y:save('input',y)),
        model.raw.norm.register_forward_pre_hook(lambda m,a:save('pre_norm',a[0])),
        model.raw.norm.register_forward_hook(lambda m,a,y:save('post_norm',y))]
    try:
        with torch.no_grad():
            for begin in range(0,len(x),256):
                xx=torch.from_numpy(np.ascontiguousarray(x[begin:begin+256]))
                tt=torch.full((len(xx),1),clock,dtype=xx.dtype)
                outputs.append(model(xx,tt).detach().numpy())
    finally:
        for hook in hooks:hook.remove()
    return {k:np.concatenate(v) for k,v in captured.items()},np.concatenate(outputs)

def metrics(pa,ph,target):
    a=pa-target[:,None,:];h=ph-target[:,None,:]
    total=((a*a+h*h)/2).mean(-1);own=(a*h).mean(-1);other=((a-h)**2/2).mean(-1)
    np.testing.assert_allclose(total,own+other,atol=1e-12)
    return total,own,other

def main(args):
    torch.set_num_threads(1);out=args.output;out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();inputs={};rows=[];native_replays=0
    def bind(p):inputs[str(p.resolve())]=sha(p)
    def progress(state='RUNNING',**extra):
        record=dict(pid=psutil.Process().pid,elapsed_seconds=time.monotonic()-started,
            rss_mib=psutil.Process().memory_info().rss/2**20,**extra)
        with (out/'telemetry.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        write(out/'status.json',dict(state=state,**record));print(json.dumps(record),flush=True)
    source=Path(__file__).parent
    for name,wanted in load(source/'source-sha256.json').items():assert sha(source/name)==wanted
    provenance=load(args.models/'provenance.json');old=Path(provenance['source_root'])
    for name,wanted in provenance['source_sha256'].items():assert sha(old/name)==wanted
    sys.path.insert(0,str(old))
    from synthetic.conservative_readout import build
    from synthetic.train import restore_model
    import synthetic.models as model_code
    assert Path(model_code.__file__).resolve().is_relative_to(old)
    for name in ['provenance.json','study.json','status.json']:bind(args.models/name)
    bind(old/'source-sha256.json')
    for name in ['population.json','standardized-draws.npz','audit.json','status.json']:bind(args.probes/name)
    pop=load(args.probes/'population.json');mu=np.array(pop['locations']);rotation=np.array(pop['rotation'])
    with np.load(args.probes/'standardized-draws.npz') as d:
        endpoint=d['sign_A']*mu+.2*d['target_noise_A'];z=d['source_A'];pair=(endpoint-z)@rotation.T
    banks=[]
    for clock in TIMES:
        path=args.probes/'banks'/f'time-{clock:.1f}.npz';bind(path)
        with np.load(path) as b:banks.append({k:b[k] for k in b.files})
        np.testing.assert_allclose((1-clock)*z+clock*endpoint,banks[-1]['A'],atol=1e-14)
    targets=np.stack([np.concatenate([b['oracle']@rotation.T,pair],axis=1) for b in banks])
    settings=load(args.models/'study.json')['arms'];tests=self_test()
    def record(arm,site,scope,label,clock,vals):
        n=len(vals[0]);mid=n//2
        for fold,sl in enumerate([slice(0,mid),slice(mid,n)]):
            rows.append(dict(arm=arm,site=site,scope=scope,label=label,time=clock,evaluation_fold=fold,
                total=float(vals[0][sl].mean()),own=float(vals[1][sl].mean()),other=float(vals[2][sl].mean())))
    for arm in ARMS:
        setting=settings[arm];folder=out/arm;folder.mkdir();oldrun=args.models/'runs'/DATASET/arm
        for name in ['step_0005000.pt','result.json']:bind(oldrun/name)
        model=build(16,setting['family'],setting['readout']).eval()
        snapshot=torch.load(oldrun/'step_0005000.pt',map_location='cpu',weights_only=False)
        restore_model(model,snapshot)
        for p in model.parameters():p.requires_grad_(False)
        for clock in TIMES:
            bp=args.models/'banks'/DATASET/f'panel_{clock:.1f}.npz';sp=oldrun/'diagnostics'/f'panel_{clock:.1f}.npz'
            bind(bp);bind(sp)
            with np.load(bp) as b,np.load(sp) as s:
                with torch.enable_grad():value=model(torch.from_numpy(b['x']).requires_grad_(True),torch.from_numpy(b['t'])).detach().numpy()
                np.testing.assert_array_equal(value,s['prediction']);native_replays+=1
        model=model.double();a_features=[]
        for clock,bank in zip(TIMES,banks):
            progress(phase='A features',arm=arm,time=clock)
            oldfield=args.probes/'fields'/arm/f'time-{clock:.1f}.npz';bind(oldfield)
            f,value=features(model,bank['A']@rotation.T,clock)
            with np.load(oldfield) as saved:
                native=value@rotation
                np.testing.assert_array_equal(native,saved['native'])
            head=f['post_norm']@model.raw.output.weight.detach().numpy().T+model.raw.output.bias.detach().numpy()
            np.testing.assert_allclose(head,value,atol=1e-12,rtol=1e-12)
            a_features.append(f)
            np.savez_compressed(folder/f'features_{clock:.1f}.npz',**f,native=native)
        stacked={site:np.stack([f[site] for f in a_features]) for site in SITES}
        co,fitrecords=fit_all(stacked,targets)
        np.savez_compressed(folder/'coefficients.npz',**co);write(folder/'fit-records.json',fitrecords)
        for time_index,(clock,bank) in enumerate(zip(TIMES,banks)):
            progress(phase='hybrid features and held-out readouts',arm=arm,time=clock)
            predA={};predH={}
            for site in SITES:
                for scope in SCOPES:
                    key=site+'_'+scope;c=co[key] if scope=='shared' else co[key][time_index]
                    predA[key]=predict(a_features[time_index][site],c,rotation)
                    predH[key]=np.empty_like(predA[key])
            oldfield=args.probes/'fields'/arm/f'time-{clock:.1f}.npz'
            with np.load(oldfield) as saved:oldA=saved['native'];oldH=saved['hybrid_diagonal']
            for coordinate in range(16):
                hybrid=bank['B'].copy();hybrid[:,coordinate]=bank['A'][:,coordinate]
                f,value=features(model,hybrid@rotation.T,clock)
                native=value@rotation
                np.testing.assert_array_equal(native[:,coordinate],oldH[:,coordinate])
                for site in SITES:
                    for scope in SCOPES:
                        key=site+'_'+scope;c=co[key] if scope=='shared' else co[key][time_index]
                        predH[key][:,:,coordinate]=predict(f[site],c,rotation)[:,:,coordinate]
            payload={}
            for key in predA:
                site,scope=key.rsplit('_',1)
                payload[key+'_A']=predA[key];payload[key+'_H']=predH[key]
                total,own,other=metrics(predA[key],predH[key],bank['oracle'])
                for j,label in enumerate(LABELS):record(arm,site,scope,label,clock,[v[:,j] for v in [total,own,other]])
            record(arm,'native','native','native',clock,[v[:,0] for v in metrics(oldA[:,None,:],oldH[:,None,:],bank['oracle'])])
            np.savez_compressed(folder/f'predictions_{clock:.1f}.npz',**payload)
    write(out/'rows.json',rows);write(out/'input-bindings.json',inputs)
    for path,wanted in inputs.items():assert sha(path)==wanted,path
    for name,wanted in load(source/'source-sha256.json').items():assert sha(source/name)==wanted
    for name,wanted in provenance['source_sha256'].items():assert sha(old/name)==wanted
    write(out/'audit.json',dict(status='PASS',self_tests=tests,native_float32_panels=native_replays,
        float64_A_replays=20,float64_H_coordinate_replays=320,rows=len(rows),input_bindings=len(inputs),
        source_sha256=sha(__file__),helper_sha256=sha(source/'linear_probe.py')))
    progress(state='COMPLETE',phase='all models complete',rows=len(rows))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--models',type=Path);p.add_argument('--probes',type=Path);p.add_argument('--output',type=Path);p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:print(json.dumps(self_test()))
    else:main(a)
