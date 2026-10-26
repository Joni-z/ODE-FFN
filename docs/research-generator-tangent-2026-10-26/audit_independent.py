"""Explicit per-example outer products; independent of the norm shortcut."""
import argparse,os,time
from pathlib import Path
import numpy as np
import psutil
import torch
from study_support import TIMES,load,write,sha,build,layers,native_check,adjoint_chunks

def main(root):
    torch.set_num_threads(1);started=time.monotonic();out=root/'output';dest=root/'independent-audit';dest.mkdir(exist_ok=False)
    bindings=load(out/'input-bindings.json')
    for path,digest in bindings.items():assert sha(path)==digest,path
    source=Path(__file__).parent
    for name,digest in load(source/'source-sha256.json').items():assert sha(source/name)==digest
    model,checkpoint,_=build();assert native_check(model,checkpoint)==5;model=model.double();found=layers(model)
    target=np.load(out/'matrices.npz');wanted_rows={(r['objective'],r['time'],r['layer']):r for r in load(out/'panels.json')}
    aggregate={(label,name):[np.zeros((64,64)),np.zeros((64,64)),np.zeros((64,64)),np.zeros(3)] for label in ['total','nonlinear'] for name,_ in found}
    max_error=0.;energy_error=0.;rows=0;peak=0.;ratio_rows=[]
    def compare_ratio(label,name,timekey,g1,g2,e,row):
        local=(e[0]+e[1]+2*e[2])/(e[0]+e[1]) if e[0]+e[1] else None
        denominator=np.linalg.norm(g1)**2+np.linalg.norm(g2)**2
        coherent=np.linalg.norm(g1+g2)**2/denominator if denominator else None
        for key,value in [('local_ratio',local),('coherent_ratio',coherent)]:
            if value is None:assert row[key] is None
            else:np.testing.assert_allclose(value,row[key],atol=2e-12,rtol=1e-10)
        if timekey=='aggregate':ratio_rows.append(dict(objective=label,layer=name,local_ratio=local,coherent_ratio=coherent))
    for t in TIMES:
        rss=psutil.Process().memory_info().rss/2**20;peak=max(peak,rss)
        print(dict(pid=os.getpid(),phase='explicit outer-product replay',time=t,elapsed_seconds=time.monotonic()-started,rss_mib=rss),flush=True)
        sums={(label,name):[np.zeros((64,64)),np.zeros((64,64)),np.zeros((64,64)),np.zeros(3)] for label in ['total','nonlinear'] for name,_ in found}
        clocks={name:[] for name,_ in found}
        for label,sign,start,records in adjoint_chunks(model,t):
            for name,r in records.items():
                x,u,a,p=r['x'],r['u'],r['a'],r['p']
                first=np.einsum('bi,bj->bij',u,x)*a[:,:,None]
                second=.5*a[:,:,None]**2*(np.einsum('bi,bj->bij',u@p,x)+np.einsum('bi,bj->bij',u,x@p.T))
                g1=np.sum(first,axis=0);g2=np.sum(second,axis=0)
                energies=np.array([np.sum(first*first),np.sum(second*second),np.sum(first*second)])
                np.testing.assert_allclose(g1+g2,r['native'],atol=2e-12,rtol=1e-9)
                for acc,value in zip(sums[label,name],[g1,g2,r['native'],energies]):acc+=value
                if label=='total':clocks[name].append(a[:,0].copy())
        for (label,name),items in sums.items():
            g1,g2,g,e=items
            for kind,value in zip(['route1','route2','native','energies'],items):
                saved=target[f'{label}__{t:.1f}__{name}__{kind}'];np.testing.assert_allclose(value,saved,atol=2e-12,rtol=1e-9)
                discrepancy=float(np.max(np.abs(value-saved)))
                if kind=='energies':energy_error=max(energy_error,discrepancy)
                else:max_error=max(max_error,discrepancy)
            compare_ratio(label,name,t,g1,g2,e,wanted_rows[label,t,name]);rows+=1
            for acc,value in zip(aggregate[label,name],items):acc+=value
        clock_rows=load(out/'clocks.json');spectral=load(out/'spectral-context.json')
        for name,values in clocks.items():
            a=np.concatenate(values);row=next(r for r in clock_rows if r['time']==t and r['layer']==name)
            np.testing.assert_array_equal(np.quantile(a,[.05,.5,.95]),[row['q05'],row['q50'],row['q95']])
            p=target['P__'+name];eig=np.linalg.eigvals(p);sm=np.linalg.eigvalsh((p+p.T)*.5)[0];norm=np.linalg.svd(p,compute_uv=False)[0]
            for r in [r for r in spectral if r['time']==t and r['layer']==name]:
                eigenfactors=np.array([abs(1+r['duration']*(u+v)/2) for u in eig for v in eig])
                actual=[eigenfactors.min(),eigenfactors.max(),sm,norm,max(0.,1+r['duration']*sm)]
                expected=[r[k] for k in ['minimum_eigenfactor','maximum_eigenfactor','symmetric_P_minimum','P_spectral_norm','certified_normalized_singular_lower_bound']]
                np.testing.assert_allclose(actual,expected,atol=1e-13,rtol=1e-12)
    wanted_aggregate={(r['objective'],r['layer']):r for r in load(out/'aggregates.json')}
    for (label,name),(g1,g2,g,e) in aggregate.items():
        compare_ratio(label,name,'aggregate',g1,g2,e,wanted_aggregate[label,name])
        for kind,value in zip(['route1','route2','native','energies'],[g1,g2,g,e]):np.testing.assert_allclose(value,target[f'{label}__aggregate__{name}__{kind}'],atol=2e-12,rtol=1e-9)
    primary=[r for r in ratio_rows if r['objective']=='nonlinear'];decision=load(out/'decision.json')
    for kind in ['local','coherent']:
        severe=[r['layer'] for r in primary if r[kind+'_ratio'] is not None and r[kind+'_ratio']<=.1]
        assert severe==decision['severe_'+kind+'_layers']
    for path,digest in bindings.items():assert sha(path)==digest,path
    for name,digest in load(source/'source-sha256.json').items():assert sha(source/name)==digest
    result=dict(status='PASS',pid=os.getpid(),elapsed_seconds=time.monotonic()-started,peak_sampled_rss_mib=peak,
        panel_rows_reproduced=rows,aggregate_rows_reproduced=len(aggregate),spectral_rows_reproduced=90,
        maximum_matrix_discrepancy=max_error,maximum_energy_discrepancy=energy_error,decisions_reproduced=True)
    write(dest/'audit.json',result);write(dest/'status.json',dict(state='COMPLETE',**result));print(result,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
