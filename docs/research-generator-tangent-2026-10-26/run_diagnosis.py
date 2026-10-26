"""Measured linear/quadratic P pullbacks of the frozen original K2 model."""
import argparse,os,time,traceback
from pathlib import Path
import numpy as np
import psutil
import torch
from study_support import TIMES,load,sha,write,build,layers,native_check,adjoint_chunks,input_paths
from tangent_math import routes,ratios,self_test

def main(root):
    torch.set_num_threads(1);started=time.monotonic();source=Path(__file__).parent
    for name,digest in load(source/'source-sha256.json').items():assert sha(source/name)==digest
    out=root/'output';out.mkdir(exist_ok=False);peak=0.;proc=psutil.Process()
    def progress(state='RUNNING',**extra):
        nonlocal peak
        rss=proc.memory_info().rss/2**20;peak=max(peak,rss)
        record=dict(state=state,pid=os.getpid(),elapsed_seconds=time.monotonic()-started,rss_mib=rss,peak_sampled_rss_mib=peak,**extra)
        write(out/'status.json',record)
        with (out/'telemetry.jsonl').open('a') as f:
            import json
            f.write(json.dumps(record)+'\n')
        print(record,flush=True)
    try:
        tests=self_test();model,checkpoint,provenance=build();native=native_check(model,checkpoint);model=model.double();found=layers(model)
        bindings={str(p.resolve()):sha(p) for p in input_paths(checkpoint,provenance)+[root/'preflight.json',source/'source-sha256.json']}
        panels=[];matrices={};global_mats={};spectral=[];clockrows=[];grad_error=0.;spectra={}
        for name,m in found:
            p=m.proj.weight.detach().numpy().copy();values=np.linalg.eigvals(p);singular=np.linalg.svd(p,compute_uv=False)
            symmetric_min=float(np.linalg.eigvalsh((p+p.T)/2)[0]);spectra[name]=(p,values,symmetric_min,float(singular[0]))
            matrices['P__'+name]=p;matrices['eigenvalues__'+name]=values;matrices['singular_values__'+name]=singular
            for label in ['total','nonlinear']:global_mats[label,name]=[np.zeros_like(p),np.zeros_like(p),np.zeros_like(p),np.zeros(3)]
        for t in TIMES:
            progress(phase='complete paired pullbacks',time=t)
            sums={(label,name):[np.zeros((64,64)),np.zeros((64,64)),np.zeros((64,64)),np.zeros(3)] for label in ['total','nonlinear'] for name,_ in found}
            clocks={name:[] for name,_ in found}
            for label,sign,start,records in adjoint_chunks(model,t):
                for name,r in records.items():
                    g1,g2,energy=routes(r['x'],r['u'],r['a'],r['p'])
                    np.testing.assert_allclose(g1+g2,r['native'],atol=2e-12,rtol=1e-9)
                    grad_error=max(grad_error,float(np.max(np.abs(g1+g2-r['native']))))
                    for a,b in zip(sums[label,name],[g1,g2,r['native'],energy]):a+=b
                    if label=='total':clocks[name].append(r['a'][:,0].copy())
            for (label,name),(g1,g2,g,energy) in sums.items():
                np.testing.assert_allclose(g1+g2,g,atol=2e-12,rtol=1e-9)
                record=dict(objective=label,time=t,layer=name,**ratios(g1,g2,energy));panels.append(record)
                for key,value in zip(['route1','route2','native','energies'],[g1,g2,g,energy]):matrices[f'{label}__{t:.1f}__{name}__{key}']=value
                for a,b in zip(global_mats[label,name],[g1,g2,g,energy]):a+=b
            for name,values in clocks.items():
                a=np.concatenate(values);assert len(a)==8192
                q=np.quantile(a,[.05,.5,.95]);p,lam,smin,pnorm=spectra[name]
                clockrows.append(dict(layer=name,time=t,minimum=float(a.min()),mean=float(a.mean()),maximum=float(a.max()),
                    q05=float(q[0]),q50=float(q[1]),q95=float(q[2])))
                for probability,duration in zip([.05,.5,.95],q):
                    factors=1+duration*(lam[:,None]+lam[None,:])/2
                    spectral.append(dict(layer=name,time=t,quantile=probability,duration=float(duration),minimum_eigenfactor=float(np.abs(factors).min()),
                        maximum_eigenfactor=float(np.abs(factors).max()),symmetric_P_minimum=smin,P_spectral_norm=pnorm,
                        certified_normalized_singular_lower_bound=max(0.,1+float(duration)*smin)))
        aggregates=[]
        for (label,name),(g1,g2,g,energy) in global_mats.items():
            aggregates.append(dict(objective=label,layer=name,**ratios(g1,g2,energy)))
            for key,value in zip(['route1','route2','native','energies'],[g1,g2,g,energy]):matrices[f'{label}__aggregate__{name}__{key}']=value
        primary=[r for r in aggregates if r['objective']=='nonlinear'];assert len(primary)==6
        severity={kind:[r['layer'] for r in primary if r[kind+'_ratio'] is not None and r[kind+'_ratio']<=.1] for kind in ['local','coherent']}
        decision=dict(primary_objective='nonlinear',threshold=.10,severe_local_layers=severity['local'],severe_coherent_layers=severity['coherent'],
            interpretation='no severe measured cancellation' if not any(severity.values()) else 'measured cancellation needs further diagnosis',architecture_selected=False,new_model_updates=0)
        np.savez_compressed(out/'matrices.npz',**matrices)
        write(out/'panels.json',panels);write(out/'aggregates.json',aggregates);write(out/'spectral-context.json',spectral);write(out/'clocks.json',clockrows);write(out/'decision.json',decision)
        write(out/'input-bindings.json',bindings)
        for path,digest in bindings.items():assert sha(path)==digest,path
        for name,digest in load(source/'source-sha256.json').items():assert sha(source/name)==digest
        write(out/'audit.json',dict(status='PASS',self_tests=tests,native_float32_panels=native,panel_rows=len(panels),aggregate_rows=len(aggregates),
            spectral_rows=len(spectral),input_bindings=len(bindings),maximum_native_P_gradient_discrepancy=grad_error,threads=torch.get_num_threads(),new_model_updates=0))
        progress(state='COMPLETE',phase='frozen generator diagnosis complete',decision=decision)
    except BaseException:
        write(out/'failure.json',dict(traceback=traceback.format_exc(),pid=os.getpid(),elapsed_seconds=time.monotonic()-started));progress(state='FAILED',phase='exception');raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
