"""Read-only fitted-readout invariants, with fixed zero-P sampling interventions."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch


def load(path: Path) -> dict:return json.loads(path.read_text())
def sha(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(8*2**20):digest.update(block)
    return digest.hexdigest()
def write(path: Path,value: object) -> None:path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def table(path: Path,rows: list[dict]) -> None:
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def coefficients(head: str,steps: int=32,cap: float=.05) -> tuple[float,float]:
    if head=='direct':return 1.,1.
    r,d=1.,0.;h=1/steps
    for index in range(steps):
        a0=-1/max(1-index*h,cap);a1=-1/max(1-(index+1)*h,cap)
        f0=-a0 if head=='clean' else 1.;f1=-a1 if head=='clean' else 1.
        factor=1+h*(a0+a1+h*a0*a1)/2
        forcing=h*((1+h*a1)*f0+f1)/2
        r,d=factor*r,factor*d+forcing
    return r,d


def null_projector(matrix: np.ndarray) -> tuple[np.ndarray,int,np.ndarray]:
    _,singular,vh=np.linalg.svd(matrix,full_matrices=True)
    cutoff=1e-10*max(float(singular[0]) if len(singular) else 0.,1.)
    rank=int(np.sum(singular>cutoff));basis=vh[rank:].T
    projector=basis@basis.T
    np.testing.assert_allclose(projector@projector,projector,rtol=0,atol=1e-8)
    np.testing.assert_allclose(projector.T,projector,rtol=0,atol=1e-8)
    assert np.max(np.abs(matrix@projector),initial=0.)<=1e-8
    return projector,rank,basis


def readout(state: dict) -> tuple[np.ndarray,np.ndarray]:
    weight=state['model']['raw.output.weight'].double().numpy();bias=state['model']['raw.output.bias'].double().numpy()
    gamma=state['model']['raw.norm.weight'].double().numpy();beta=state['model']['raw.norm.bias'].double().numpy()
    centre=np.eye(len(gamma))-np.ones((len(gamma),len(gamma)))/len(gamma)
    return (weight*gamma[None,:])@centre,weight@beta+bias


def setup(root: Path):
    settings=load(root/'settings.json');base=Path(settings['cycle07']);sys.path.insert(0,str(base/'source'))
    import study as previous
    assert Path(previous.__file__).resolve()==base/'source/study.py'
    _,native,core=previous.setup(base);torch.set_num_threads(1)
    return settings,base,previous,native,core


def verify_inputs(root: Path) -> None:
    for name,digest in load(root/'bindings.json').items():assert sha(Path(name))==digest,name
    _,base,previous,native,_=setup(root)
    for name,digest in load(base/'sealed-runtime.json').items():assert sha(base/name)==digest,name
    previous.verify_inputs(base,native)


def prepare(root: Path,base: Path) -> None:
    root.mkdir(exist_ok=False);source=root/'source';source.mkdir()
    for name in ('protocol.md','proof.md','invariant_study.py','invariant_fixtures.py'):shutil.copyfile(Path(__file__).parent/name,source/name)
    shutil.copyfile(Path(__file__).parent.parent/'readout-flow-invariant.md',source/'theory-review.md')
    repo=Path(__file__).resolve().parents[3];old=base.parent/'oct27-cycle05'
    controls=[];registered=load(old/'bindings.json')
    for entry in load(old/'settings.json')['laws']:
        if entry['law'] in ('gmm_ring8_2d','rotated_product_bimodal_16d'):
            checkpoint=Path(entry['references']['swiglu128'])/'step_0005000.pt'
            assert sha(checkpoint)==registered[str(checkpoint)]
            controls.append(dict(law=entry['law'],checkpoint=str(checkpoint),dim=entry['dim']))
    assert len(controls)==2
    write(root/'settings.json',dict(cycle07=str(base.resolve()),controls=controls,
        registration_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()))
    paths=list(source.iterdir())+[root/'settings.json',base/'sealed-runtime.json']+[Path(c['checkpoint']) for c in controls]
    write(root/'bindings.json',{str(p):sha(p) for p in paths});verify_inputs(root)
    print(dict(prepared=str(root),bindings=len(paths),cohort=18,zero_P=6,dimension_controls=2),flush=True)


def preflight(root: Path) -> None:
    verify_inputs(root)
    from invariant_fixtures import verify
    result=verify(null_projector,coefficients);assert result['status']=='PASS'
    write(root/'preflight.json',result);write(root/'ready.json',dict(bindings_sha256=sha(root/'bindings.json'),preflight_sha256=sha(root/'preflight.json')))
    print(result,flush=True)


def identity_metrics(source: np.ndarray,generated: np.ndarray,projector: np.ndarray,c0: np.ndarray,r: float,d: float) -> dict:
    start=source.astype(np.float64)@projector;observed=generated.astype(np.float64)@projector
    expected=r*start+d*(c0@projector);error=observed-expected
    maximum=float(np.max(np.abs(error)));rms=float(np.sqrt(np.mean(error**2)))
    assert maximum<=1e-4,('source-wise identity',maximum)
    a=observed-observed.mean(0);b=start-start.mean(0)
    cov=a.T@a/(len(a)-1);reference=r*r*(b.T@b)/(len(b)-1)
    relative=float(np.linalg.norm(cov-reference)/max(np.linalg.norm(reference),1e-30))
    assert relative<=1e-4,('empirical covariance',relative)
    return dict(identity_max=maximum,identity_rms=rms,covariance_relative_frobenius=relative,
        observed_projected_energy_per_ambient=float(np.mean(observed**2)),predicted_projected_energy_per_ambient=float(np.mean(expected**2)))


def run(root: Path) -> None:
    started=time.monotonic();verify_inputs(root);settings,base,previous,native,core=setup(root)
    ready=load(root/'ready.json');assert sha(root/'bindings.json')==ready['bindings_sha256'] and sha(root/'preflight.json')==ready['preflight_sha256']
    out=root/'output';out.mkdir(exist_ok=False);observations=out/'raw';observations.mkdir()
    basis=np.load(base/'basis.npy');rows=[];raw_rows=[];interventions=[];controls=[];telemetry=[]
    import psutil
    def status(complete=False,latest=''):
        elapsed=time.monotonic()-started;rss=psutil.Process().memory_info().rss
        row=dict(state='COMPLETE' if complete else 'RUNNING',completed=len(rows),total=18,latest=latest,pid=os.getpid(),elapsed_seconds=elapsed,rss_bytes=rss)
        write(out/'status.json',row);telemetry.append(row)
        assert elapsed<600,'registered wall-time ceiling'
    status();write(out/'invocation.json',dict(executable=sys.executable,argv=sys.argv,pid=os.getpid(),registration_commit=settings['registration_commit']))
    for entry in load(base/'settings.json')['laws']:
        law=entry['law'];banks=Path(entry['banks'])
        with np.load(banks/'sampling.npz') as data:source=data['source']
        for arm in previous.ARMS:
            family,head=arm.rsplit('_',1);folder=base/'output/runs'/law/arm;state=torch.load(folder/'step_0005000.pt',weights_only=False)
            model=previous.build(base,law,arm);native.restore_model(model,state);model.eval();assert model.raw.norm.eps>0
            b,c0=readout(state);projector,rank,left=null_projector(b.T)
            normal_projector,combined_rank,normal_left=null_projector(np.vstack((b.T,basis.T)))
            assert rank<=63 and 128-combined_rank>=63
            r,d=coefficients(head)
            with np.load(folder/'diagnostics/samples.npz') as data:generated=data['generated']
            actual=identity_metrics(source,generated,projector,c0,r,d);normal=identity_metrics(source,generated,normal_projector,c0,r,d)
            q=entry['normal_variance'];dimension=128-combined_rank;mean=d*(c0@normal_left)
            w2=float(np.sum(mean**2)+dimension*(abs(r)-np.sqrt(q))**2)
            population_energy=float((np.sum(mean**2)+dimension*r*r)/128)
            np.savez_compressed(observations/f'{law}-{arm}.npz',projector=projector,normal_projector=normal_projector,B=b,c0=c0,
                expected_projected_sample=r*(source@projector)+d*(c0@projector))
            rows.append(dict(law=law,family=family,head=head,readout_rank=rank,null_dimension=128-rank,normal_null_dimension=dimension,
                Heun_R=r,Heun_d=d,**actual,normal_identity_max=normal['identity_max'],normal_covariance_relative_frobenius=normal['covariance_relative_frobenius'],
                target_normal_variance=q,population_full_W2_squared_lower_bound=w2,population_full_W2_squared_bound_per_ambient=w2/128,
                population_normal_energy_bound_per_ambient=population_energy,
                empirical_normal_null_energy_per_ambient=normal['observed_projected_energy_per_ambient']))
            for clock in previous.TIMES:
                with np.load(banks/f'panel_{clock:.1f}.npz') as data:x,t=data['x'],data['t']
                with torch.no_grad():raw=model.raw(torch.from_numpy(x),torch.from_numpy(t)).double().numpy()
                error=raw@projector-c0@projector;maximum=float(np.max(np.abs(error)));assert maximum<=2e-5,maximum
                raw_rows.append(dict(law=law,family=family,head=head,time=clock,anchors=len(raw),raw_projection_max=maximum))
            if family=='ode_k2':
                changed=copy.deepcopy(model);changed_names=[]
                with torch.no_grad():
                    for name,layer in changed.ode_layers():layer.proj.weight.zero_();changed_names.append('raw.'+name+'.proj.weight')
                assert len(changed_names)==6
                for name,value in model.state_dict().items():
                    if name not in changed_names:assert torch.equal(value,changed.state_dict()[name]),name
                for name,value in model.named_buffers():assert torch.equal(value,dict(changed.named_buffers())[name]),name
                counterfactual=native.legacy.generated_samples(changed,torch.from_numpy(source),32).numpy()
                np.savez_compressed(observations/f'{law}-{arm}-zero-P.npz',generated=counterfactual)
                check=identity_metrics(source,counterfactual,projector,c0,r,d);normal_check=identity_metrics(source,counterfactual,normal_projector,c0,r,d)
                difference=counterfactual.astype(np.float64)-generated
                magnitude=float(np.sqrt(np.mean(difference**2)))
                interventions.append(dict(law=law,head=head,P_matrices_zeroed=len(changed_names),whole_sample_displacement_rms=magnitude,
                    nonvacuous=magnitude>1e-7,unchanged_non_P_tensors=True,**check,normal_identity_max=normal_check['identity_max'],
                    change_in_null_coordinates_max=float(np.max(np.abs(difference@projector)))))
            status(latest=law+'/'+arm);print(dict(law=law,arm=arm,identity_max=actual['identity_max'],elapsed=time.monotonic()-started),flush=True)
    for entry in settings['controls']:
        state=torch.load(entry['checkpoint'],weights_only=False);b,c0=readout(state);projector,rank,_=null_projector(b.T)
        controls.append(dict(law=entry['law'],dim=entry['dim'],readout_rank=rank,null_dimension=entry['dim']-rank,
            projector_frobenius=float(np.linalg.norm(projector)),applies_to128D=False))
        assert rank==entry['dim'] and np.linalg.norm(projector)==0.
    assert (len(rows),len(raw_rows),len(interventions),len(controls))==(18,90,6,2)
    verify_inputs(root)
    for name,values in [('invariants.csv',rows),('raw-output-checks.csv',raw_rows),('P-interventions.csv',interventions),('dimension-controls.csv',controls),('telemetry.csv',telemetry)]:table(out/name,values)
    write(out/'receipt.json',dict(status='PASS',models=18,raw_probe_panels=90,raw_probe_anchors=5760,zero_P_interventions=6,dimension_controls=2,
        max_original_identity_error=max(r['identity_max'] for r in rows),max_zero_P_identity_error=max(r['identity_max'] for r in interventions),
        new_training=False,new_architecture_nominated=False,elapsed_seconds=time.monotonic()-started,pid=os.getpid(),
        peak_sampled_rss_gib=max(r['rss_bytes'] for r in telemetry)/2**30,registration_commit=settings['registration_commit']))
    status(True);print(dict(status='PASS',elapsed=time.monotonic()-started),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','preflight','run'));parser.add_argument('--root',type=Path,required=True);parser.add_argument('--base',type=Path)
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.root,args.base)
    elif args.action=='preflight':preflight(args.root)
    else:run(args.root)
