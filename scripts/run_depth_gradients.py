"""Frozen cross-depth gradients on exact matched shared/cloned functions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import psutil
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from synthetic.recurrent_potential import build
from synthetic.depth_gradients import (independent_copies,describe,projected,transferred_alignment,
    block_coordinates,loss_gradients,block_vectors,copy_block_vectors)
from synthetic.population import fingerprint_batch
from synthetic.train import restore_model,load_targets

DATASET='rotated_product_bimodal_16d'
FAMILIES=('ode_k2','swiglu128','swiglu192')
STEPS=(0,1000,2500,5000)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def table(p,rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def input_jacobian(model,x,t):
    xx=x.detach().clone().requires_grad_(True)
    value=model(xx,t)
    jac=torch.stack([torch.autograd.grad(value[:,j].sum(),xx,retain_graph=j+1<16)[0] for j in range(16)],1)
    return value.detach(),jac.detach()


def mean_loss(model,bank):
    values=[]
    with torch.no_grad():
        for i in range(8):
            x,t,y=[torch.from_numpy(bank[key][i]).double() for key in ('state','time','velocity')]
            values.append(float((model(x,t)-y).square().mean()))
    return sum(values)/len(values)


def criterion(p,all_metrics,ga,gb):
    lost_p=p['E1']+p['E2'];lost_all=all_metrics['E1']+all_metrics['E2']
    share=lost_p/lost_all if lost_all>0 else None
    a0=transferred_alignment(ga,gb,(0,))
    a01=transferred_alignment(ga,gb,(0,1))
    passed=(p['retained'] is not None and p['retained']<=.5 and share is not None and share>=.5
        and p['linear_share_of_lost'] is not None and p['linear_share_of_lost']>=.7
        and a0 is not None and a01 is not None and a01>=.2 and a01-a0>=.05)
    return dict(retained_P=p['retained'],P_share_of_blocked=share,linear_share_of_lost_P=p['linear_share_of_lost'],
                common_transfer=a0,affine_transfer=a01,passes=bool(passed))


def main(recurrent_runtime,old_recurrent,output):
    torch.set_num_threads(1)
    manifest=load(ROOT/'source-sha256.json')
    for name,value in manifest.items():assert sha(ROOT/name)==value
    assert load(recurrent_runtime/'output/status.json')['state']=='COMPLETE'
    for old_source in (recurrent_runtime/'source',Path(load(old_recurrent/'output-product/provenance.json')['source_root'])):
        for name,value in load(old_source/'source-sha256.json').items():assert sha(old_source/name)==value
    for name in ('synthetic/models.py','synthetic/recurrent_depth.py','synthetic/recurrent_potential.py','synthetic/conservative_readout.py','synthetic/train.py','synthetic/population.py'):
        assert sha(ROOT/name)==sha(recurrent_runtime/'source'/name)
    output.mkdir(parents=True,exist_ok=False)
    bindings={};rows=[];summaries=[];transfers=[];identities=[];finite=[];selection=[]
    process=psutil.Process();started=time.monotonic();telemetry=[]
    def progress(**kw):
        rec=dict(elapsed_seconds=time.monotonic()-started,pid=process.pid,rss_MiB=process.memory_info().rss/2**20,
                 available_GiB=psutil.virtual_memory().available/2**30,**kw)
        telemetry.append(rec);write(output/'status.json',dict(state='RUNNING',**rec))
        with (output/'telemetry.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
        print(json.dumps(rec),flush=True)
    data=Path(load(recurrent_runtime/'output/study.json')['data_roots'][DATASET])
    data_manifest=load(data/'manifest.json')
    assert data_manifest['datasets'][DATASET]['source']=='standard_normal'
    assert data_manifest['datasets'][DATASET]['coupling']=='independent'
    path=data/DATASET/'train.npz';bindings[str(path)]=sha(path)
    assert sha(path)==data_manifest['files'][DATASET+'/train.npz']['sha256']
    bindings[str(data/'manifest.json')]=sha(data/'manifest.json')
    bindings[str(recurrent_runtime/'output/study.json')]=sha(recurrent_runtime/'output/study.json')
    targets=torch.from_numpy(load_targets(path,16))
    rng=torch.Generator().manual_seed(100000);digest=hashlib.sha256(b'').hexdigest()
    expected={0:(rng.get_state().clone(),digest)}
    for step in range(1,5001):
        yy=targets[torch.randint(len(targets),(512,),generator=rng)]
        z=torch.randn(yy.shape,generator=rng);tt=torch.rand((512,1),generator=rng)
        digest=fingerprint_batch(digest,yy,z,tt)
        if step in STEPS:expected[step]=(rng.get_state().clone(),digest)
    banks={}
    (output/'banks').mkdir()
    for step in STEPS:
        rng.set_state(expected[step][0]);rolling=expected[step][1]
        values={key:[] for key in ('target','source','time','state','velocity')};hashes=[]
        for batch in range(16):
            yy=targets[torch.randint(len(targets),(512,),generator=rng)]
            z=torch.randn(yy.shape,generator=rng);tt=torch.rand((512,1),generator=rng)
            xx=(1-tt)*z+tt*yy;velocity=yy-z
            rolling=fingerprint_batch(rolling,yy,z,tt);hashes.append(rolling)
            for key,v in (('target',yy),('source',z),('time',tt),('state',xx),('velocity',velocity)):values[key].append(v.numpy())
        arrays={key:np.stack(v) for key,v in values.items()}
        banks[step]=[{key:v[:8] for key,v in arrays.items()},{key:v[8:] for key,v in arrays.items()}]
        np.savez_compressed(output/'banks'/f'step-{step}.npz',**arrays)
        write(output/'banks'/f'step-{step}-hashes.json',hashes)
    maxima=dict(value=0.,input_J=0.,gradient=0.,common=0.,finite=0.)
    for family in FAMILIES:
        for readout in ('direct','gradient'):
            new=readout=='gradient' or family=='swiglu128'
            arm=family+'_shared_'+readout
            actual=arm if new else family+'_shared_final'
            folder=(recurrent_runtime/'output' if new else old_recurrent/'output-product')/'runs'/DATASET/actual
            assert load(folder/'status.json')['state']=='COMPLETE'
            for step in STEPS:
                progress(arm=arm,step=step)
                p=folder/f'step_{step:07d}.pt';bindings[str(p)]=sha(p)
                state=torch.load(p,map_location='cpu',weights_only=False)
                assert state['step']==step
                assert torch.equal(state['data_rng'],expected[step][0]) and state['input_stream_digest']==expected[step][1]
                shared=build(16,family,'shared_'+readout).double().eval()
                restore_model(shared if new else shared.raw,state)
                cloned=independent_copies(shared)
                names,sizes,groups=block_coordinates(shared,family)
                coordinate_path=output/(family+'-coordinates.json')
                coordinates=dict(names=names,sizes=sizes,groups=groups)
                if coordinate_path.exists():assert load(coordinate_path)==coordinates
                else:write(coordinate_path,coordinates)
                xx,tt=[torch.from_numpy(banks[step][0][key][0,:3]).double() for key in ('state','time')]
                av,aj=input_jacobian(shared,xx,tt);bv,bj=input_jacobian(cloned,xx,tt)
                torch.testing.assert_close(av,bv,rtol=1e-10,atol=1e-11)
                torch.testing.assert_close(aj,bj,rtol=1e-9,atol=1e-10)
                maxima['input_J']=max(maxima['input_J'],float((aj-bj).abs().max()))
                bank_gradients=[];shared_gradients=[];batch_loss=[]
                for bank_index,bank in enumerate(banks[step]):
                    gradients=[];sg=[];losses=[]
                    for batch in range(8):
                        x,t,y=[torch.from_numpy(bank[key][batch]).double() for key in ('state','time','velocity')]
                        a,la,ga,ca=loss_gradients(shared,x,t,y)
                        b,lb,gb,cb=loss_gradients(cloned,x,t,y)
                        torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-11)
                        torch.testing.assert_close(ga[0],gb.sum(0),rtol=1e-7,atol=1e-9)
                        ce=0.
                        assert ca.keys()==cb.keys()
                        for name in ca:
                            torch.testing.assert_close(ca[name],cb[name],rtol=1e-7,atol=1e-9)
                            ce=max(ce,float((ca[name]-cb[name]).abs().max()))
                        maxima['value']=max(maxima['value'],float((a-b).abs().max()))
                        ge=float((ga[0]-gb.sum(0)).abs().max())
                        maxima['gradient']=max(maxima['gradient'],ge);maxima['common']=max(maxima['common'],ce)
                        identities.append(dict(arm=arm,step=step,bank='AB'[bank_index],batch=batch,
                            loss_shared=la,loss_cloned=lb,gradient_sum_error=ge,common_gradient_error=ce))
                        gradients.append(gb);sg.append(ga[0]);losses.append(la)
                        for group,index in groups.items():
                            rows.append(dict(arm=arm,family=family,readout=readout,step=step,bank='AB'[bank_index],batch=batch,
                                             group=group,**describe(gb[:,index])))
                    bank_gradients.append(torch.stack(gradients));shared_gradients.append(torch.stack(sg));batch_loss.append(losses)
                case=output/'gradients'/arm;case.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(case/f'step-{step}.npz',independent=torch.stack(bank_gradients).numpy(),
                    shared=torch.stack(shared_gradients).numpy(),loss=np.array(batch_loss))
                ga,gb=[v.mean(0) for v in bank_gradients]
                by_group={}
                for group,index in groups.items():
                    pa,pb=describe(ga[:,index]),describe(gb[:,index]);by_group[group]=pa
                    for label,metrics in (('A',pa),('B',pb)):
                        summaries.append(dict(arm=arm,family=family,readout=readout,step=step,bank=label,group=group,**metrics))
                    transfers.append(dict(arm=arm,family=family,readout=readout,step=step,group=group,
                        common_alignment=transferred_alignment(ga[:,index],gb[:,index],(0,)),
                        affine_alignment=transferred_alignment(ga[:,index],gb[:,index],(0,1))))
                if family=='ode_k2':
                    index=groups['P']
                    select=criterion(by_group['P'],by_group['all'],ga[:,index],gb[:,index])
                    consistency=[]
                    for batch in range(8):
                        g=bank_gradients[0][batch]
                        consistency.append(criterion(describe(g[:,index]),describe(g),g[:,index],gb[:,index])['passes'])
                    selection.append(dict(arm=arm,step=step,bankA_batches_passing=sum(consistency),**select))
                if step==5000:
                    index=groups['P'] if family=='ode_k2' else groups['FFN']
                    original=block_vectors(cloned).clone()
                    for mode,modes in (('common',(0,)),('affine',(0,1))):
                        sub=projected(ga[:,index],modes)
                        if float(sub.norm())==0:continue
                        direction=torch.zeros_like(ga);direction[:,index]=-sub/sub.norm()
                        expected_derivative=float((gb*direction).sum());eps=1e-5
                        try:
                            copy_block_vectors(cloned,original+eps*direction);plus=mean_loss(cloned,banks[step][1])
                            copy_block_vectors(cloned,original-eps*direction);minus=mean_loss(cloned,banks[step][1])
                        finally:copy_block_vectors(cloned,original)
                        numeric=(plus-minus)/(2*eps)
                        error=abs(numeric-expected_derivative)/max(1.,abs(expected_derivative))
                        assert error<1e-6,(arm,mode,error)
                        maxima['finite']=max(maxima['finite'],error)
                        finite.append(dict(arm=arm,mode=mode,bankB_analytic=expected_derivative,bankB_finite=numeric,relative_error=error))
                    torch.testing.assert_close(block_vectors(cloned),original,rtol=0,atol=0)
    selected=[r for r in selection if r['arm']=='ode_k2_shared_gradient' and r['step'] in (2500,5000)]
    assert len(selected)==2
    eligible=all(r['passes'] for r in selected)
    decision=dict(eligible_for_separate_P_depth_protocol=eligible,late_P_cases=selected,accepted_architecture=False,
        action='register one matched affine-depth P cold experiment' if eligible else 'do not launch P-depth training or select another mode/group')
    table(output/'batch-modes.csv',rows);table(output/'mean-modes.csv',summaries)
    table(output/'transfers.csv',transfers);table(output/'identities.csv',identities)
    table(output/'finite-directions.csv',finite);table(output/'P-selection.csv',selection)
    write(output/'decision.json',decision);write(output/'input-bindings.json',bindings)
    for name,value in manifest.items():assert sha(ROOT/name)==value
    for name,value in bindings.items():assert sha(name)==value
    audit=dict(status='PASS',learned_states=24,matched_batch_gradient_cases=len(identities),
        complete_input_J_probes=24,bank_mean_rows=len(summaries),finite_direction_checks=len(finite),maximum_errors=maxima,
        source_manifest_sha256=sha(ROOT/'source-sha256.json'),sampled_RSS_max_MiB=max(r['rss_MiB'] for r in telemetry),
        minimum_available_GiB=min(r['available_GiB'] for r in telemetry))
    write(output/'audit.json',audit);write(output/'status.json',dict(state='COMPLETE',elapsed_seconds=time.monotonic()-started))
    print(json.dumps(dict(audit=audit,decision=decision)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recurrent-runtime',type=Path,required=True);p.add_argument('--old-recurrent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    main(a.recurrent_runtime.resolve(),a.old_recurrent.resolve(),a.output.resolve())
