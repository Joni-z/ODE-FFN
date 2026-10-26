"""Native replay of the placement intervention and all historical references."""
import copy
import json
import os
import sys
import threading
import time as timer
from pathlib import Path

import numpy as np
import torch

ROOT=Path(sys.argv[1]);sys.path.insert(0,str(ROOT/'source'))
import bias_study as study
from bias_core import BiasCore,input_jacobian
from map_core import input_jacobian as plain_jacobian
from local_math import value_jacobian
from teacher_support import TIMES,SNAPSHOTS,load,write,append,sha,signature,assert_equal


def risk(a,b):
    a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    mse=float(np.mean((a-b)**2));energy=float(np.mean(b*b))
    return dict(mse=mse,relative_mse=mse/energy,target_energy=energy)


def compare(a,b):
    for key,value in a.items():np.testing.assert_allclose(value,b[key],rtol=2e-12,atol=2e-14,err_msg=key)


def audit():
    torch.set_num_threads(1);out=ROOT/'output';assert load(out/'status.json')['state']=='COMPLETE'
    ready=load(ROOT/'ready.json')
    for path,digest in {**ready['bindings'],**ready['prior_documents']}.items():assert sha(path)==digest,path
    started=timer.monotonic();stop=threading.Event()
    def monitor():
        while not stop.is_set():append(ROOT/'audit-monitor.jsonl',study.maps.old.observe());stop.wait(5)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    try:
        streams={};rngs={}
        for teacher in study.maps.TEACHERS:
            for j in range(3):
                z,y=study.maps.cache_arrays(study.DATA,'train',j,teacher)
                rng=torch.Generator().manual_seed(0);digests=dict.fromkeys(('indices','input','label'),study.maps.old.EMPTY)
                expected={0:digests.copy()};states={0:rng.get_state()}
                for step in range(1,5001):
                    study.maps.old.consumed(z,y,rng,digests)
                    if step in SNAPSHOTS:expected[step]=digests.copy();states[step]=rng.get_state()
                streams[teacher,j]=expected;rngs[teacher,j]=states
        print(dict(stage='full_streams_replayed',streams=6,batches=30000),flush=True)
        cohort=[];statistics=[];forwards=panels=jacobians=matrices=offsets=0
        maxima=dict(jacobian=0.,offset_gradient=0.)
        for teacher in study.maps.TEACHERS:
            for j in range(3):
                for family in ('w_map',*study.NEW,'swiglu128'):
                    reused=family not in study.NEW
                    folder=study.reference_folder(teacher,family,j) if reused else out/'runs'/f'{teacher}--{family}--{j}'
                    if reused:
                        entry=load(out/'reused'/f'{teacher}--{family}--{j}.json')
                        for name,digest in entry['files_sha256'].items():assert sha(folder/name)==digest,name
                    result=load(folder/'result.json');provenance=load(folder/'provenance.json')
                    assert load(folder/'status.json')['state']=='COMPLETE'
                    module=study.cold(family,j);initial=copy.deepcopy(module.state_dict());assert signature(module)==provenance['initial_signature']
                    costs=study.budget(module)
                    for key in ['parameters','batch1_forward_macs','batch512_forward_macs','batch512_forward_backward_macs']:
                        if key in provenance['costs']:assert costs[key]==provenance['costs'][key]
                    metric_rows=[json.loads(l) for l in (folder/'metrics.jsonl').read_text().splitlines()];assert len(metric_rows)==11
                    global_rng=None
                    for step in SNAPSHOTS:
                        state=torch.load(folder/f'step_{step:07d}.pt',map_location='cpu',weights_only=False)
                        assert state['step']==step and state['consumed_digests']==streams[teacher,j][step]
                        assert torch.equal(state['sampling_rng'],rngs[teacher,j][step])
                        if step==0:assert_equal(initial,state['model']);global_rng=state['torch_rng']
                        else:assert torch.equal(global_rng,state['torch_rng'])
                        for entry in state['optimizer']['state'].values():assert float(entry['step'])==step
                        module.load_state_dict(state['model']);assert_equal(dict(module.named_buffers()),state['runtime_buffers'])
                        z,y=study.maps.cache_arrays(study.DATA,'validation',j,teacher);value=study.maps.old.predict(module,z)
                        assert np.array_equal(value,np.load(folder/f'validation_{step:07d}.npy'))
                        compare(risk(value,y),next(r for r in metric_rows if r['step']==step))
                        if step==5000:compare(risk(value,y),result)
                        if step==0 and family in study.NEW:
                            baseline=study.reference_folder(teacher,'w_map',j)
                            assert np.array_equal(value,np.load(baseline/'validation_0000000.npy'))
                        forwards+=1
                        if family in study.NEW:
                            s=load(folder/f'statistics_{step:07d}.json');b=module.offset.detach().double().numpy()
                            assert np.array_equal(b,np.array(s['offset']))
                            np.testing.assert_allclose(np.linalg.norm(b),s['offset_norm'],rtol=2e-12,atol=2e-14)
                            for name,p in [('matrix1',module.matrix1),('matrix2',module.matrix2)]:
                                np.testing.assert_allclose(np.linalg.svd(p.detach().double().numpy(),compute_uv=False),s['matrices'][name]['singular_values'],rtol=2e-10,atol=2e-12);matrices+=1
                            double=copy.deepcopy(module).double()
                            j0=study.maps.old.native_jacobian(double,torch.zeros(1,64,dtype=torch.float64))[1][0].numpy()
                            sign,logdet=np.linalg.slogdet(j0);assert sign==s['origin_jacobian_sign']
                            np.testing.assert_allclose(logdet,s['origin_jacobian_logabsdet'],rtol=2e-10,atol=2e-12)
                            statistics.append(dict(teacher=teacher,position=j,student=family,step=step,statistics=s));offsets+=1
                    assert signature(module)==result['final_signature'] and result['consumed_digests']==streams[teacher,j][5000]
                    for clock in TIMES:
                        z,y=study.maps.cache_arrays(study.DATA,f'panel-{clock:.1f}',j,teacher);value=study.maps.old.predict(module,z)
                        assert np.array_equal(value,np.load(folder/f'panel_{clock:.1f}.npy'));compare(risk(value,y),next(r for r in result['panels'] if r['time']==clock));panels+=1
                    z,_=study.maps.cache_arrays(study.DATA,'validation',j,teacher);xx=torch.from_numpy(np.array(z[:128],copy=True))
                    _,native=study.maps.old.native_jacobian(module,xx);assert np.array_equal(native.numpy(),np.load(folder/'jacobian.npy'))
                    truth=study.maps.jacobian_target(study.DATA,j,teacher).astype(np.float64)
                    je=float(np.mean(np.sum((native.numpy().astype(np.float64)-truth)**2,axis=(1,2))))
                    np.testing.assert_allclose(je,result['jacobian_error'],rtol=2e-12,atol=2e-14)
                    np.testing.assert_allclose(je/float(np.mean(np.sum(truth**2,axis=(1,2)))),result['relative_jacobian_error'],rtol=2e-12,atol=2e-14)
                    double=copy.deepcopy(module).double();_,native=study.maps.old.native_jacobian(double,xx.double())
                    if family in study.NEW:analytic=input_jacobian(double,xx.double())
                    elif family=='w_map':analytic=plain_jacobian(double,xx.double())
                    else:analytic=value_jacobian(double,xx.double(),'swiglu128')[1]
                    torch.testing.assert_close(native,analytic,rtol=1e-10,atol=1e-11)
                    maxima['jacobian']=max(maxima['jacobian'],float((native-analytic).detach().abs().max()));jacobians+=1
                    if family in study.NEW:
                        x=xx[:13].double();y=torch.cos(torch.arange(x.numel(),dtype=torch.float64)).reshape_as(x)
                        pred=double(x);g=torch.autograd.grad((pred-y).square().mean(),double.offset)[0];up=2*(pred-y)/x.numel()
                        if family=='output_bias':expected=up.sum(0)
                        else:
                            pre=x@double.matrix1.T+double.offset;s=torch.sigmoid(pre)
                            expected=((up@double.matrix2)*(s+pre*s*(1-s))).sum(0)
                        torch.testing.assert_close(g,expected,rtol=1e-10,atol=1e-11)
                        maxima['offset_gradient']=max(maxima['offset_gradient'],float((g-expected).detach().abs().max()))
                    cohort.append(dict(teacher=teacher,student=family,position=j,reused=reused,folder=str(folder),costs=costs))
                    print(dict(stage='cell_replayed',teacher=teacher,position=j,student=family,reused=reused),flush=True)
        assert len(cohort)==24 and len(list((out/'runs').iterdir()))==12
        for path,digest in {**ready['bindings'],**ready['prior_documents']}.items():assert sha(path)==digest,path
        write(ROOT/'audit-cohort.json',cohort);write(ROOT/'audit-statistics.json',statistics)
        receipt=dict(status='PASS',new_cells=12,reused_cells=12,checkpoint_forwards=forwards,endpoint_panels=panels,
                     endpoint_jacobians=jacobians,independently_checked_new_matrix_records=matrices,offset_vectors=offsets,
                     unique_streams=6,replayed_stream_batches=30000,maxima=maxima,input_bindings=len(ready['bindings']),
                     preserved_prior_files=len(ready['prior_documents']),ready_sha256=sha(ROOT/'ready.json'),source_sha256=sha(__file__),
                     cohort_sha256=sha(ROOT/'audit-cohort.json'),statistics_sha256=sha(ROOT/'audit-statistics.json'),
                     elapsed_seconds=timer.monotonic()-started,pid=os.getpid())
        write(ROOT/'audit.json',receipt);print(receipt,flush=True)
    finally:stop.set();watcher.join(timeout=6)


if __name__=='__main__':audit()
