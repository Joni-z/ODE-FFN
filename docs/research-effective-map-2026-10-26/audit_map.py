"""Independent native replay of new paired cores and historical controls."""
import copy
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

ROOT=Path(sys.argv[1])
sys.path.insert(0,str(ROOT/'source'))
import map_study as study
from map_core import MapCore,direct_copy,sequential,input_jacobian
from teacher_support import TIMES,SNAPSHOTS,load,write,append,sha,signature,assert_equal
from local_math import value_jacobian


def risk(a,b):
    a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    mse=float(np.vdot((a-b).ravel(),(a-b).ravel()).real/a.size)
    energy=float(np.vdot(b.ravel(),b.ravel()).real/b.size)
    return dict(mse=mse,relative_mse=mse/energy,target_energy=energy)


def compare_metrics(a,b):
    for key,value in a.items():np.testing.assert_allclose(value,b[key],rtol=2e-12,atol=2e-14,err_msg=key)


def validate_matrices(module,stats):
    if isinstance(module,MapCore):
        first,second=module.effective_matrices()
        values={'effective1':first,'effective2':second}
        if module.coordinates=='p_map':values.update(P1=module.matrix1,P2=module.matrix2)
    else:values={'w12':module.w12.weight,'w3':module.w3.weight}
    for name,value in values.items():
        p=value.detach().double().numpy();row=stats['matrices'][name]
        np.testing.assert_allclose(np.linalg.svd(p,compute_uv=False),row['singular_values'],rtol=2e-10,atol=2e-12)
        if p.shape[0]==p.shape[1]:
            sign,logdet=np.linalg.slogdet(p)
            assert sign==row['determinant_sign']
            np.testing.assert_allclose(logdet,row['logabsdet'],rtol=2e-10,atol=2e-12)
            np.testing.assert_allclose(np.linalg.eigvalsh((p+p.T)/2).min(),row['symmetric_minimum'],rtol=2e-10,atol=2e-12)
            if name.startswith('P'):
                eig=np.linalg.eigvals(p);mods=np.abs(1+.5*(eig[:,None]+eig[None,:]))
                np.testing.assert_allclose(mods.min(),row['differential_minimum_eigenmodulus'],rtol=2e-10,atol=2e-12)
                np.testing.assert_allclose(mods.max()/mods.min(),row['differential_condition_lower_bound'],rtol=2e-10,atol=2e-12)
                actual=values['effective'+name[-1]].detach().double().numpy()
                difference=actual-(np.eye(64)+p+.5*p@p)
                np.testing.assert_allclose(np.linalg.svd(difference,compute_uv=False)[0],row['effective_roundoff_spectral_norm'],rtol=2e-6,atol=2e-13)
    z=torch.zeros(1,64,dtype=torch.float64)
    double=copy.deepcopy(module).double()
    jac=input_jacobian(double,z)[0] if isinstance(module,MapCore) else value_jacobian(double,z,'swiglu128')[1][0]
    # For P coordinates, matrix metadata describes the executed float32 effective map.
    if isinstance(module,MapCore):
        a,b=module.effective_matrices();jac=.5*b.detach().double()@a.detach().double()
    p=jac.detach().numpy();sign,logdet=np.linalg.slogdet(p)
    assert sign==stats['origin_jacobian']['determinant_sign']
    np.testing.assert_allclose(logdet,stats['origin_jacobian']['logabsdet'],rtol=2e-10,atol=2e-12)
    np.testing.assert_allclose(np.linalg.svd(p,compute_uv=False)[-1],stats['origin_jacobian']['minimum_singular_value'],rtol=2e-10,atol=2e-12)
    return len(values)


def audit():
    torch.set_num_threads(1);out=ROOT/'output';cache=out/'cache'
    assert load(out/'status.json')['state']=='COMPLETE'
    ready=load(ROOT/'ready.json')
    for path,digest in {**ready['bindings'],**ready['prior_documents']}.items():assert sha(path)==digest,path
    started=time.monotonic();stop=threading.Event()
    def monitor():
        while not stop.is_set():append(ROOT/'audit-monitor.jsonl',study.old.observe());stop.wait(5)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    try:
        metadata=load(cache/'metadata.json')
        for name,digest in metadata['files_sha256'].items():assert sha(cache/name)==digest,name
        for path,digest in metadata['external_arrays'].items():assert sha(path)==digest,path
        target_rows=0
        for j in range(3):
            module=study.target('flow_map',j);before=signature(module)
            assert before==metadata['flow_target_signatures'][str(j)]
            for split in study.SPLITS:
                z,y=study.cache_arrays(cache,split,j,'flow_map')
                assert np.array_equal(study.old.predict(module,z),y);target_rows+=len(z)
            z,_=study.cache_arrays(cache,'validation',j,'flow_map')
            _,jac=study.old.native_jacobian(module,torch.from_numpy(np.array(z[:128],copy=True)))
            assert np.array_equal(jac.numpy(),study.jacobian_target(cache,j,'flow_map'))
            assert before==signature(module)
        expected={};sampling={}
        for teacher in study.TEACHERS:
            for j in range(3):
                z,y=study.cache_arrays(cache,'train',j,teacher);rng=torch.Generator().manual_seed(0)
                digests=dict.fromkeys(('indices','input','label'),study.old.EMPTY)
                states={0:digests.copy()};rngs={0:rng.get_state()}
                for step in range(1,5001):
                    study.old.consumed(z,y,rng,digests)
                    if step in SNAPSHOTS:states[step]=digests.copy();rngs[step]=rng.get_state()
                expected[teacher,j]=states;sampling[teacher,j]=rngs
        print(dict(stage='targets_and_streams_replayed',target_rows=target_rows,unique_streams=6,batches=30000),flush=True)
        cohort=[];matrix_records=[];forwards=panels=jacobians=matrix_count=adjoints=0;maxima=dict(forward=0.,jacobian=0.,adjoint=0.)
        for teacher in study.TEACHERS:
            for j in range(3):
                for family in study.FAMILIES:
                    reused=teacher=='swiglu128' and family=='swiglu128'
                    folder=study.reused_folder(j) if reused else out/'runs'/f'{teacher}--{family}--{j}'
                    if reused:
                        reuse=load(out/'reused'/f'position-{j}.json')
                        for name,digest in reuse['files_sha256'].items():assert sha(folder/name)==digest,name
                    result=load(folder/'result.json');provenance=load(folder/'provenance.json')
                    assert load(folder/'status.json')['state']=='COMPLETE'
                    module=study.cold(family,j);initial=copy.deepcopy(module.state_dict())
                    assert signature(module)==provenance['initial_signature']
                    costs=study.compute_budget(module)
                    if not reused:assert costs==provenance['costs']
                    rows=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()];assert len(rows)==11
                    global_rng=None
                    for step in SNAPSHOTS:
                        state=torch.load(folder/f'step_{step:07d}.pt',map_location='cpu',weights_only=False)
                        assert state['step']==step and state['consumed_digests']==expected[teacher,j][step]
                        assert torch.equal(state['sampling_rng'],sampling[teacher,j][step])
                        if step==0:assert_equal(initial,state['model']);global_rng=state['torch_rng']
                        else:assert torch.equal(state['torch_rng'],global_rng)
                        for parameter in state['optimizer']['state'].values():assert float(parameter['step'])==step
                        module.load_state_dict(state['model']);assert_equal(dict(module.named_buffers()),state['runtime_buffers'])
                        z,y=study.cache_arrays(cache,'validation',j,teacher);value=study.old.predict(module,z)
                        assert np.array_equal(value,np.load(folder/f'validation_{step:07d}.npy'))
                        measure=risk(value,y);compare_metrics(measure,next(r for r in rows if r['step']==step))
                        if step==5000:compare_metrics(measure,result)
                        forwards+=1
                        stats=study.matrix_stats(module) if reused else load(folder/f'matrices_{step:07d}.json')
                        matrix_count+=validate_matrices(module,stats)
                        matrix_records.append(dict(teacher=teacher,position=j,student=family,step=step,reused=reused,stats=stats))
                    assert signature(module)==result['final_signature']
                    assert result['consumed_digests']==expected[teacher,j][5000]
                    for clock in TIMES:
                        z,y=study.cache_arrays(cache,f'panel-{clock:.1f}',j,teacher);value=study.old.predict(module,z)
                        assert np.array_equal(value,np.load(folder/f'panel_{clock:.1f}.npy'))
                        compare_metrics(risk(value,y),next(r for r in result['panels'] if r['time']==clock));panels+=1
                    z,_=study.cache_arrays(cache,'validation',j,teacher);xx=torch.from_numpy(np.array(z[:128],copy=True))
                    _,jac=study.old.native_jacobian(module,xx);assert np.array_equal(jac.numpy(),np.load(folder/'jacobian.npy'))
                    truth=study.jacobian_target(cache,j,teacher).astype(np.float64)
                    je=float(np.mean(np.sum((jac.numpy().astype(np.float64)-truth)**2,axis=(1,2))))
                    np.testing.assert_allclose(je,result['jacobian_error'],rtol=2e-12,atol=2e-14)
                    np.testing.assert_allclose(je/float(np.mean(np.sum(truth**2,axis=(1,2)))),result['relative_jacobian_error'],rtol=2e-12,atol=2e-14)
                    double=copy.deepcopy(module).double();value,native_j=study.old.native_jacobian(double,xx.double())
                    if isinstance(module,MapCore):reference=sequential(double,xx.double());analytic=input_jacobian(double,xx.double())
                    else:reference,analytic=value_jacobian(double,xx.double(),'swiglu128')
                    torch.testing.assert_close(value,reference,rtol=1e-10,atol=1e-11);torch.testing.assert_close(native_j,analytic,rtol=1e-10,atol=1e-11)
                    maxima['forward']=max(maxima['forward'],float((value-reference).detach().abs().max()));maxima['jacobian']=max(maxima['jacobian'],float((native_j-analytic).detach().abs().max()));jacobians+=1
                    if family=='p_map':
                        direct=direct_copy(double);target=torch.cos(torch.arange(xx.numel(),dtype=torch.float64)).reshape_as(xx)
                        gp=torch.autograd.grad((double(xx.double())-target).square().mean(),tuple(double.parameters()))
                        gw=torch.autograd.grad((direct(xx.double())-target).square().mean(),tuple(direct.parameters()))
                        for p,g,h in zip((double.matrix1,double.matrix2),gp,gw):
                            expected_gradient=h+.5*(p.T@h+h@p.T)
                            torch.testing.assert_close(g,expected_gradient,rtol=1e-10,atol=1e-11)
                            maxima['adjoint']=max(maxima['adjoint'],float((g-expected_gradient).detach().abs().max()));adjoints+=1
                    cohort.append(dict(teacher=teacher,student=family,position=j,folder=str(folder),reused=reused,costs=costs))
                    print(dict(stage='cell_replayed',teacher=teacher,position=j,student=family,reused=reused),flush=True)
        for j in range(3):
            p=study.cold('p_map',j);w=study.cold('w_map',j)
            for family,module in [('p_map',p),('w_map',w)]:module.load_state_dict(torch.load(out/'runs'/f'flow_map--{family}--{j}'/'step_0000000.pt',map_location='cpu',weights_only=False)['model'])
            assert all(torch.equal(a,b) for a,b in zip(p.effective_matrices(),w.effective_matrices()))
        assert len(list((out/'runs').iterdir()))==15 and len(cohort)==18
        for path,digest in {**ready['bindings'],**ready['prior_documents']}.items():assert sha(path)==digest,path
        write(ROOT/'audit-cohort.json',cohort);write(ROOT/'audit-matrices.json',matrix_records)
        receipt=dict(status='PASS',new_cells=15,reused_cells=3,checkpoint_forwards=forwards,endpoint_time_panels=panels,
                     endpoint_jacobians=jacobians,independently_checked_matrix_records=matrix_count,adjoint_matrix_checks=adjoints,
                     unique_consumed_streams=6,replayed_stream_batches=30000,flow_target_rows=target_rows,
                     maxima=maxima,input_bindings=len(ready['bindings']),preserved_prior_files=len(ready['prior_documents']),
                     ready_sha256=sha(ROOT/'ready.json'),source_sha256=sha(__file__),cohort_sha256=sha(ROOT/'audit-cohort.json'),
                     matrices_sha256=sha(ROOT/'audit-matrices.json'),elapsed_seconds=time.monotonic()-started,pid=os.getpid())
        write(ROOT/'audit.json',receipt);print(receipt,flush=True)
    finally:stop.set();watcher.join(timeout=6)


if __name__=='__main__':audit()
